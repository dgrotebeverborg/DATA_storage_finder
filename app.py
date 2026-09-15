import json
import logging
import math
import threading
import time
from logic.matching import match_with_reason, sanitize_for_json

from flask import Flask, request, jsonify, render_template, session
from werkzeug.middleware.proxy_fix import ProxyFix

import os
from collections import defaultdict, deque


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# The chatbot needs langchain/chromadb plus a running Ollama instance;
# keep it opt-in so the rest of the site works without that setup.
CHATBOT_ENABLED = _env_flag("CHATBOT_ENABLED", False)

if CHATBOT_ENABLED:
    from chatbot.storage_rag import (
        ask_storage_question_structured,
        detect_question_language,
        clarification_prompt,
    )
    from chatbot.metrics_dashboard import compute_metrics_summary

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
app.secret_key = os.environ.get("SECRET_KEY", "dev-insecure-change-me")
app.config["SESSION_COOKIE_SECURE"] = _env_flag("SESSION_COOKIE_SECURE", True)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")

trusted_hosts = os.environ.get("TRUSTED_HOSTS", "").strip()
if trusted_hosts:
    app.config["TRUSTED_HOSTS"] = [
        host.strip() for host in trusted_hosts.split(",") if host.strip()
    ]

CHAT_RATE_LIMIT_WINDOW_SECONDS = int(
    os.environ.get("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60")
)
CHAT_RATE_LIMIT_REQUESTS = int(os.environ.get("CHAT_RATE_LIMIT_REQUESTS", "12"))
_chat_rate_limit_lock = threading.Lock()
_chat_rate_limit_state: dict[str, deque] = defaultdict(deque)

logger = logging.getLogger(__name__)

CLARIFICATION_ORDER = ["sensitivity", "collaboration", "volume"]


# Load data on startup
DATA_PATH = os.path.join("data", "storage_data_2.json")
with open(DATA_PATH) as f:
    storage_data = json.load(f)

# Combine active and preservation into one list
# all_solutions = storage_data["active_storage"] + storage_data["preservation_storage"]
all_solutions = storage_data["storage_data_2"]
# Safe lowercase helper
def safe_lower(val):
    return str(val).lower() if val and not isinstance(val, float) else ""


def _to_bool_or_text(user_input: str):
    txt = (user_input or "").strip().lower()
    yes_vals = {"yes", "y", "ja", "zeker", "correct", "klopt"}
    no_vals = {"no", "n", "nee", "nope"}
    if txt in yes_vals:
        return "yes"
    if txt in no_vals:
        return "no"
    return txt


def _next_clarification_key(answered: dict) -> str | None:
    for key in CLARIFICATION_ORDER:
        if key not in answered:
            return key
    return None


def _looks_like_recommendation(question: str) -> bool:
    q = (question or "").lower()
    patterns = (
        "welke storage", "welke opslag", "wat moet ik kiezen", "opslag kiezen", "advies",
        "which storage", "what should i use", "choose storage", "storage advice", "recommend",
        "which one should i choose", "which one to choose",
    )
    return any(p in q for p in patterns)


def _is_short_followup_token(user_input: str) -> bool:
    t = (user_input or "").strip().lower()
    return t in {"yes", "no", "ja", "nee", "choose", "vergelijken", "compare", "beleid", "policy", "privacy", "beleid/privacy"}


def _looks_like_clarification_question(text: str) -> bool:
    t = (text or "").lower()
    markers = (
        "sensitive data", "gevoelige data", "gdpr",
        "multiple people", "meerdere mensen",
        "gb or tb", "gb of tb", "data volume", "orde van grootte",
        "what do you want to do exactly", "waar gaat je vraag precies over",
    )
    return any(m in t for m in markers)


def _find_latest_recommendation_question(history: list) -> str | None:
    for q, _a in reversed(history):
        if _looks_like_recommendation(q):
            return q
    return None


def _clarification_key_from_text(text: str) -> str | None:
    t = (text or "").lower()
    if "sensitive data" in t or "gevoelige data" in t or "gdpr" in t or "persoonsgegevens" in t:
        return "sensitivity"
    if "multiple people" in t or "meerdere mensen" in t or "access at the same time" in t:
        return "collaboration"
    if "gb or tb" in t or "gb of tb" in t or "data volume" in t or "orde van grootte" in t:
        return "volume"
    return None


def _client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _chat_rate_limit_remaining(ip: str) -> tuple[bool, int]:
    now = time.time()
    with _chat_rate_limit_lock:
        bucket = _chat_rate_limit_state[ip]
        while bucket and now - bucket[0] >= CHAT_RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= CHAT_RATE_LIMIT_REQUESTS:
            retry_after = max(
                1, int(CHAT_RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0]))
            )
            return False, retry_after

        bucket.append(now)
        return True, 0

@app.route("/chat/ask", methods=["POST"])
def ask_rag():
    if not CHATBOT_ENABLED:
        return jsonify({"response": "The chat assistant is currently disabled."}), 503

    started_at = time.time()
    client_ip = _client_ip()
    allowed, retry_after = _chat_rate_limit_remaining(client_ip)
    if not allowed:
        logger.warning("chat_rate_limited ip=%s retry_after=%s", client_ip, retry_after)
        response = jsonify({
            "response": "Too many chat requests. Please wait a moment and try again."
        })
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "Please enter a message."})

        if "chat_history" not in session:
            session["chat_history"] = []
        if "chat_lang" not in session:
            session["chat_lang"] = detect_question_language(user_input, [])

        limited_history = session["chat_history"][-2:]
        lang = session.get("chat_lang", "en")

        # Lock session language by first meaningful message.
        if len(session["chat_history"]) == 0:
            lang = detect_question_language(user_input, [])
            session["chat_lang"] = lang

        # Explicit language switch commands.
        if "in english" in user_input.lower() or "please english" in user_input.lower():
            session["chat_lang"] = "en"
            session["chat_history"].append((user_input, "Sure. I will continue in English."))
            return jsonify({
                "response": "Sure. I will continue in English.",
                "citations": [],
                "needs_clarification": False,
                "clarification_question": None,
                "confidence": 1.0,
                "route": "language_switch",
                "used_docs": 0,
            })
        if "in dutch" in user_input.lower() or "nederlands" in user_input.lower():
            session["chat_lang"] = "nl"
            session["chat_history"].append((user_input, "Prima. Ik ga verder in het Nederlands."))
            return jsonify({
                "response": "Prima. Ik ga verder in het Nederlands.",
                "citations": [],
                "needs_clarification": False,
                "clarification_question": None,
                "confidence": 1.0,
                "route": "language_switch",
                "used_docs": 0,
            })

        pending = session.get("pending_clarification")
        if not pending and _is_short_followup_token(user_input) and session["chat_history"]:
            last_bot = session["chat_history"][-1][1]
            if _looks_like_clarification_question(last_bot):
                base_q = _find_latest_recommendation_question(session["chat_history"]) or (session["chat_history"][-1][0])
                lang_now = session.get("chat_lang", lang)
                next_key = _clarification_key_from_text(last_bot) or "sensitivity"
                pending = {
                    "base_question": base_q,
                    "lang": lang_now,
                    "answers": {},
                    "next_key": next_key,
                }
                session["pending_clarification"] = pending

        if pending:
            answers = pending.get("answers", {})
            current_key = pending.get("next_key")
            if current_key in CLARIFICATION_ORDER:
                answers[current_key] = _to_bool_or_text(user_input)

            next_key = _next_clarification_key(answers)
            if next_key:
                pending["answers"] = answers
                pending["next_key"] = next_key
                session["pending_clarification"] = pending
                next_q = clarification_prompt(next_key, pending.get("lang", lang))
                session["chat_history"].append((user_input, next_q))
                return jsonify({
                    "response": next_q,
                    "citations": [],
                    "needs_clarification": True,
                    "clarification_question": next_q,
                    "confidence": 0.0,
                    "route": "clarification_state",
                    "used_docs": 0,
                })

            # Build one enriched query and continue to RAG without re-entering clarification.
            base_q = pending.get("base_question", "")
            enriched = (
                f"{base_q}\n"
                f"Constraints: sensitive_data={answers.get('sensitivity')}; "
                f"collaboration={answers.get('collaboration')}; "
                f"volume={answers.get('volume')}"
            )
            session.pop("pending_clarification", None)
            result = ask_storage_question_structured(
                enriched,
                limited_history,
                user_language=pending.get("lang", lang),
                latest_topic=session.get("last_topic"),
                skip_clarification=True,
            )
        else:
            result = ask_storage_question_structured(
                user_input,
                limited_history,
                user_language=lang,
                latest_topic=session.get("last_topic"),
            )

        if result.detected_topic:
            session["last_topic"] = result.detected_topic

        session["chat_history"].append((user_input, result.response))

        if result.needs_clarification:
            if _looks_like_recommendation(user_input):
                next_key = _clarification_key_from_text(result.clarification_question or "") or "sensitivity"
                session["pending_clarification"] = {
                    "base_question": user_input,
                    "lang": lang,
                    "answers": {},
                    "next_key": next_key,
                }
            else:
                session.pop("pending_clarification", None)
        else:
            session.pop("pending_clarification", None)

        return jsonify({
            "response": result.response,
            "citations": result.citations,
            "needs_clarification": result.needs_clarification,
            "clarification_question": result.clarification_question,
            "confidence": result.confidence,
            "route": result.route,
            "used_docs": result.used_docs,
        })

    except Exception as e:
        logger.exception("chat_request_failed ip=%s", client_ip)
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"⚠️ Error: {str(e)}"}), 500
    finally:
        duration_ms = int((time.time() - started_at) * 1000)
        logger.info("chat_request_completed ip=%s duration_ms=%s", client_ip, duration_ms)


@app.route("/")
def index():
    return render_template("index.html", solutions=all_solutions)

@app.route("/solution/<name>")
def solution_detail(name):
    solution = next((s for s in all_solutions if s["name"] == name), None)
    if not solution:
        return "Solution not found", 404

    # Maak een shallow copy en vervang alleen NaN door lege string
    cleaned = {}
    for k, v in solution.items():
        if isinstance(v, float) and math.isnan(v):
            cleaned[k] = ""
        else:
            cleaned[k] = v  # ✅ laat dicts/lists zoals 'categories' intact

    return render_template("solution_detail.html", solution=cleaned)



@app.route("/compare")
def compare():
    names = request.args.getlist("names")
    selected = [s for s in all_solutions if s.get("name") in names]

    category_order = [
        "General Information",
        "Technical",
        "Security & availability",
        "Privacy",
        "Financial",
        "Support",
        "UDCC Recommendation"
    ]

    ignore_fields = {"name", "categories", "subcategories"}
    # Optioneel: descriptions niet vergelijken/tonen in compare
    def is_ignored(field: str) -> bool:
        if field in ignore_fields:
            return True
        if field.endswith("_description"):
            return True
        return False

    # 1) Union van alle velden in de selectie
    all_fields = set()
    for sol in selected:
        for k in sol.keys():
            if not is_ignored(k):
                all_fields.add(k)

    # 2) Category/subcategory bepalen per veld (kijk naar alle solutions)
    def pick_category(field: str) -> str:
        for sol in selected:
            cats = sol.get("categories", {}) or {}
            c = (cats.get(field) or "").strip()
            if c and c.lower() != "nan" and c != "ID":
                return c
        return "General Information"

    def pick_subcategory(field: str) -> str:
        for sol in selected:
            subs = sol.get("subcategories", {}) or {}
            sc = (subs.get(field) or "").strip()
            if sc and sc.lower() != "nan":
                return sc
        return ""

    def pretty_field(field: str) -> str:
        return (field
                .replace("_", " ")
                .replace("(", "")
                .replace(")", "")
                .replace("-", " ")
                .title())

    # 3) Normalize voor verschil-detectie (missing => "Not known")
    def normalize_value(v):
        if v is None:
            return "Not known"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "Not known"
        if isinstance(v, list):
            return " | ".join(str(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        s = str(v).strip()
        return s if s else "Not known"

    # 4) diff_fields bepalen
    diff_fields = set()
    for field in all_fields:
        vals = [normalize_value(sol.get(field)) for sol in selected]
        if len(set(vals)) > 1:
            diff_fields.add(field)

    # 5) fields_by_category bouwen (vaste rijvolgorde)
    fields_by_category = {c: [] for c in category_order}
    # ook eventuele categorieën buiten category_order
    extra_categories = {}

    for f in sorted(all_fields, key=lambda x: pretty_field(x).lower()):
        cat = pick_category(f)
        item = {
            "field": f,
            "pretty": pretty_field(f),
            "subcat": pick_subcategory(f),
        }
        if cat in fields_by_category:
            fields_by_category[cat].append(item)
        else:
            extra_categories.setdefault(cat, []).append(item)

    return render_template(
        "compare.html",
        solutions=selected,
        category_order=category_order,
        fields_by_category=fields_by_category,
        extra_categories=extra_categories,
        diff_fields=diff_fields,
    )


@app.route("/wizard")
def wizard():
    return render_template("wizard.html")

@app.route("/wizard/results")

def wizard_results():
    phase = request.args.get("phase")
    sensitive = request.args.get("sensitive")
    collab = request.args.get("collab")
    volume = request.args.get("volume")

    candidates = storage_data["storage_data_2"]

    matches = []
    non_matches = []

    for s in candidates:
        ok, reasons = match_with_reason(s, phase, sensitive, collab, volume)

        s_copy = s.copy()
        if ok:
            matches.append(s_copy)
        else:
            s_copy["reasons"] = reasons
            non_matches.append(s_copy)

    return render_template("wizard_results.html", matches=matches, non_matches=non_matches)


@app.route("/api/filter", methods=["POST"])
def api_filter():
    data = request.get_json()
    phase = data.get("phase")
    sensitive = data.get("sensitive")
    collab = data.get("collab")
    volume = data.get("volume")

    # Alles komt nu uit storage_data_2
    candidates = storage_data["storage_data_2"]

    matches = []
    non_matches = []

    print("Aantal oplossingen:", len(candidates))

    for s in candidates:
        ok, reasons = match_with_reason(s, phase, sensitive, collab, volume)

        s_copy = s.copy()
        if ok:
            matches.append(s_copy)
        else:
            s_copy["reasons"] = reasons
            non_matches.append(s_copy)

    return jsonify({
        "matches": sanitize_for_json(matches),
        "non_matches": sanitize_for_json(non_matches)
    })


@app.route("/api/solutions")
def api_solutions():
    return jsonify(all_solutions)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": "storagefinder",
        "solutions_loaded": len(all_solutions),
        "chat_rate_limit": {
            "requests": CHAT_RATE_LIMIT_REQUESTS,
            "window_seconds": CHAT_RATE_LIMIT_WINDOW_SECONDS,
        },
    })

@app.route("/chat")
def chat():
    if not CHATBOT_ENABLED:
        return "The chat assistant is currently disabled.", 503
    return render_template("chat.html")

@app.route("/chat/metrics")
def chat_metrics_page():
    if not CHATBOT_ENABLED:
        return "The chat assistant is currently disabled.", 503
    return render_template("chat_metrics.html")

@app.route("/api/chat/metrics")
def chat_metrics_api():
    if not CHATBOT_ENABLED:
        return jsonify({"error": "chatbot disabled"}), 503
    days_raw = request.args.get("days", "14")
    try:
        days = max(1, min(int(days_raw), 120))
    except Exception:
        days = 14
    return jsonify(compute_metrics_summary(days=days))

@app.route("/chat/reset", methods=["POST"])
def reset_chat():
    if not CHATBOT_ENABLED:
        return jsonify({"message": "Chat is disabled."}), 503
    session.pop("chat_history", None)
    session.pop("pending_clarification", None)
    session.pop("chat_lang", None)
    session.pop("last_topic", None)
    return jsonify({"message": "Chat reset."})

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=_env_flag("FLASK_DEBUG", False),
    )
