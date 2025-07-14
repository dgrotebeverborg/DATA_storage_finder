from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
from flask import Response
import math
from logic.matching import match_with_reason, sanitize_for_json
from chatbot.storage_rag import ask_storage_question

from flask import Flask, request, jsonify, render_template, session
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session memory

CHROMA_DIR = "chroma_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "models/llama3/llama-pro-8b-instruct.Q4_K_M.gguf"  # ✅ Your local path


# Load data on startup
DATA_PATH = os.path.join("data", "storage_data.json")
with open(DATA_PATH) as f:
    storage_data = json.load(f)

# Combine active and preservation into one list
all_solutions = storage_data["active_storage"] + storage_data["preservation_storage"]

# Safe lowercase helper
def safe_lower(val):
    return str(val).lower() if val and not isinstance(val, float) else ""

@app.route("/chat/ask", methods=["POST"])
def ask_rag():
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "Please enter a message."})

        if "chat_history" not in session:
            session["chat_history"] = []

        limited_history = session["chat_history"][-2:]

        result = ask_storage_question(user_input, limited_history)

        # Nu is result een string
        session["chat_history"].append((user_input, result))

        return jsonify({"response": result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"⚠️ Error: {str(e)}"}), 500


@app.route("/")
def index():
    return render_template("index.html", solutions=all_solutions)

@app.route("/solution/<name>")
def solution_detail(name):
    solution = next((s for s in all_solutions if s["name"] == name), None)
    if not solution:
        return "Solution not found", 404

    cleaned = {}
    for k, v in solution.items():
        if isinstance(v, float) and math.isnan(v):
            cleaned[k] = ""
        else:
            cleaned[k] = str(v)

    return render_template("solution_detail.html", solution=cleaned)

@app.route("/compare")
def compare():
    names = request.args.getlist("names")
    selected = [s for s in all_solutions if s["name"] in names]
    return render_template("compare.html", solutions=selected)

@app.route("/wizard")
def wizard():
    return render_template("wizard.html")

@app.route("/wizard/results")
def wizard_results():
    phase = request.args.get("phase")
    sensitive = request.args.get("sensitive")
    collab = request.args.get("collab")
    volume = request.args.get("volume")

    candidates = (
        storage_data["active_storage"] if phase == "active"
        else storage_data["preservation_storage"]
    )

    matches = []
    non_matches = []

    for s in candidates:
        ok, reasons = match_with_reason(s, sensitive, collab, volume)
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
    phase = data.get("phase") or None
    sensitive = data.get("sensitive") or None
    collab = data.get("collab") or None
    volume = data.get("volume") or None

    candidates = (
        storage_data["active_storage"] if phase == "active"
        else storage_data["preservation_storage"] if phase == "preservation"
        else storage_data["active_storage"] + storage_data["preservation_storage"]
    )

    matches = []
    non_matches = []
    print("Aantal active oplossingen:", len(storage_data["active_storage"]))
    print("Aantal preservation oplossingen:", len(storage_data["preservation_storage"]))

    for s in candidates:
        ok, reasons = match_with_reason(s, sensitive, collab, volume)
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

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/chat/reset", methods=["POST"])
def reset_chat():
    session.pop("chat_history", None)
    return jsonify({"message": "Chat reset."})

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
