import json, os
from itertools import combinations
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hoofdmap
DATA_FILE = os.path.join(os.path.dirname(__file__), "storage_data_2.json")
PDF_DIR = os.path.join(BASE_DIR, "pdf")
TXT_OUTPUT = os.path.join(PDF_DIR, "uu_storage_faq_en.txt")
PDF_OUTPUT = os.path.join(PDF_DIR, "uu_storage_faq_en.pdf")

os.makedirs(PDF_DIR, exist_ok=True)

def generate_faq_text(data):
    lines = []

    # 1️⃣ Per storage solution
    for item in data:
        name = item.get("name", "Unknown")
        details = item.get("storage_details", "").strip() or "No description available."
        host = item.get("Host", "?")
        data_location = item.get("data_location", "?")
        gdpr = item.get("GDPR_compliant", "?")
        collab = item.get("collaboration", "?")
        price = item.get("Price per TB / month", "?")

        lines.append(f"Q: What is {name}?\nA: {details}\n")
        lines.append(f"Q: Who provides {name}?\nA: It is provided by {host}.\n")
        lines.append(f"Q: Where is data from {name} stored?\nA: {data_location}.\n")
        lines.append(f"Q: Is {name} GDPR compliant?\nA: {gdpr}.\n")
        lines.append(f"Q: Can I collaborate using {name}?\nA: {collab}.\n")
        lines.append(f"Q: What is the cost of using {name}?\nA: {price} EUR/TB/month (if applicable).\n")
        lines.append("-" * 80 + "\n")

    # 2️⃣ Synthetic comparisons
    names = [i.get("name") for i in data if i.get("name")]
    for a, b in combinations(names, 2):
        lines.append(f"Q: What is the difference between {a} and {b}?\n")
        lines.append(
            f"A: {a} and {b} are both storage or collaboration services used at Utrecht University, "
            f"but they serve different purposes. {a} may be more suitable for long-term, secure research data management, "
            f"while {b} is typically used for shorter-term or personal file storage. "
            f"Always choose based on data sensitivity, collaboration needs, and retention requirements.\n"
        )
        lines.append("-" * 80 + "\n")

    return lines


def save_text(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"✅ Saved text FAQ: {path}")


def save_pdf(lines, path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 11)
    y = height - 60

    for line in lines:
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 60
        c.drawString(50, y, line.strip())
        y -= 16

    c.save()
    print(f"✅ Saved PDF: {path}")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in current directory")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f).get("storage_data_2", [])

    faq_lines = generate_faq_text(data)
    save_text(faq_lines, TXT_OUTPUT)
    save_pdf(faq_lines, PDF_OUTPUT)
