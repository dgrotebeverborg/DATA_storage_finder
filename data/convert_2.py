import pandas as pd
import json
import math
# test remote
def convert_david_overview_to_json(excel_path, output_path):
    xls = pd.ExcelFile(excel_path)

    # 📘 1. Lees de hoofd-sheet
    overview_df = pd.read_excel(xls, sheet_name="Overview_for_David_copyM")
    overview_df.columns = [str(c).strip() for c in overview_df.columns]

    # Vul lege 'Column name'-cellen op (merged cell fix)
    if "Column name" in overview_df.columns:
        overview_df["Column name"] = overview_df["Column name"].ffill().astype(str).str.strip()

    # 📗 2. Lees de Datamodel-sheet voor beschrijvingen
    try:
        datamodel_df = pd.read_excel(xls, sheet_name="Datamodel")
        if "Column name" in datamodel_df.columns and "Explanation" in datamodel_df.columns:
            attribute_descriptions = dict(zip(datamodel_df["Column name"], datamodel_df["Explanation"]))
        else:
            attribute_descriptions = {}
    except Exception:
        attribute_descriptions = {}

    # 📒 3. Maak mapping van kolommen naar categorieën en subcategorieën
    category_mapping = {}
    subcategory_mapping = {}
    for _, row in overview_df.iterrows():
        colname = str(row.get("Column name", "")).strip()
        topic = str(row.get("Topic", "")).strip()
        topic2 = str(row.get("Topic 2", "")).strip()
        if colname:
            category_mapping[colname] = topic if topic else ""
            subcategory_mapping[colname] = topic2 if topic2 else ""

    # 📙 4. Relevante kolommen = alle oplossingen (maar UU Internal negeren)
    irrelevant_cols = {"#", "Topic", "Topic 2", "Column name"}
    ignore_solution_cols = {
        "st21",
        "uu internal",
        "st22",
    }  # case-insensitive

    storage_options = [
        col for col in overview_df.columns
        if col not in irrelevant_cols
           and str(col).strip().lower() not in ignore_solution_cols
    ]

    # 📘 5. Helperfunctie voor opschonen
    def _clean_val(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"nan", "n/a", "none"}:
            return None
        return s

    # 📗 6. Bouw output
    structured_output = []
    for option in storage_options:
        if pd.isna(option) or str(option).strip() == "":
            continue

        entry = {
            "name": option,
            "categories": {},
            "subcategories": {},  # <-- nieuw veld
        }

        for _, row in overview_df.iterrows():
            key = row.get("Column name")
            key = _clean_val(key)
            if not key:
                continue

            val = _clean_val(row.get(option))
            if val is None:
                continue

            # combineer dubbele keys tot lijst
            if key in entry:
                existing = entry[key]
                if isinstance(existing, list):
                    if val not in existing:
                        existing.append(val)
                else:
                    if val != existing:
                        entry[key] = [existing, val]
            else:
                entry[key] = val

            # voeg beschrijving toe
            if key in attribute_descriptions:
                entry[f"{key}_description"] = attribute_descriptions[key]

            # voeg categorie en subcategorie toe
            if key in category_mapping:
                entry["categories"][key] = category_mapping[key]
            if key in subcategory_mapping:
                entry["subcategories"][key] = subcategory_mapping[key]

        structured_output.append(entry)

    # 📘 7. Export naar JSON
    output_data = {"storage_data_2": structured_output}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON opgeslagen naar {output_path}")
    print(f"📊 Oplossingen geconverteerd: {len(structured_output)}")


if __name__ == "__main__":
    excel_file = "Storage Overview for DISC.xlsm"
    output_file = "storage_data_2.json"
    convert_david_overview_to_json(excel_file, output_file)
