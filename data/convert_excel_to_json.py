import pandas as pd
import json
import os

def convert_excel_to_json(excel_path, output_path):
    xls = pd.ExcelFile(excel_path)

    # Sheet 1: Preservation
    df_pres = pd.read_excel(xls, sheet_name='Data Preservation Solutions')
    pres_data = df_pres.set_index(df_pres.columns[0]).transpose()
    pres_data.index.name = "name"
    preservation = pres_data.reset_index().to_dict(orient="records")

    # Sheet 2: Active Storage
    df_active = pd.read_excel(xls, sheet_name='Active Data Storage Solutions')
    active_data = df_active.set_index(df_active.columns[0]).transpose()
    active_data.index.name = "name"
    active = active_data.reset_index().to_dict(orient="records")

    # Sheet 3: Matrix (already in good format)
    df_matrix = pd.read_excel(xls, sheet_name='storagematrixtemp')
    df_matrix.rename(columns={df_matrix.columns[0]: 'Attribute'}, inplace=True)
    matrix = df_matrix.to_dict(orient="records")

    combined = {
        "active_storage": active,
        "preservation_storage": preservation,
        "matrix_data": matrix
    }

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"✅ Exported to {output_path}")

if __name__ == "__main__":
    # Adjust these paths if needed
    excel_file = os.path.join(os.path.dirname(__file__), "UU Reccommeded Storage and Preseservation solutions for RDM_Combined overview.xlsx")
    output_file = os.path.join(os.path.dirname(__file__), "storage_data.json")
    convert_excel_to_json(excel_file, output_file)
