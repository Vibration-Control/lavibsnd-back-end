import json

def load_rows_from_file(filepath):
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            # split on spaces or tabs, filter empty chunks, convert to float
            values = [float(x) for x in line.strip().replace(",", ".").split()]
            if values:
                rows.append(values)
    return rows

def convert_to_primary_system_json(rows):
    return {
        "primarySystemNaturalFrequencies": rows[0],
        "primarySystemModalDamping": rows[1],
        "primarySystemModes": rows[2:]
    }

# ---- USE HERE ----
input_file = "LinhadeTransimissao.eig"   # your file with rows of numbers
output_file = "primary_system.json"

rows = load_rows_from_file(input_file)
result = convert_to_primary_system_json(rows)

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print("JSON generated successfully!")
