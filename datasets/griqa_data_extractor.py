import pandas as pd
import os
from datasets import load_dataset, get_dataset_config_names

dataset_name = "lucacontalbo/GRI-QA"

configs = get_dataset_config_names(dataset_name)

all_data = {}
for config in configs:
    all_data[config] = load_dataset(dataset_name, config)

datasets = {}
for key in all_data:
    df = all_data[key]["train"].to_pandas()
    rows = []
    for _, row in df.iterrows():
        question = row["question"]
        sql_query = []
        method = key+"_"+row["question_type_ext"]
        table = "\n\n".join([f"Table {i}\n\n{t}" for i, t in enumerate(row["tables_html"])])
        constraints = []
        label = row["value"]
        rows.append([question, sql_query, method, table, constraints, label])
    datasets[key] = pd.DataFrame(rows, columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])

for key in datasets:
    print(f"Saving {key} dataset...")
    os.makedirs("../datasets/griqa/", exist_ok=True)
    datasets[key].to_csv(f"../datasets/griqa/{key}.csv", index=False)