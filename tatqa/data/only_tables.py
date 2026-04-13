import json, os
import pandas as pd

def is_float(nbr):
    try:
        nbr = nbr.replace(",", "")
        float(nbr)
        return True
    except:
        return False

filenames = [
    "train.json",
    "dev.json",
    "test.json",
]

filtered_data = {
    "train": [],
    "dev": [],
    "test": []
}

for name in filenames:
    with open(name, "r") as f:
        data = json.load(f)

    name_short = name.split(".")[0]
    for d in data:
        questions = []
        for question in d["questions"]:
            if question["answer_from"] != "table":
                continue
            if isinstance(question["answer"], list) and len(question["answer"]) != 1:
                continue
            if isinstance(question["answer"], list) and not is_float(str(question["answer"][0])):
                continue
            if isinstance(question["answer"], str) and not is_float(question["answer"]):
                continue

            filtered_data[name_short].append([pd.DataFrame(d["table"]["table"][1:], columns=d["table"]["table"][0]).to_html(index=True), question["question"], question["answer"]])

    df = pd.DataFrame(filtered_data[name_short], columns=["Table", "Question", "Label"])
    df.to_csv(os.path.join("only_tables", f"{name_short}.csv"), index=False)