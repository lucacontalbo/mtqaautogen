from inference_models import OpenAIModel, remove_markdown_syntax, extract_result, QwenModel

from prompts.inference import prompt
import pandas as pd
from tqdm import tqdm
from utils import is_float, get_args_test
import os

import re
import string
from collections import Counter

import ast

def extract_number(text: str):
    match = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    value = match.group(0).replace(",", "")
    return float(value)

def is_string_containing_python_list(s: str) -> bool:
    try:
        value = ast.literal_eval(s)
        return isinstance(value, list)
    except (ValueError, SyntaxError):
        return False

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation or ch in ["."])

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

if __name__ == '__main__':
    args = get_args_test()
    if "gpt" in args["model"].lower():
        model = OpenAIModel(model_name=args["model"])
    elif "qwen" in args["model"].lower():
        model = QwenModel(model_name=args["model"])
    else:
        raise NotImplementedError()

    df = pd.read_csv(args['filepath'])
    #method = args['filepath'].split("/")[-1].split(".")[0]
    if args["tatqa"]:
        filepath = os.path.join(args["model"], args['filepath'].split("datasets/")[-1])
        to_save_path = os.path.join("results", "tatqa", f"{filepath.split('.')[0]}")
        method = ""
    elif args["bird"]:
        filepath = os.path.join(args["model"], args['filepath'].split("datasets/")[-1])
        to_save_path = os.path.join("results", "bird", f"{filepath.split('.')[0]}")
        method = df.iloc[0]['Method']
    else:
        filepath = os.path.join(args["model"], args['filepath'].split("datasets/")[-1])
        to_save_path = os.path.join("results", "ours", f"{filepath.split('.')[0]}")
        method = df.iloc[0]['Method']

    predictions = []
    results = []

    for i,row in tqdm(df.iterrows(), desc="Iterating over evaluation instances..."):
        question = row['Question']
        table = row['Table']
        if isinstance(row['Label'], list) or is_string_containing_python_list(row['Label']):
            if is_string_containing_python_list(row['Label']):
                label = ast.literal_eval(row['Label'])
            else:
                label = row['Label']
            label = str(label[0])
        else:
            label = str(row['Label'])
        attr = {"question": question, "table": table}
        result, logging = model.query(prompt, attr=attr)
        results.append(logging["text"])
        if i == 0:
            print(logging["text"])
            print()
            print()
            print("**************************")
            print()
            print()
            print(result, flush=True)
        result = extract_result(remove_markdown_syntax(result), "Final answer:")
        result = result.replace("%", "").strip()
        result_to_float = extract_number(result)
        label_to_float = extract_number(label)
        if method in ["percentage_change"] and result_to_float is not None and label_to_float is not None:
            predictions.append([result, label, round(float(result_to_float), 2) == round(float(label_to_float), 2), f1_score(result, label)])
        elif result_to_float is not None and label_to_float is not None:
            predictions.append([result, label, round(float(result_to_float), 6) == round(float(label_to_float), 6), f1_score(result, label)])
        else:
            result, label = str(result), str(label)
            #result = normalize_answer(result)
            #label = normalize_answer(label)
            predictions.append([result, label, result.lower().strip() == label.lower().strip(), f1_score(result, label)])

    res = pd.DataFrame(predictions, columns=['Prediction', 'Label', 'Match', 'F1'])
    cot_reasoning = pd.DataFrame(results, columns=['CoT Reasoning'])
    os.makedirs(to_save_path, exist_ok=True)
    save_path = os.path.join(to_save_path, "preds.csv")
    res.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    accuracy = res['Match'].mean()
    print(f"Accuracy: {accuracy:.4f}")
    f1 = res['F1'].mean()
    print(f"Average F1 Score: {f1:.4f}")

    score_path = os.path.join(to_save_path, "score.csv")

    pd.DataFrame({"Accuracy": [accuracy]}).to_csv(score_path, index=False)
    print(f"Score saved to {score_path}")

    result_path = os.path.join(to_save_path, "reasoning.csv")
    cot_reasoning.to_csv(result_path, index=False)
    print(f"CoT reasoning saved to {result_path}")
