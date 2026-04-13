from openai_model import OpenAIModel, remove_markdown_syntax, extract_result
from prompts.inference import prompt
import pandas as pd
from tqdm import tqdm
from utils import is_float, get_args_test
import os

if __name__ == '__main__':
    args = get_args_test()
    model = OpenAIModel()

    df = pd.read_csv(args['filepath'])
    to_save_path = os.path.join("results", "tatqa", f"{'/'.join(args['filepath'].split('/')[1:]).split('.')[0]}")
    method = args['filepath'].split("/")[-1].split(".")[0]
    predictions = []
    results = []

    for i,row in tqdm(df.iterrows()):
        question = row['Question']
        table = row['Table']
        label = row['Label']
        attr = {"question": question, "table": table}
        result, _ = model.query(prompt, attr=attr)
        results.append(result)
        result = extract_result(remove_markdown_syntax(result), "Final answer:")
        result = result.replace("%", "").strip()
        if method in ["percentage_change"] and is_float(result) and is_float(label):
            predictions.append([result, label, round(float(result), 2) == round(float(label), 2)])
        elif is_float(result) and is_float(label):
            predictions.append([result, label, round(float(result), 6) == round(float(label), 6)])
        else:
            result, label = str(result), str(label)
            predictions.append([result, label, result.lower().strip() == label.lower().strip()])

    res = pd.DataFrame(predictions, columns=['Prediction', 'Label', 'Match'])
    cot_reasoning = pd.DataFrame(results, columns=['CoT Reasoning'])
    os.makedirs(to_save_path, exist_ok=True)
    save_path = os.path.join(to_save_path, "preds.csv")
    res.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

    accuracy = res['Match'].sum() / len(res)
    print(f"Accuracy: {accuracy:.4f}")

    score_path = os.path.join(to_save_path, "score.csv")

    pd.DataFrame({"Accuracy": [accuracy]}).to_csv(score_path, index=False)
    print(f"Score saved to {score_path}")

    result_path = os.path.join(to_save_path, "reasoning.csv")
    cot_reasoning.to_csv(result_path, index=False)
    print(f"CoT reasoning saved to {result_path}")
