import pandas as pd
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', type=str, required=True, help='Model name')

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = get_args()
    df_preds = pd.read_csv(args["filepath"])
    dataset_path = os.path.join("datasets", "/".join(args["filepath"].split("/")[1:-1]))+".csv"
    df = pd.read_csv(dataset_path)
    wrong_idx = df_preds[df_preds["Match"] == False].index
    predictions = df_preds.loc[wrong_idx, "Prediction"].tolist()
    labels = df_preds.loc[wrong_idx, "Label"].tolist()
    df = df.iloc[wrong_idx,:]
    df = df.reset_index()
    for i, row in df.iterrows():
        print(i)
        table = row["Table"] #pd.read_html(row["Table"])
        print(f"*** SAMPLE {i}, original index {wrong_idx[i]} ***")
        print("TABLE:")
        print(table)
        print("PREDICTION:")
        print(predictions[i])
        print("LABEL:")
        print(labels[i])
