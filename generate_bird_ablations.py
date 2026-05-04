from models import MTAutoGen
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    mt = MTAutoGen({})
    #method = "average"
    path = f"datasets/bird/"

    if True:
        print("Started generation of single table BIRD datasets")
        single_datasets = mt.run_bird_comparison(num_tables=1)
        for method in single_datasets:
            non_relational_dataset, relational_dataset = [], []
            for k in single_datasets[method]:
                for instance in range(len(single_datasets[method][k])):
                    decimals = single_datasets[method][k][instance][1]["decimals"]
                    if k == "non_relational":
                        with pd.option_context("display.float_format", lambda x: f"{x:.{decimals}f}"):
                            non_relational_dataset.append([
                                single_datasets[method][k][instance][1]["nl_question"],
                                single_datasets[method][k][instance][1]["query"],
                                method,
                                single_datasets[method][k][instance][0].to_html(index=True),
                                "",
                                # constraints, which are empty in this case. I keep it to have the same shape as the other generated datasets
                                single_datasets[method][k][instance][1]["label"],
                            ])
                    else:
                        with pd.option_context("display.float_format", lambda x: f"{x:.{decimals}f}"):
                            relational_dataset.append([
                                single_datasets[method][k][instance][1]["nl_question"],
                                single_datasets[method][k][instance][1]["query"],
                                method,
                                single_datasets[method][k][instance][0].to_html(index=True),
                                "",
                                # constraints, which are empty in this case. I keep it to have the same shape as the other generated datasets
                                single_datasets[method][k][instance][1]["label"],
                            ])

                if k == "non_relational":
                    df = pd.DataFrame(non_relational_dataset, columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])
                else:
                    df = pd.DataFrame(relational_dataset, columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])

                os.makedirs(os.path.join(path, method), exist_ok=True)
                df.to_csv(os.path.join(path, method, k+".csv"), index=False)

        print(f"Saving of BIRD single table datasets completed. Datasets saved in {path}")
    else:
        print("Started generation of multi table BIRD datasets")

        multi_datasets = mt.run_bird_comparison(num_tables=3, question_type="parallel")

        for method in multi_datasets:
            multi_table, multi_table_unit_diff, multi_table_rel = [], [], []
            for k in multi_datasets[method]:
                for instance in range(len(multi_datasets[method][k])):
                    decimals = multi_datasets[method][k][instance][1]["decimals"]
                    text = ""
                    for i, table in enumerate(multi_datasets[method][k][instance][0]):
                        with pd.option_context("display.float_format",
                                               lambda x: f"{x:.{decimals[i]}f}"):
                            text += f"Table {i}\n\n{table.to_html(index=True)}\n\n"
                    text = text.strip()

                    if k == "unit_converted":
                        multi_table_unit_diff.append([
                            multi_datasets[method][k][instance][1]["nl_question"],
                            multi_datasets[method][k][instance][1]["query"],
                            method,
                            text,
                            "",
                            # constraints, which are empty in this case. I keep it to have the same shape as the other generated datasets
                            multi_datasets[method][k][instance][1]["label"],
                        ])
                    elif k == "not_unit_converted":
                        multi_table.append([
                            multi_datasets[method][k][instance][1]["nl_question"],
                            multi_datasets[method][k][instance][1]["query"],
                            method,
                            text,
                            "",
                            # constraints, which are empty in this case. I keep it to have the same shape as the other generated datasets
                            multi_datasets[method][k][instance][1]["label"],
                        ])
                    else:
                        multi_table_rel.append([
                            multi_datasets[method][k][instance][1]["nl_question"],
                            multi_datasets[method][k][instance][1]["query"],
                            method,
                            text,
                            "",
                            # constraints, which are empty in this case. I keep it to have the same shape as the other generated datasets
                            multi_datasets[method][k][instance][1]["label"],
                        ])

                if k == "unit_converted":
                    df = pd.DataFrame(multi_table_unit_diff, columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])
                elif k == "not_unit_converted":
                    df = pd.DataFrame(multi_table, columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])
                else:
                    df = pd.DataFrame(multi_table_rel, columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])

                os.makedirs(os.path.join(path, method), exist_ok=True)
                df.to_csv(os.path.join(path, method, "multi_table_parallel_"+k+".csv"))

        print(f"Saving of BIRD multi table datasets completed. Datasets saved in {path}")