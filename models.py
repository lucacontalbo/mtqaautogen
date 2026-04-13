import os
import re
import pickle

from openai_model import OpenAIModel, extract_result, remove_markdown_syntax
from prompts.generate_relational_table import prompt as prompt_generate_relational_table
from prompts.generate_relational_table import prompt_domain as prompt_generate_relational_table_domain
from prompts.generate_relational_table import prompt_domain_unit_specific as prompt_generate_relational_table_domain_unit
from prompts.generate_question_prompt import prompt as prompt_generate_question
from prompts.generate_question_prompt import prompt_percentage_change as prompt_generate_question_percentage_change
from prompts.generate_question_prompt import prompt_comparison as prompt_generate_question_comparison
from prompts.generate_question_prompt import prompt_multi as prompt_generate_question_multi
from prompts.generate_question_prompt import prompt_multi_unit_variation as prompt_generate_question_multi_unit_variation
from prompts.generate_question_prompt import prompt_multi_fk as prompt_generate_question_multi_fk
from prompts.constraint_prompt import constraint_prompt_gpt as prompt_constraint
from prompts.question_verification import prompt as prompt_verify_question
from prompts.question_verification import prompt_multi as prompt_verify_question_multi
from prompts.question_verification import prompt_multi_unit_variation as prompt_verify_question_multi_unit_variation

from sql_sampler import SQLSampler
from perturbations import Perturber, IntraAmbiguousPerturber
from constrainer import Constrainer, parse_expr, extract_columns
from unit_converter import is_unit_in_domain, get_random_unit, get_value, get_n_canonical_units, is_value_going_up
import random
import pandas as pd
import itertools
import operator
import math
from tqdm import tqdm
from copy import deepcopy

from utils import is_float

#pd.set_option('display.float_format', '{:.0f}'.format) # to avoid scientific notation in pandas

SQL_TEMPLATES = {
    "extractive": """SELECT {target} FROM {table_name} WHERE {constraints};""",
    "comparative": """SELECT (SELECT {target} FROM {table_name} WHERE {constraints1}) {comparison} (SELECT {target} FROM {table_name} WHERE {constraints2}) as {target};""",
    "superlative": """SELECT {aggr}({target}) AS {target} FROM ({union_table});""",
    "sum": """SELECT {aggr}({target}) AS {target} FROM ({union_table});""",
    "average": """SELECT {aggr}({target}) AS {target} FROM ({union_table});""",
    #returns NULL if there is a division by zero
    #"percentage_change": """SELECT ((SELECT {target} FROM {table_name} WHERE {constraints1}) - (SELECT {target} FROM {table_name} WHERE {constraints2})) * 100.0 / NULLIF((SELECT {target} FROM {table_name} WHERE {constraints1}), 0) AS {target};""",
    #"relational_comparative_col": """SELECT t1.{target} {comparison} t2.{target} FROM {table_name} t1, {table_name} t2 WHERE {constraints}""",
}

class MTAutoGen:
    def __init__(self, args):
        self.args = args
        self.gpt_model = OpenAIModel(temperature=0.7, p=0.8)
        self.num_columns_range = [4,5]
        self.past_table_names = []
        self.past_table_values = []

        self.sql_sampler = SQLSampler()
        self.perturber = Perturber()
        self.constrainer = Constrainer()
        #self.unit_converter = UnitConverter()
        self.intra_ambiguous_perturber = IntraAmbiguousPerturber()
        self.num_perturbations = 1
        self.rnd = random.Random(-1)

    def set_random_seed(self, i):
        self.rnd = random.Random(i)

    def check_table_sanity(
            self,
            result: dict,
            num_columns: int,
    ) -> bool:
        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals = \
        result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
        result["value_col"], result["id_col"], result["unit_of_measurement"], result["number_of_decimals"]

        sanity = True
        sanity = sanity and (num_columns == len(table_attributes))
        sanity = sanity and (len(table_attributes) == len(table_attributes_long) == len(table_types) == len(ranges))
        sanity = sanity and (value_col in table_attributes)
        sanity = sanity and value_col in table_attributes
        if not sanity:
            return sanity
        idx = table_attributes.index(value_col)
        #sanity = sanity and (table_types[idx] in ["int", "float"])
        units_cleaned = set([unit for unit in units if unit is not None and unit != "None"])
        sanity = sanity and (len(units_cleaned.difference(set(decimals.keys()))) == 0) #every unit of measurement must have a number of decimals defined

        sanity = sanity and (not " " in value_col) and (not "," in value_col) and (not " " in id_col) and (not "," in id_col)
        return sanity

    def generate_relational_table(
            self,
            domain: str = None,
            num_columns: int = None,
            col_cardinality: int = 10,
            canonical_units: list = []
    ):
        if num_columns is None:
            num_columns = random.randint(self.num_columns_range[0], self.num_columns_range[1])

        if domain is not None:
            examples = ""
            if domain in ["finance", "healthcare", "products", "environmental"]:
                path = f"prompts/example_tables/{domain}/"
                for filename in os.listdir(path):
                    if not filename.endswith(".html"):
                        continue

                    table = pd.read_html(os.path.join(path, filename))[0]
                    try:
                        table.columns = [col if "unnamed" not in col.lower() else "" for col in table.columns]
                    except:
                        pass

                    table = table.to_html(index=False) # removing useless style attributes
                    examples += table + "\n\n"
            else:
                raise NotImplementedError

            examples = examples.strip()
            attr = {
                "num_columns": num_columns,
                "col_cardinality": col_cardinality,
                "past": ", ".join(self.past_table_names),
                "past_values": ", ".join(self.past_table_names),
                "domain": domain,
                "examples": examples
            }

            if not canonical_units:
                result_query, _ = self.gpt_model.query(prompt_generate_relational_table_domain, attr=attr)
            else:
                attr["units"] = canonical_units
                result_query, _ = self.gpt_model.query(prompt_generate_relational_table_domain_unit, attr=attr)
        else:
            attr = {"num_columns": num_columns, "col_cardinality": col_cardinality, "past": ", ".join(self.past_table_names), "past_values": ", ".join(self.past_table_names)}
            result_query, _ = self.gpt_model.query(prompt_generate_relational_table, attr=attr)

        if result_query is not None:
            try:
                result = eval(remove_markdown_syntax(extract_result(result_query, "Final answer: ")))
            except:
                return None
                #raise ValueError("wrong gpt generation in 'generate_relational_table'")

            sanity = self.check_table_sanity(result, num_columns) #table_attributes, table_attributes_long, table_types, ranges, units, value_col, id_col, num_columns, decimals)
            if not sanity:
                print("wrong sanity")
                #raise ValueError("wrong sanity")
                return None
        else:
            return None

        return result # table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals

    def generate_semantic_constraints(self, input, domain):
        attr = {"input": input, "domain": domain}
        result_query, _ = self.gpt_model.query(prompt_constraint, attr=attr)
        if result_query is not None:
            try:
                result = eval(remove_markdown_syntax(extract_result(result_query, "Final answer:")))
            except:
                return None
        else:
            return None

        return result

    def fill_dense_relational_table(
            self,
            table_attributes,
            table_types,
            ranges,
            units,
            decimals,
            value_col,
            id_col,
            pivot_keys=None,
            shuffle=False,
            semantic_constraints=None,
            col_to_keep=None,
            value_to_keep=None,
    ):
        if pivot_keys is None:
            pivot_keys = [attr for attr, typ in zip(table_attributes, table_types) if typ == "categorical" or (typ == "int" and attr != value_col)]

        attr_info = list(zip(table_attributes, table_types, ranges))

        type_value_col = table_types[table_attributes.index(value_col)]
        pos_value_col = table_attributes.index(value_col)
        pivot_info = [list(info) for info in attr_info if info[0] in pivot_keys]
        value_info = [list(info) for info in attr_info if info[0] == value_col and type_value_col in ["float", "int"]] #not in pivot_keys]

        # generate full cartesian product of pivot values
        if col_to_keep is not None and value_to_keep is not None:
            pos_col_to_keep = table_attributes.index(col_to_keep)
            type_col_to_keep = table_types[pos_col_to_keep]
            if type_col_to_keep == "int":
                domains_col_to_keep = [value_to_keep]
                domains_col_to_keep.extend([random.randint(*ranges[pos_col_to_keep]) for _ in range(4)])
                for idx in range(len(pivot_info)):
                    if pivot_info[idx][0] == col_to_keep:
                        pivot_info[idx][2] = domains_col_to_keep
                        break

        pivot_domains = [info[2] for info in pivot_info]
        pivot_attr_names = [info[0] for info in pivot_info] # if info[0] != id_col]
        pivot_combinations = list(itertools.product(*pivot_domains))

        """if semantic_constraints is not None and isinstance(semantic_constraints, dict):
            combinations = []
            for combo in pivot_combinations:
                allowed = True
                for values in semantic_constraints.values():
                    for value in values:
                        value = set(value)
                        combo_set = set(combo)
                        if value in combo_set:
                            allowed = False
                            break
                    if not allowed:
                        break
                if allowed:
                    combinations.append(combo)
        else:
            combinations = pivot_combinations"""
        combinations = pivot_combinations

        rows = []
        for combo in combinations:
            row = list(combo)
            for attr, typ, rng in value_info:
                if typ == "int":
                    if is_float(rng[0]) and math.isinf(rng[0]):
                        if rng[0] < 0:
                            first_value = -4_000_000_000
                        else:
                            first_value = 4_000_000_000
                    else:
                        first_value = rng[0]

                    if is_float(rng[1]) and math.isinf(rng[1]):
                        if rng[1] < 0:
                            second_value = -4_000_000_000
                        else:
                            second_value = 4_000_000_000
                    else:
                        second_value = rng[1]
                    val = self.rnd.randint(int(first_value), int(second_value))
                elif typ == "float":
                    val = round(self.rnd.uniform(rng[0], rng[1]), int(decimals[units[0]]))
                elif typ == "categorical":
                    val = self.rnd.choice(rng)
                else:
                    raise ValueError(f"Unknown type {typ}")
                #row.append(val)

                # insert the val at the correct pos_value_col position
                row.insert(pos_value_col, val)

            rows.append(row)
            #if len(rows) > 2000: # TODO: specify better somewhere else this constraint
            #    break

        #final_columns = [info[0] for info in pivot_info + value_info]
        #df = pd.DataFrame(rows, columns=final_columns)
        final_columns = [info[0] for info in pivot_info]
        for attr, _, _ in value_info:
            final_columns.insert(table_attributes.index(attr), attr)

        df = pd.DataFrame(rows, columns=final_columns)

        #value_cols = [info[0] for info in value_info]
        #df = df[[c for c in df.columns if c not in value_cols] + value_cols] # repositioning value_cols at the end
        df = df.drop_duplicates(subset=pivot_attr_names, keep="first")

        if semantic_constraints is not None and isinstance(semantic_constraints, dict) and "intra_row_constraints" in semantic_constraints and "inter_row_constraints" in semantic_constraints:
            df, df_constraints = self.constrainer.get_bounds(df, semantic_constraints, value_col, id_col, ranges, random_state=self.rnd.randint(0, 10**9))
            for i,_ in df.iterrows():
                if type_value_col == "float":
                    df.at[i, value_col] = round(df.at[i, value_col], decimals[units[0]])

        if df is not None and shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        #max_num_rows = self.num_rows_range[1]
        #if len(df) > max_num_rows:
        #    df = df.sample(n=max_num_rows).reset_index(drop=True)

        return df

    def fill_relational_table(self, table_attributes: list, table_types: list, ranges: list):
        num_rows = self.rnd.randint(self.num_rows_range[0], self.num_rows_range[1])
        table = []
        for i in range(num_rows):
            row = []
            for j in range(len(table_attributes)):
                if table_types[j] == "categorical":
                    value = self.rnd.choice(ranges[j])
                elif table_types[j] == "float":
                    value = round(self.rnd.uniform(ranges[j][0], ranges[j][1]), 6)
                elif table_types[j] == "int":
                    value = self.rnd.randint(ranges[j][0], ranges[j][1])
                else:
                    raise ValueError(f"Unknown attribute type: {table_types[j]}")

                row.append(value)
            table.append(row)

        return pd.DataFrame(table, columns=table_attributes)

    def choose_constraint_on_new_table(self,
            columns,
            column_to_vary,
            old_col_values,
            table,
            inter_table_contradiction = False,
        ):

        colname = columns[column_to_vary]

        if inter_table_contradiction:
            candidates = [v for v in table[colname].unique() if v in old_col_values]
        else:
            candidates = [v for v in table[colname].unique() if v not in old_col_values]

        if not candidates:
            return None, None

        vc = table[colname].value_counts(dropna=False)
        best_val = max(candidates, key=lambda v: (int(vc.get(v, 0)), str(v)))

        return best_val

    def generate_label(
            self, table: pd.DataFrame,
            table_name: str,
            value_col: str,
            id_col: str,
            method="extractive",
            columns: list = None,
            constraint: list = None,
            column_to_vary: int = None,
            old_col_values: list = None,
            inter_table_contradiction: bool = False,
            best_val = None,
            impose_target_for_extractive: bool = False,
            all_cols: bool = False,
            col_to_keep: str = None,
            value_to_keep: str = None
    ):
        self.sql_sampler.clear_memory()

        apply_multi_table = False
        if (columns is not None or constraint is not None) and method != "extractive":
            # constrained label generation is only supported for the multi-table scenario, where "extractive" is used
            return None, None
        elif (columns is not None or constraint is not None) and method == "extractive" and len(old_col_values) > 0:
            apply_multi_table = True

        if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
            if not apply_multi_table:
                clusters, columns = self.sql_sampler.get_combination(table, table_name, value_col, all_cols=all_cols,
                                                                     col_to_keep=col_to_keep, value_to_keep=value_to_keep)

                if len(clusters) == 0:
                    return None, None
            else:
                self.sql_sampler.load_table(table, table_name)

            if method == "extractive":
                if not apply_multi_table:
                    constraint = self.rnd.choice(clusters)

                if len(constraint) == len(table.columns):
                    return None, None

                columns_aval = [col for col in table.columns if col not in columns]
                if len(columns_aval) == 0:
                    return None, None

                if column_to_vary is not None and best_val is not None:
                    # in multi-table, we always get the numerical value
                    target = value_col
                    #constraint[column_to_vary] = random.choice([val for val in table[columns[column_to_vary]].unique() if val not in old_col_values])

                    if isinstance(constraint, tuple):
                        constraint = list(constraint)
                    constraint[column_to_vary] = best_val
                else:
                    if impose_target_for_extractive:
                        target = value_col
                    else:
                        target = self.rnd.choice(columns_aval)

                constraint_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint)])  # TODO: add inequality constraints
                formats = {"target": target, "constraints": constraint_txt, "table_name": table_name}
            elif method == "comparative":
                if len(clusters) < 2:
                    return None, None

                constraint = self.rnd.sample(clusters, 2)
                if any([len(c) == len(table.columns) for c in constraint]):
                    return None, None

                target = value_col
                constraint1_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[0])])
                constraint2_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[1])])
                comparison = self.rnd.choice([">", "<", "=", ">=", "<="])
                formats = {"target": target, "constraints1": constraint1_txt, "constraints2": constraint2_txt, "table_name": table_name, "comparison": comparison}
            elif method in ["superlative", "sum", "average"]:
                if method == "superlative":
                    num_constraint = self.rnd.choice([el for el in range(3,5)])
                else:
                    num_constraint = self.rnd.choice([el for el in range(2,5)])

                if len(clusters) < num_constraint:
                    return None, None

                constraint = self.rnd.sample(clusters, num_constraint)
                if any([len(c) == len(table.columns) for c in constraint]):
                    return None, None

                target = value_col
                if method == "superlative":
                    aggr = self.rnd.choice(["MAX", "MIN"])
                elif method == "sum":
                    aggr = "SUM"
                else:
                    aggr = "AVG"

                union_all_selects = []
                for constr_pair in constraint:
                    where_rules = []
                    for col, constr in zip(columns, constr_pair):
                        where_rules.append(f"{col} = \"{constr}\"")
                    where_rules = " AND ".join(where_rules)
                    union_all_selects.append(f"SELECT {target} AS {target} FROM {table_name} WHERE {where_rules}")

                union_table = " UNION ALL ".join(union_all_selects)
                formats = {"aggr": aggr, "union_table": union_table, "target": target}
            elif method == "percentage_change":
                if len(clusters) < 2:
                    return None, None

                constraint = self.rnd.sample(clusters, 2)
                if any([len(c) == len(table.columns) for c in constraint]):
                    return None, None

                target = value_col
                constraint1_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[0])])
                constraint2_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[1])])
                formats = {"target": target, "constraints1": constraint1_txt, "constraints2": constraint2_txt, "table_name": table_name}

            full_mask = pd.Series(False, index=table.index)
            if not isinstance(constraint[0], tuple):
                constraint_tmp = [constraint]
            else:
                constraint_tmp = constraint

            """
            constraint_tmp is basically always [constraint]
            we keep the for loop to facilitate further developments: technically, it is possible that we need to apply,
            for multiple values, multiple constraints.
            """
            for constrs in constraint_tmp:
                mask = pd.Series(True, index=table.index)
                for col, constr in zip(columns, constrs):
                    mask &= (table[col] == constr)
                full_mask |= mask

            sql_query = SQL_TEMPLATES[method].format(**formats)
            data = {
                "query": sql_query,
                "target": target,
                "constraint": constraint if not isinstance(constraint[0], tuple) else sum(constraint, ()),
                "columns": columns if method == "extractive" else columns*len(constraint)
            }
        else:
            raise NotImplementedError()

        result = self.sql_sampler.execute(sql_query)

        result = result[target].item()
        if method == "comparative":
            if result == 0:
                result = "no"
            else:
                result = "yes"

        if is_float(result):
            result = round(float(result), 6)

        data["label"] = str(result)
        return data, full_mask

    def substitute_column_names(self, table: pd.DataFrame, new_column_names: list[str]):
        assert len(new_column_names) == len(table.columns)
        mapping = {old: new for old, new in zip(table.columns, new_column_names)}
        table_renamed = table.rename(columns=mapping)
        return table_renamed

    def generate_label_multitable(self, list_of_data, method):
        data_merged = {
            "query": [],
            "target": [],
            "constraint": [],
            "columns": [],
            "labels": []
        }
        for data in list_of_data:
            data_merged["query"].append(data["query"])
            data_merged["target"].append(data["target"])
            data_merged["constraint"].extend(data["constraint"])
            data_merged["columns"].extend(data["columns"])
            data_merged["labels"].append(data["label"])

        data_merged["method"] = method

        if method == "comparative":
            if len(data_merged["labels"]) != 2:
                raise ValueError("comparative multi-table requires exactly 2 tables")
            if not is_float(data_merged["labels"][0]) or not is_float(data_merged["labels"][1]):
                return None

            ops = {
                ">": operator.gt,
                "<": operator.lt,
                "==": operator.eq,  # or "==" if you prefer
                ">=": operator.ge,
                "<=": operator.le,
            }

            comparison = self.rnd.choice([">", "<", "==", ">=", "<="])
            data_merged["comparison"] = f"first_value {comparison} second_value"
            if ops[comparison](float(data_merged["labels"][0]), float(data_merged["labels"][1])):
                data_merged["label"] = "yes"
            else:
                data_merged["label"] = "no"
        elif method in ["superlative", "sum", "average"]:
            if False in [is_float(label) for label in data_merged["labels"]]: #not is_float(data_merged["labels"][0]) or not is_float(data_merged["labels"][1]):
                return None

            if method == "superlative":
                comparison = self.rnd.choice([max, min])
                data_merged["comparison"] = "get the maximum value" if comparison.__name__ == "max" else "get the minimum value"
            elif method == "sum":
                comparison = sum
                data_merged["comparison"] = "get the sum of the values"
            elif method == "average":
                comparison = lambda x: sum(x) / len(x)
                data_merged["comparison"] = "get the average of the values"

            data_merged["label"] = str(comparison([float(label) for label in data_merged["labels"]]))
        elif method == "percentage_change":
            if len(data_merged["labels"]) != 2:
                raise ValueError("comparative multi-table requires exactly 2 tables")
            if not is_float(data_merged["labels"][0]) or not is_float(data_merged["labels"][1]):
                return None
            if float(data_merged["labels"][1]) == 0:
                return None

            data_merged["comparison"] = "((first_value * second_value) * 100) / second_value"
            data_merged["label"] = str(float(data_merged["labels"][0]) - float(data_merged["labels"][1]) * 100.0 / float(data_merged["labels"][1]))

        return data_merged

    def generate_label_multitable_fk(self, list_of_data, method):
        data_merged = {
            "query": [],
            "target": [],
            "constraint": [],
            "columns": [],
            "labels": []
        }

        for data in list_of_data:
            data_merged["query"].append(data["query"])
            data_merged["target"].append(data["target"])
            data_merged["constraint"].extend(data["constraint"])
            data_merged["columns"].extend(data["columns"])
            data_merged["labels"].append(data["label"])

        data_merged["method"] = method
        data_merged["label"] = data_merged["labels"][-1]

        return data_merged

    def check_nlquestion_validity(
            self,
            nl_question: str,
            sql_query: str,
            table: str | list[str],
            label: str,
            multi: bool = False,
            unit: str = None,
    ):
        if isinstance(table, list):
            table = "\n\n".join([f"Table {i+1}:\n{t}" for i,t in enumerate(table)])
        attr = {"nl_question": nl_question, "table": table, "sql_question": sql_query, "sql_result": label}

        if unit is None:
            result, _ = self.gpt_model.query(prompt_verify_question if not multi else prompt_verify_question_multi, attr=attr)
        else:
            attr["unit"] = unit
            if not multi:
                raise ValueError("cannot have unit specified but no multi setting")
            result, _ = self.gpt_model.query(prompt_verify_question_multi_unit_variation, attr=attr)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final answer:"))
            if "yes" in result.lower():
                return nl_question
            else:
                return result

        return result

    def check_nlquestion_validity_fk(
            self,
            nl_question: str,
            data: dict,
            table: str | list[str],
            label: str,
            multi: bool = False,
            unit: str = None,
    ):
        if isinstance(table, list):
            table = "\n\n".join([f"Table {i+1}:\n{t}" for i,t in enumerate(table)])

        for j, sql_query in enumerate(data["query"][:-1]):
            attributes = self.get_select_attributes(sql_query)
            if len(attributes) != 1:
                raise ValueError("only single attribute supported in multi-table with foreign key scenario")

            for k, sql_query2 in enumerate(data["query"][j+1:]):
                for attribute in attributes:
                    if re.search(rf'\b{re.escape(attribute)}\b\s*(?:=|!=|<>|>=|<=|>|<)\s*(?:"[^"]*"|\'[^\']*\'|\S+)', sql_query2, re.IGNORECASE):
                        data["query"][j+1+k] = self.substitute_where_clause(sql_query2, attribute)

        attr = {"nl_question": nl_question, "table": table, "sql_question": [data["query"][0], data["query"][-1]], "sql_result": label}

        if unit is None:
            result, _ = self.gpt_model.query(prompt_verify_question if not multi else prompt_verify_question_multi, attr=attr)
        else:
            attr["unit"] = unit
            if not multi:
                raise ValueError("cannot have unit specified but no multi setting")
            result, _ = self.gpt_model.query(prompt_verify_question_multi_unit_variation, attr=attr)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final answer:"))
            if "yes" in result.lower():
                return nl_question
            else:
                return result

        return result

    def check_nlquestion_validity_multi(self, nl_question: str, query_sqls: list[str], list_of_tables: list[str], label: str, aggregation: str, number_of_decimals: int):
        with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals}f}"):
            table_htmls = [table.to_html(index=False) for table in list_of_tables]
        text = "\n\n".join([f"SQL Query {i + 1}: {sql_query}\nTable {i + 1}: {table}" for i, (sql_query, table) in
                            enumerate(zip(query_sqls, table_htmls))])

        attr = {"nl_question": nl_question, "text": text, "aggregation": aggregation, "result": label}

        result, _ = self.gpt_model.query(prompt_verify_question, attr=attr)
        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final answer:"))
            if "yes" in result.lower():
                return nl_question
            else:
                return result

        return result

    def generate_question(self, table: pd.DataFrame, data: dict, number_of_decimals: int, method: str = None):
        with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals}f}"):
            table_html = table.to_html(index=False)
        query_sql = data["query"]
        label = data["label"]
        attr = {"table": table_html, "query": query_sql, "result": label}
        if method == "percentage_change":
            prompt_tmp = prompt_generate_question_percentage_change
        elif method == "comparative":
            prompt_tmp = prompt_generate_question_comparison
        else:
            prompt_tmp = prompt_generate_question

        result, _ = self.gpt_model.query(prompt_tmp, attr=attr, create_question=True)
        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final question:"))
        return result

    def generate_question_multitable(
            self,
            list_of_tables: list,
            data: dict,
            number_of_decimals: list[int],
            method: str = "superlative",
            unit: str = None
    ):
        table_htmls = []
        for i, table in enumerate(list_of_tables):
            with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals[i]}f}"):
                #table_htmls = [table.to_html(index=False) for table in list_of_tables]
                table_htmls.append(table.to_html(index=False))

        query_sqls = data["query"]
        label = data["label"]
        text = "\n\n".join([f"SQL Query {i+1}: {sql_query}\nTable {i+1}: {table}" for i, (sql_query, table) in enumerate(zip(query_sqls, table_htmls))])

        attr = {"text": text, "aggregation": data["comparison"], "result": label, "method": method}
        if unit is None:
            result, _ = self.gpt_model.query(prompt_generate_question_multi, attr=attr, create_question=True)
        else:
            attr["unit"] = unit
            result, _ = self.gpt_model.query(prompt_generate_question_multi_unit_variation, attr=attr, create_question=True)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final question:"))
        return result, table_htmls

    def get_select_attributes(self, sql_query: str):
        select_part = re.search(r"select\s+(.*?)\s+from\b", sql_query, re.IGNORECASE | re.DOTALL).group(1)
        return [x.strip() for x in select_part.split(",")]

    def substitute_where_clause(self, sql_query: str, attribute: str):
        return re.sub(
            rf'(\b{re.escape(attribute)}\b\s*(?:=|!=|<>|>=|<=|>|<)\s*)(?:"[^"]*"|\'[^\']*\'|\S+)',
            r'\1"this value depends on the previous instruction"',
            sql_query,
            flags=re.IGNORECASE
        )

    def generate_question_multitable_fk(
            self,
            list_of_tables: list,
            data: dict,
            number_of_decimals: list[int],
            method: str = "superlative",
    ):
        table_htmls = []
        for i, table in enumerate(list_of_tables):
            with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals[i]}f}"):
                #table_htmls = [table.to_html(index=False) for table in list_of_tables]
                table_htmls.append(table.to_html(index=False))

        for j, sql_query in enumerate(data["query"][:-1]):
            attributes = self.get_select_attributes(sql_query)
            if len(attributes) != 1:
                raise ValueError("only single attribute supported in multi-table with foreign key scenario")

            for k, sql_query2 in enumerate(data["query"][j+1:]):
                for attribute in attributes:
                    if re.search(rf'\b{re.escape(attribute)}\b\s*(?:=|!=|<>|>=|<=|>|<)\s*(?:"[^"]*"|\'[^\']*\'|\S+)', sql_query2, re.IGNORECASE):
                        data["query"][j+1+k] = self.substitute_where_clause(sql_query2, attribute)

        query_sqls = data["query"]
        label = data["label"]
        #text = "\n\n".join([f"SQL Query {i+1}: {sql_query}\nTable {i+1}: {table}" for i, (sql_query, table) in enumerate(zip(query_sqls, table_htmls))])
        text = "SQL Queries:\n\n"+"\n\n".join(f"Query {i+1}:{query}" for i, query in enumerate([query_sqls[0], query_sqls[-1]]))
        text += "\n\nTables:\n\n" + "\n\n".join(f"Table {i+1}:\n{table}" for i, table in enumerate(table_htmls))

        attr = {"text": text, "result": label, "method": method}
        result, _ = self.gpt_model.query(prompt_generate_question_multi_fk, attr=attr, create_question=True)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final question:"))
        return result, table_htmls

    def run(self, method="extractive", domain=None):
        if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
            aggr = "first"
        else:
            raise NotImplementedError()

        # generation of relational table
        result = self.generate_relational_table(domain=domain)
        print(result)
        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals = \
        result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
        result["value_col"], result["id_col"], result["unit_of_measurement"], result["number_of_decimals"]

        if table_name is None:
            return None, None
        if sum([el == "float" for el in table_types]) > 1:
            return None, None

        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
             return None, None
        print(constraints)

        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        # generate random table by filling the values randomly for num_rows rows
        try:
            table = self.fill_dense_relational_table(table_attributes, table_types, ranges, units, decimals, value_col, id_col, semantic_constraints=constraints)
        except Exception as e:
            return None, None

        # run sql loading and sql generation/execution
        data, full_mask = self.generate_label(table, table_name, value_col, id_col, method=method)
        if data is None:
            return None, None

        print("Initial table")
        print(table)
        apply_null_perturbation = random.randint(0,1)
        if apply_null_perturbation == 1: # null perturbation must be applied before column name substitution
            attrs = {
                "table": table,
                "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                "value_col": value_col,
            }
            table = self.perturber.null_perturbation(**attrs)

        # substituting column names with "web-table"-like ones
        try:
            table = self.substitute_column_names(table, table_attributes_long)
            value_col = table_attributes_long[table_attributes.index(value_col)]
            id_col = table_attributes_long[table_attributes.index(id_col)]
        except Exception as e:
            return None, None

        print("After column name substitution")
        print(table)
        print("perturb")
        # perform perturbations
        for perturbation in self.perturber.pre_hct_perturbations:
            perturbation_mask = random.randint(0,1)
            if perturbation_mask == 0:
                continue

            attrs = {
                "table": table,
                "value_col": value_col
            }

            table, value_col = perturbation(**attrs)

        unit_in_cell = random.randint(0, 1)
        print("Post pre perturbations")
        print(table)
        print(f"Value col: {value_col}, id_col: {id_col}")
        table, value_col = self.perturber.insert_unit_of_measurement(table, value_col, units, unit_in_cell=unit_in_cell)
        table_hct = self.perturber.multiheader_perturbation(table, value_col, id_col, table_types, aggr=aggr, unit_in_cell=unit_in_cell, full_mask=full_mask)
        if table_hct is None:
            return None, None

        for perturbation in self.perturber.post_hct_perturbations:
            perturbation_mask = random.randint(0,1)
            if perturbation_mask == 0:
                continue

            attrs = {
                "table": table_hct
            }

            table_hct = perturbation(**attrs)

        data["nl_question"] = self.generate_question(table_hct, data, decimals[units[0]], method=method)
        if data["nl_question"] is None:
            return None, None

        with pd.option_context("display.float_format", lambda x: f"{x:.{decimals[units[0]]}f}"):
            data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"], table.to_html(index=False), data["label"])

        if data["nl_question"] is None:
            return None, None

        print("Initial table")
        print(table.head())
        print("Label generated")
        print(data)
        print(table_hct)

        data["decimals"] = decimals[units[0]]

        return table_hct, data

    def get_table_view(
            self,
            attributes: list,
            attributes_long: list,
            attributes_types: list,
            ranges: list,
            value_col: str,
            id_col: str,
            columns: list = None,
            columns_considered_to_add: list = []
    ):
        """
        returns a random selection of attributes from a table, while keeping value_col and id_col columns
        """

        value_idx = attributes.index(value_col)
        indices = [value_idx, attributes.index(id_col)]
        if columns is not None:
            indices.extend([attributes.index(col) for col in columns])
        indices = list(set(indices))

        #mean_num_attrs = len(attributes) // num_tables
        #mean_num_attrs = 6
        if columns is None:
            num_columns_to_add = self.rnd.randint(
                5 - len(indices), #self.perturber.num_columns_range[0] - len(indices),
                5 - len(indices) #self.perturber.num_columns_range[1] - len(indices)
            )
            possible_columns_to_add = [i for i in range(len(attributes)) if i not in indices]
        else:
            num_columns_to_add = self.rnd.randint(1, 2)
            #possible_columns_to_add = [i for i in range(len(attributes)) if i not in indices and attributes[i] in columns_considered_to_add]
            possible_columns_to_add = [] # TODO: finalize additional feature adding

        if len(possible_columns_to_add) != 0:
            columns_to_add = self.rnd.sample(possible_columns_to_add, min(len(possible_columns_to_add), num_columns_to_add))
            indices.extend(columns_to_add) #random.sample([i for i in range(len(attributes)) if i not in indices], num_attrs-len(indices)))
            indices = sorted(list(set(indices)))
        else:
            columns_to_add = []

        indices = [i for i in indices if i != value_idx] + [value_idx]
        ranges_tmp = deepcopy(ranges)
        if columns is not None:
            for i in columns_to_add: # we reduce the amount of labels for the added columns to reduce computational complexity
                ranges_tmp[i] = self.rnd.sample(ranges_tmp[i], min(len(ranges_tmp[i]), self.rnd.randint(2,3)))

        get_view = lambda x: [x[i] for i in indices]
        attributes_view, attributes_long_view, attributes_types_view, range_view = (get_view(attributes),
                                                                                   get_view(attributes_long),
                                                                                   get_view(attributes_types),
                                                                                   get_view(ranges_tmp))

        return attributes_view, attributes_long_view, attributes_types_view, range_view #, [attributes[col] for col in columns_to_add]

    def get_table_view_fk(
            self,
            attributes: list,
            attributes_long: list,
            attributes_types: list,
            ranges: list,
            value_col: str,
            col_to_keep: str,
            cols_to_avoid: list,
            is_final: bool = False,
    ):
        """
        returns a random selection of attributes from a table, while keeping value_col and id_col columns
        """

        if is_final:
            indices = [attributes.index(value_col)]
        else:
            indices = []

        if col_to_keep is not None:
            indices.append(attributes.index(col_to_keep))

        indices = list(set(indices))

        #mean_num_attrs = len(attributes) // num_tables
        #mean_num_attrs = 6
        num_columns_to_add = self.rnd.randint(
            5 - len(indices), #self.perturber.num_columns_range[0] - len(indices),
            5 - len(indices) #self.perturber.num_columns_range[1] - len(indices)
        )
        possible_columns_to_add = [i for i in range(len(attributes)) if attributes[i] not in cols_to_avoid+[col_to_keep, value_col] and attributes_types[i] != "float"]

        if len(possible_columns_to_add) != 0:
            columns_to_add = self.rnd.sample(possible_columns_to_add, min(len(possible_columns_to_add), num_columns_to_add))
            indices.extend(columns_to_add) #random.sample([i for i in range(len(attributes)) if i not in indices], num_attrs-len(indices)))
            indices = sorted(list(set(indices)))
        else:
            columns_to_add = []

        ranges_tmp = deepcopy(ranges)
        #if columns is not None:
        #    for i in columns_to_add: # we reduce the amount of labels for the added columns to reduce computational complexity
        #        ranges_tmp[i] = random.sample(ranges_tmp[i], min(len(ranges_tmp[i]), random.randint(2,3)))

        get_view = lambda x: [x[i] for i in indices]
        attributes_view, attributes_long_view, attributes_types_view, range_view = (get_view(attributes),
                                                                                   get_view(attributes_long),
                                                                                   get_view(attributes_types),
                                                                                   get_view(ranges_tmp))

        return attributes_view, attributes_long_view, attributes_types_view, range_view #, [attributes[col] for col in columns_to_add]


    def run_multitable(
            self,
            method="comparative",
            num_tables=2,
            domain=None,
            inter_table_contradiction=False,
            normalization_ambiguity=False,
            apply_unit_conversions=True
    ):
        if num_tables <= 1:
            return self.run(method=method, domain=domain)

        if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
            aggr = "first"
        else:
            raise NotImplementedError()

        if method == "extractive":
            return None, None

        # generation of relational table
        result = self.generate_relational_table(domain=domain, num_columns=self.perturber.num_columns_range[1], col_cardinality=10)
        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["id_col"], result["unit_of_measurement"], result["number_of_decimals"]

        print(result)
        if table_name is None:
            return None, None
        if sum([el == "float" for el in table_types]) > 1:
            return None, None

        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
            return None, None

        print(constraints)
        #self.past_table_names.append(table_name)
        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        unit_normalization_mask = [0] * num_tables
        if normalization_ambiguity:
            unit_normalization_mask[random.randint(0,len(unit_normalization_mask)-1)] = 1

        # generate random table by filling the values randomly for num_rows rows
        list_of_data = []
        list_of_table_hct = []
        columns, constraint_cols, column_to_vary, old_col_values = None, None, None, []
        for i in range(num_tables):
            table_attributes_view, table_attributes_long_view, table_types_view, range_view = self.get_table_view(
                table_attributes,
                table_attributes_long,
                table_types,
                ranges,
                value_col,
                id_col,
                columns=columns,
            )
            #try:
            table = self.fill_dense_relational_table(table_attributes_view, table_types_view, range_view, units, decimals, value_col, id_col, semantic_constraints=constraints)
            #except Exception as e:
            #    return None, None

            # run sql loading and sql generation/execution
            # we force the method to be extractive, as the true method will be applied across tables later
            data, full_mask = self.generate_label(
                table,
                table_name,
                value_col,
                id_col,
                method="extractive",
                columns=columns,
                constraint=constraint_cols,
                column_to_vary=column_to_vary,
                old_col_values=old_col_values,
                inter_table_contradiction=inter_table_contradiction,
            )

            if data is None:
                return None, None

            if columns is None and constraint_cols is None:
                columns = data["columns"]
                constraint_cols = data["constraint"]

                # I make sure that inter row constraints are kept even when dealing with multiple tables
                # for that, in different tables, I vary an attribute that is not part of the inter-row constraints (if any)
                columns_to_consider = [col for col in columns if col not in str(constraints["inter_row_constraints"])]
                # the column to vary must not be the value_col
                columns_to_consider = [col for col in columns_to_consider if col != value_col]
                # the column to vary must have at least num_tables different values, to ensure that we can vary it across tables
                columns_to_consider = [col for col in columns_to_consider if table[col].nunique() >= num_tables]
                if len(columns_to_consider) == 0:
                    return None, None

                max_val, best_col = -1, None
                for col in columns_to_consider:
                    pos = columns.index(col)
                    num_samples = len(table[table[columns[pos]] == data["constraint"][pos]])
                    if num_samples > max_val:
                        max_val = num_samples
                        best_col = col

                column_to_vary = columns.index(best_col)
                #column_to_vary = columns.index(random.choice(columns_to_consider))
                """best_col = max(
                    columns_to_consider,
                    key=lambda c: (table[c].nunique(), -columns.index(c))
                )
                column_to_vary = columns.index(best_col)"""

            old_col_values.append(data["constraint"][column_to_vary])

            table = table[table[columns[column_to_vary]] == data["constraint"][column_to_vary]]

            print(f"Table {i+1} before perturbation")
            print(table)
            apply_null_perturbation = random.randint(0, 1)
            if apply_null_perturbation == 1:  # null perturbation must be applied before column name substitution
                attrs = {
                    "table": table,
                    "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                    "value_col": value_col,
                }
                table = self.perturber.null_perturbation(**attrs)

            # substituting column names with "web-table"-like ones
            try:
                table = self.substitute_column_names(table, table_attributes_long_view)
                new_value_col = table_attributes_long_view[table_attributes_view.index(value_col)]
                new_id_col = table_attributes_long_view[table_attributes_view.index(id_col)]
            except Exception as e:
                return None, None

            # perform perturbations
            for perturbation in self.perturber.pre_hct_perturbations:
                perturbation_mask = random.randint(0, 1)
                if perturbation_mask == 0:
                    continue

                attrs = {
                    "table": table,
                    "value_col": new_value_col
                }

                table, new_value_col = perturbation(**attrs)

            if i != 0 and apply_unit_conversions:
                new_unit, conversion_factor = self.unit_converter.random_convertible_unit(units[0])
                if new_unit is not None:
                    units = [new_unit]
                    table[new_value_col] = table[new_value_col].apply(lambda x: conversion_factor(x) if pd.notnull(x) else x)

            unit_in_cell = random.randint(0, 1)
            if unit_normalization_mask[i] == 0:
                table, new_value_col = self.perturber.insert_unit_of_measurement(table, new_value_col, units, unit_in_cell=unit_in_cell)
            else:
                # unit of measurement not inserted for normalization ambiguity
                new_value_col = value_col

            table_hct = self.perturber.multiheader_perturbation(table, new_value_col, new_id_col, table_types_view, aggr=aggr, unit_in_cell=unit_in_cell, full_mask=full_mask)
            if table_hct is None:
                return None, None

            for perturbation in self.perturber.post_hct_perturbations:
                perturbation_mask = random.randint(0, 1)
                if perturbation_mask == 0:
                    continue

                attrs = {
                    "table": table_hct
                }

                table_hct = perturbation(**attrs)

            print(f"Table {i+1} after perturbation")
            print(table_hct)
            list_of_table_hct.append(table_hct)
            list_of_data.append(data)

        data = self.generate_label_multitable(list_of_data, method=method)
        if data is None:
            return None, None

        data["nl_question"] = self.generate_question_multitable(list_of_table_hct, data, decimals[units[0]])
        if data["nl_question"] is None:
            return None, None

        with pd.option_context("display.float_format", lambda x: f"{x:.{decimals[units[0]]}f}"):
            data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"], [t.to_html(index=False) for t in list_of_table_hct], data["label"])
        if data["nl_question"] is None:
            return None, None

        print("Label generated")
        print(data)
        for t in list_of_table_hct:
            print(t.head())

        data["decimals"] = decimals[units[0]]
        return list_of_table_hct, data

    def run_loop(self, method="extractive", num_tables=1, num_samples=1, domain=None, sequential=False):
        samples = []
        for _ in tqdm(range(num_samples), desc="Generating samples..."):
            if num_tables == 1:
                table, data = self.run(method=method, domain=domain)
            elif num_tables > 1:
                if sequential:
                    raise NotImplementedError
                table, data = self.run_multitable(method=method, num_tables=num_tables, domain=domain)
            else:
                raise ValueError("num_tables must be >= 1")

            if table is None or data is None:
                continue

            with pd.option_context("display.float_format", lambda x: f"{x:.{data['decimals']}f}"):
                samples.append([data["nl_question"], data["query"], method, table.to_html(index=True), data["label"]])

        return pd.DataFrame(samples, columns=["Question", "SQL Query", "Method", "Table", "Label"])

    def run_one_table_ablations(
            self,
            domain=None
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        #if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
        aggr = "first"

        # generation of relational table
        result = self.generate_relational_table(domain=domain)
        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["id_col"], result["unit_of_measurement"], result["number_of_decimals"]

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"

        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
            return None, None, "semantic constraints generation error"

        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        # generate random table by filling the values randomly for num_rows rows
        #try:
        table = self.fill_dense_relational_table(table_attributes, table_types, ranges, units, decimals, value_col,
                                                     id_col, semantic_constraints=constraints)
        #except Exception as e:
        #    return None, None, "table generation error (filling table)"

        datasets = {}
        perturbations_to_apply = [self.perturber.null_perturbation] + \
                                 self.perturber.pre_hct_perturbations + \
                                 self.perturber.post_hct_perturbations + \
                                 self.intra_ambiguous_perturber.perturbations

        for method in SQL_TEMPLATES:
            # run sql loading and sql generation/execution
            data, full_mask = self.generate_label(table, table_name, value_col, id_col, method=method)
            if data is None:
                continue

            datasets[method] = {}

            for num_perturbation, perturbation in enumerate(perturbations_to_apply):
                new_table = table.copy()
                new_full_mask = full_mask
                new_table_types = deepcopy(table_types)
                if num_perturbation == 0: # null perturbation
                    attrs = {
                        "table": new_table,
                        "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                        "value_col": value_col,
                    }
                    new_table = self.perturber.null_perturbation(**attrs)

                # substituting column names with "web-table"-like ones
                try:
                    new_table = self.substitute_column_names(new_table, table_attributes_long)
                    new_value_col = table_attributes_long[table_attributes.index(value_col)]
                    new_id_col = table_attributes_long[table_attributes.index(id_col)]
                except Exception as e:
                    return None, None, "column name substitution error"

                if num_perturbation != 0 and perturbation in self.perturber.pre_hct_perturbations:
                    attrs = {
                        "table": new_table,
                        "value_col": new_value_col
                    }

                    new_table, new_value_col = perturbation(**attrs)

                if num_perturbation != 0 and perturbation in self.intra_ambiguous_perturber.perturbations:
                    attrs = {
                        "table": new_table
                    }
                    if perturbation == self.intra_ambiguous_perturber.entity_missing:
                        attrs["full_mask"] = full_mask
                        new_table, new_full_mask = perturbation(**attrs)
                    elif perturbation == self.intra_ambiguous_perturber.intra_table_contradiction:
                        attrs["value_col"] = new_value_col
                        new_table = perturbation(**attrs)
                    else:
                        raise ValueError(f"Unknown function {perturbation} in intra_ambiguous perturbations")

                try:
                    unit_in_cell = self.rnd.randint(0, 1)
                    new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, units, unit_in_cell=unit_in_cell)
                    table_before_pivot = new_table.copy()
                    table_hct, rows_chosen, cols_chosen, _ = self.perturber.multiheader_perturbation(new_table, new_value_col, new_id_col, new_table_types, aggr=aggr,
                                                                        unit_in_cell=unit_in_cell, full_mask=new_full_mask)

                except:
                    continue

                if table_hct is None:
                    continue

                table_hct = self.perturber.restore_needed_cells_after_value_merge(
                    table_before_pivot=table_before_pivot,
                    pivot_table=table_hct,
                    value_col=new_value_col,
                    full_mask=full_mask,
                    rows_chosen=rows_chosen,
                    cols_chosen=cols_chosen,
                )

                if num_perturbation != 0 and perturbation in self.perturber.post_hct_perturbations:
                    attrs = {
                        "table": table_hct,
                        "strength": 50
                    }

                    table_hct = perturbation(**attrs)

                if "nl_question" not in data: 
                    data["nl_question"] = self.generate_question(table_hct, data, decimals[units[0]], method=method)
                    if data["nl_question"] is None:
                        continue

                    with pd.option_context("display.float_format", lambda x: f"{x:.{decimals[units[0]]}f}"):
                        data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"],
                                                                             table_hct.to_html(index=False), data["label"])
                    if data["nl_question"] is None:
                        continue

                datasets[method][perturbation.__name__] = (table_hct, data, constraints)

            # adding a sample with all perturbations randomly applied with lower strength
            new_table = table.copy()
            new_full_mask = full_mask
            new_table_types = deepcopy(table_types)

            if self.rnd.randint(0,1) == 1: # apply perturbation with 50% chance
                attrs = {
                    "table": new_table,
                    "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                    "value_col": value_col,
                }
                new_table = self.perturber.null_perturbation(**attrs)

            try:
                new_table = self.substitute_column_names(new_table, table_attributes_long)
                new_value_col = table_attributes_long[table_attributes.index(value_col)]
                new_id_col = table_attributes_long[table_attributes.index(id_col)]
            except Exception as e:
                return None, None, "column name substitution error"

            for perturbation in self.perturber.pre_hct_perturbations:
                if self.rnd.randint(0,1) == 1:
                    attrs = {
                        "table": new_table,
                        "value_col": new_value_col
                    }

                    new_table, new_value_col = perturbation(**attrs)

            try:
                unit_in_cell = self.rnd.randint(0, 1)
                new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, units,
                                                                                     unit_in_cell=unit_in_cell)
                table_before_pivot = new_table.copy()
                table_hct, rows_chosen, cols_chosen, _ = self.perturber.multiheader_perturbation(new_table, new_value_col, new_id_col, new_table_types,
                                                                    aggr=aggr,
                                                                    unit_in_cell=unit_in_cell, full_mask=new_full_mask)
            except:
                continue

            if table_hct is None:
                continue

            table_hct = self.perturber.restore_needed_cells_after_value_merge(
                table_before_pivot=table_before_pivot,
                pivot_table=table_hct,
                value_col=new_value_col,
                full_mask=full_mask,
                rows_chosen=rows_chosen,
                cols_chosen=cols_chosen,
            )

            for perturbation in self.perturber.post_hct_perturbations:
                if self.rnd.randint(0,1) == 1:
                    attrs = {
                        "table": table_hct
                    }

                    table_hct = perturbation(**attrs)

            datasets[method]["all_perturbations"] = (table_hct, data, constraints)

        return datasets, decimals[units[0]], None


    def run_multi_table_ablations(
            self,
            domain=None,
            num_tables=2,
            inter_table_contradiction=False,
            col_cardinality=10,
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        #if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
        aggr = "first"
        sampled_canonical_units = get_n_canonical_units(domain, n=20)

        # generation of relational table
        result = self.generate_relational_table(
            domain=domain,
            num_columns=6, #self.perturber.num_columns_range[1],
            col_cardinality=col_cardinality,
            canonical_units=sampled_canonical_units,
        )

        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["id_col"], result["unit_of_measurement"], result["number_of_decimals"]

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"
        if is_unit_in_domain(units[0], domain) is None:
            return None, None, "wrong unit generated"

        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
            return None, None, "semantic constraints generation error"

        columns_considered_to_add = [col for col in table_attributes if col not in str(constraints["inter_row_constraints"]) and col not in str(constraints["intra_row_constraints"])]

        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        list_of_data = []
        list_of_table_hct = []
        #columns, constraint_cols, column_to_vary, old_col_values = None, None, None, []
        datasets = {}
        tables_unit_converted, tables_not_unit_converted = {}, {}
        data_unit_converted, data_not_unit_converted = {}, {}
        new_decimals = decimals
        new_units = units
        view_constraints_tot_unit, view_constraints_tot_not_unit = {}, {}
        initial_offset = random.randint(0, 10000) # used to guarantee that different runs use different random perturbations

        for it, num_tables in enumerate([2, 3, 5, 10, 20]):
            tables_unit_converted[str(num_tables)] = []
            tables_not_unit_converted[str(num_tables)] = []
            data_unit_converted[str(num_tables)] = []
            data_not_unit_converted[str(num_tables)] = []
            view_constraints_tot_unit[str(num_tables)] = []
            view_constraints_tot_not_unit[str(num_tables)] = []
            datasets[str(num_tables)] = {}

        columns, constraint_cols, column_to_vary, old_col_values = None, None, None, []
        good = True
        for it, num_tables in enumerate([2, 3, 5, 10, 20]):
            print(f"Generating for {num_tables} tables...")
            #columns, constraint_cols, column_to_vary, old_col_values = None, None, None, []

            if it == 0:
                previous_num_tables = 0
            else:
                previous_num_tables = [2,3,5,10,20][it-1]

            for i in range(previous_num_tables, num_tables):
                self.set_random_seed(initial_offset+i)
                table_attributes_view, table_attributes_long_view, table_types_view, range_view = self.get_table_view(
                    table_attributes,
                    table_attributes_long,
                    table_types,
                    ranges,
                    value_col,
                    id_col,
                    columns=columns,
                    columns_considered_to_add=columns_considered_to_add
                )

                if column_to_vary is None:
                    column_to_vary = self.rnd.choice([idx for idx in range(len(table_attributes_view)) if table_types_view[idx] not in ["float", "int"]])

                for idx in range(len(range_view)):
                    if idx != column_to_vary and table_types_view[idx] not in ["float", "int"]:
                        range_view[idx] = range_view[idx][:6] # we reduce the combinations of values to avoid too much overhead


                # generate random table by filling the values randomly for num_rows rows
                try:
                    view_constraints = {"inter_row_constraints": [], "intra_row_constraints": []}
                    for constraint_type in constraints:
                        for rule in constraints[constraint_type]:
                            ok = True
                            if constraint_type == "inter_row_constraints":
                                conditions = [m.group("cond") for m in re.compile(self.constrainer.ATOM_EXTRACT, re.VERBOSE | re.DOTALL).finditer(rule)]
                            else:
                                #conditions = [rule]
                                m = self.constrainer.PATTERN_intra.match(rule)
                                if not m:
                                    raise ValueError(f"Invalid intra rule: {rule!r}")
                                conditions = [m.group("condition").strip()]
                                expr = [m.group("expr").strip()]
                                conditions.extend(expr)

                            for cond in conditions:
                                cond = cond.strip()
                                if cond[0] in ["(", ")"]:
                                    cond = cond[1:]
                                if cond[-1] in ["(", ")"]:
                                    cond = cond[:-1]
                                columns_from_rule = extract_columns(parse_expr(cond))
                                for col in columns_from_rule:
                                    if col not in table_attributes_view:
                                        ok = False
                                        break
                                if not ok:
                                    break

                            if ok:
                                view_constraints[constraint_type].append(rule)

                    table = self.fill_dense_relational_table(table_attributes_view, table_types_view, range_view, units, decimals, value_col,
                                                                 id_col, semantic_constraints=view_constraints)

                    best_val = None
                    if i != 0:
                        best_val = self.choose_constraint_on_new_table(
                            columns,
                            column_to_vary,
                            old_col_values,
                            table,
                            inter_table_contradiction
                        )

                        if isinstance(constraint_cols, tuple):
                            constraint_cols = list(constraint_cols)

                        constraint_cols[column_to_vary] = best_val

                        # rows where the selected columns match the given values
                        mask = (table[columns] == constraint_cols).all(axis=1)

                        # keep all non-matching rows, and deduplicate matching rows
                        table = pd.concat([
                            table[~mask],
                            table[mask].drop_duplicates(subset=columns, keep="first")
                        ], ignore_index=True)

                except Exception as e:
                    good = False
                    break
                    #return None, None, "table generation error (filling table)"

                # run sql loading and sql generation/execution
                #data, full_mask = self.generate_label(table, table_name, value_col, id_col, method=method)
                try:
                    data, full_mask = self.generate_label(
                        table,
                        table_name,
                        value_col,
                        id_col,
                        method="extractive", # we force the method to be extractive, as the true method will be applied across tables later
                        columns=columns,
                        constraint=constraint_cols,
                        column_to_vary=column_to_vary,
                        old_col_values=old_col_values,
                        inter_table_contradiction=inter_table_contradiction,
                        best_val=best_val,
                        all_cols=True,
                        impose_target_for_extractive=True
                    )
                except:
                    good = False
                    break

                if data is None:
                    good = False
                    break
                    #return None, None, f"error in data generation for the {i+1}-th table"

                if columns is None and constraint_cols is None:
                    columns = data["columns"]
                    constraint_cols = data["constraint"]

                    if column_to_vary is None:
                        # I make sure that inter row constraints are kept even when dealing with multiple tables
                        # for that, in different tables, I vary an attribute that is not part of the inter-row constraints (if any)
                        columns_to_consider = [col for col in columns if
                                               col not in str(view_constraints["inter_row_constraints"])]
                        # the column to vary must not be the value_col
                        columns_to_consider = [col for col in columns_to_consider if col != value_col]
                        # the column to vary must have at least num_tables different values, to ensure that we can vary it across tables
                        columns_to_consider = [col for col in columns_to_consider if table[col].nunique() >= num_tables]
                        if len(columns_to_consider) == 0:
                            good = False
                            break
                            #return None, None, "column to differ between tables error"

                        max_val, best_col = -1, None
                        for col in columns_to_consider:
                            pos = columns.index(col)
                            num_samples = len(table[table[columns[pos]] == data["constraint"][pos]])
                            if num_samples > max_val:
                                max_val = num_samples
                                best_col = col

                        column_to_vary = columns.index(best_col)

                old_col_values.append(data["constraint"][column_to_vary])

                table = table[table[columns[column_to_vary]] == data["constraint"][column_to_vary]]

                rnd_apply_nan = self.rnd.randint(0,1)
                unit_in_cell = self.rnd.randint(0, 1)
                rnd_apply_pre = []
                for _ in self.perturber.pre_hct_perturbations:
                    rnd_apply_pre.append(self.rnd.randint(0,1))

                rnd_apply_post = []
                for _ in self.perturber.post_hct_perturbations:
                    rnd_apply_post.append(self.rnd.randint(0,1))

                # adding a sample with all perturbations randomly applied with lower strength
                for apply_unit_conversions in [True, False]:
                    new_table = table.copy()
                    new_full_mask = full_mask
                    new_table_types = deepcopy(table_types)

                    self.perturber.set_random_seed(initial_offset+i)
                    if rnd_apply_nan == 1: # apply perturbation with 50% chance
                        attrs = {
                            "table": new_table,
                            "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                            "value_col": value_col,
                        }

                        new_table = self.perturber.null_perturbation(**attrs)

                    try:
                        new_table = self.substitute_column_names(new_table, table_attributes_long_view)
                        new_value_col = table_attributes_long[table_attributes.index(value_col)]
                        new_id_col = table_attributes_long[table_attributes.index(id_col)]
                    except Exception as e:
                        #return None, None, "column name substitution error"
                        good = False
                        break

                    try:
                        new_units = units
                        new_decimals = decimals
                        if i != 0 and apply_unit_conversions:
                            needed_decimals = -1
                            for j in range(5):
                                surrogate_unit, target_unit = get_random_unit(units[0], domain, seed=initial_offset+i+j)
                                highest_needed_decimals = -1
                                for k,row in new_table.iterrows():
                                    try:
                                        value_converted, needed_decimals = get_value(
                                            surrogate_unit,
                                            target_unit,
                                            domain,
                                            row[new_value_col],
                                            self.perturber.nan_fill_str
                                        )
                                    except:
                                        break
                                        #return None, None, "error: chosen unit whose factor_to_canonical is zero"

                                    if needed_decimals is not None:
                                        if needed_decimals == -1:
                                            break
                                            # return None, None, "couldn't find rounding with the specified number of decimal values"

                                        #new_table.at[k, new_value_col] = round(value_converted, needed_decimals)
                                        new_table.at[k, new_value_col] = value_converted
                                        if needed_decimals > highest_needed_decimals:
                                            highest_needed_decimals = needed_decimals

                                if needed_decimals != -1:
                                    break

                            if needed_decimals == -1:
                                good = False
                                break
                                #return None, None, "couldn't find rounding with the specified number of decimal values"

                            #data["label"] = get_value(surrogate_unit, target_unit, domain, data["label"], self.perturber.nan_fill_str)
                            if surrogate_unit != units[0]:
                                target_unit = f"{target_unit} {units[0].replace('[', '').replace(']', '')}"

                            new_units = [target_unit]
                            new_decimals = {target_unit: max(decimals[units[0]], highest_needed_decimals)}
                    except:
                        good = False
                        break
                        #return None, None, "error in unit retrieval"

                    for num_pert, perturbation in enumerate(self.perturber.pre_hct_perturbations):
                        if rnd_apply_pre[num_pert] == 1 and i != 0:
                            attrs = {
                                "table": new_table,
                                "value_col": new_value_col
                            }

                            new_table, new_value_col = perturbation(**attrs)

                    try:
                        new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                             unit_in_cell=unit_in_cell)
                        table_before_pivot = new_table.copy()
                        table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table, new_value_col, new_id_col, new_table_types,
                                                                            aggr=aggr,
                                                                            unit_in_cell=unit_in_cell, full_mask=new_full_mask)
                    except:
                        good = False
                        break
                        #return None, None, "table generation error (filling)"

                    if table_hct is None:
                        good = False
                        break
                        #return None, None, "table generation error (pivoting)"

                    table_hct = self.perturber.restore_needed_cells_after_value_merge(
                        table_before_pivot=table_before_pivot,
                        pivot_table=table_hct,
                        value_col=new_value_col,
                        full_mask=full_mask,
                        rows_chosen=rows_chosen,
                        cols_chosen=cols_chosen,
                    )

                    for num_pert, perturbation in enumerate(self.perturber.post_hct_perturbations):
                        if perturbation == self.perturber.insert_blank_columns and option == 0: # may cancel unit of measurement
                            continue
                        if rnd_apply_post[num_pert] == 1:
                            attrs = {
                                "table": table_hct
                            }

                            table_hct = perturbation(**attrs)

                    data["decimals"] = new_decimals[new_units[0]]
                    data["units"] = new_units[0]
                    if apply_unit_conversions:
                        for num_tables2 in [2,3,5,10,20]:
                            if num_tables2 < num_tables:
                                continue
                            tables_unit_converted[str(num_tables2)].append(table_hct.copy())
                            data_unit_converted[str(num_tables2)].append(deepcopy(data))
                            view_constraints_tot_unit[str(num_tables2)].append(view_constraints)
                    else:
                        for num_tables2 in [2,3,5,10,20]:
                            if num_tables2 < num_tables:
                                continue
                            tables_not_unit_converted[str(num_tables2)].append(table_hct.copy())
                            data_not_unit_converted[str(num_tables2)].append(deepcopy(data))
                            view_constraints_tot_not_unit[str(num_tables2)].append(view_constraints)

                    #list_of_table_hct.append(table_hct)
                    #list_of_data.append(data)
                #datasets[method]["all_perturbations"] = (table_hct, data, constraints)

            if not good:
                return datasets, decimals[units[0]], None

            for perturbation_name, list_of_table_hct, list_of_data, view_constr in zip(
                    ["unit_converted", "not_unit_converted"],
                    [tables_unit_converted[str(num_tables)], tables_not_unit_converted[str(num_tables)]],
                    [data_unit_converted[str(num_tables)], data_not_unit_converted[str(num_tables)]],
                    [view_constraints_tot_unit[str(num_tables)], view_constraints_tot_not_unit[str(num_tables)]]
            ):
                for method_name in SQL_TEMPLATES:
                    if method_name in ["extractive", "comparative", "percentage_change"]:
                        continue

                    if method_name not in datasets[str(num_tables)]:
                        datasets[str(num_tables)][method_name] = {}

                    data = self.generate_label_multitable(list_of_data, method=method_name)
                    if data is None:
                        return None, None, "error in generating multitable data"

                    data["nl_question"], tables_final = self.generate_question_multitable(
                        list_of_table_hct,
                        data,
                        [d["decimals"] for d in list_of_data],
                        method_name,
                        unit=list_of_data[0]["units"]
                    ) #list_of_data["decimals"])
                    if data["nl_question"] is None:
                        return None, None, "error in generating nl question"

                    #with pd.option_context("display.float_format", lambda x: f"{x:.{list_of_data['decimals']}f}"):

                    data["nl_question"] = self.check_nlquestion_validity(
                        data["nl_question"],
                        data["query"],
                        tables_final,
                        data["label"],
                        multi=True,
                        unit=list_of_data[0]["units"]
                    )
                    if data["nl_question"] is None:
                        return None, None, "error in checking nl question"

                    #print("Label generated")
                    #print(data)
                    #for i, t in enumerate(list_of_table_hct):
                    #    print(f"TABLE {i}")
                    #    print(t)

                    data["decimals"] = [d["decimals"] for d in list_of_data]
                    datasets[str(num_tables)][method_name][perturbation_name] = (list_of_table_hct, data, view_constr)
                    #return list_of_table_hct, data

        return datasets, decimals[units[0]], None

    def run_multi_table_ablations_fk( # table splitted by foreign key
            self,
            domain=None,
            num_tables=2,
            inter_table_contradiction=False,
            col_cardinality=8,
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        #if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
        aggr = "first"
        sampled_canonical_units = get_n_canonical_units(domain, n=20)

        # generation of relational table
        #result = self.generate_relational_table(
        #    domain=domain,
        #    num_columns=21, #21 columns, so get_table_view_fk chooses 5 new attributes on the first table, and 4 on the others, allowing maximum 5 tables. This constraint can be relaxed by allowing duplicate data across tables
        #    col_cardinality=col_cardinality,
        #    canonical_units=sampled_canonical_units,
        #)
        #with open("result.pkl", "wb") as writer:
        #    pickle.dump(result, writer)

        with open("result.pkl", "rb") as reader:
            result = pickle.load(reader) # TODO: remove

        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, id_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["id_col"], result["unit_of_measurement"], result["number_of_decimals"]

        ranges = [r[:5] for r in ranges] # TODO: remove

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"
        if is_unit_in_domain(units[0], domain) is None:
            return None, None, "wrong unit generated"

        #constraints = self.generate_semantic_constraints(result, domain=domain)
        #with open("constraints.pkl", "wb") as writer:
        #    pickle.dump(constraints, writer)
        with open("constraints.pkl", "rb") as reader: # TODO: remove
            constraints = pickle.load(reader)

        if constraints is None:
            return None, None, "semantic constraints generation error"

        columns_considered_to_add = [col for col in table_attributes if col not in str(constraints["inter_row_constraints"]) and col not in str(constraints["intra_row_constraints"])]

        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        #cols_to_avoid = []

        columns, constraint_cols, column_to_vary, old_col_values = None, None, None, []
        datasets = {}
        errors = []

        #table = self.fill_dense_relational_table(table_attributes, table_types, ranges, units, decimals,
        #                                         value_col, id_col, semantic_constraints=constraints)

        initial_offset = random.randint(0, 10000) # used to guarantee that different runs use different random perturbations
        range_tables = [2, 3, 5]
        for num_tables_idx, num_tables in enumerate(range_tables):
            print(f"Generating with {num_tables} tables...")
            datasets[str(num_tables)] = {}

            for method_name in SQL_TEMPLATES:
                if method_name in ["extractive", "comparative", "percentage_change"]:
                    continue

                """list_of_data = []
                list_of_table_hct = []"""

                new_decimals = decimals
                new_units = units
                patience = 5
                for p in range(patience):
                    self.set_random_seed(initial_offset+p)
                    self.perturber.set_random_seed(initial_offset+p)
                    good = True
                    fk_column = None
                    value_to_keep = None
                    cols_to_avoid = []
                    list_of_data = []
                    list_of_table_hct = []

                    for i in range(num_tables):
                        """if (p > 0 and num_tables_idx > 0 and i >= range_tables[num_tables_idx-1]) or (p > 0 and num_tables_idx == 0):
                            self.set_random_seed(initial_offset+p)
                            self.perturber.set_random_seed(initial_offset+p)"""

                        new_value_col = deepcopy(value_col)

                        table_attributes_view, table_attributes_long_view, table_types_view, range_view = self.get_table_view_fk(
                            table_attributes,
                            table_attributes_long,
                            table_types,
                            ranges,
                            new_value_col,
                            fk_column,
                            cols_to_avoid,
                            is_final = True if i == num_tables-1 else False,
                        )

                        if i != num_tables-1:
                            new_value_col = self.rnd.choice([attr for attr in table_attributes_view if attr != fk_column])

                        #try:
                        view_constraints = {"inter_row_constraints": [], "intra_row_constraints": []}
                        for constraint_type in constraints:
                            for rule in constraints[constraint_type]:
                                ok = True
                                if constraint_type == "inter_row_constraints":
                                    conditions = [m.group("cond") for m in re.compile(self.constrainer.ATOM_EXTRACT,
                                                                                      re.VERBOSE | re.DOTALL).finditer(
                                        rule)]
                                else:
                                    # conditions = [rule]
                                    m = self.constrainer.PATTERN_intra.match(rule)
                                    if not m:
                                        raise ValueError(f"Invalid intra rule: {rule!r}")
                                    conditions = [m.group("condition").strip()]
                                    expr = [m.group("expr").strip()]
                                    conditions.extend(expr)

                                for cond in conditions:
                                    cond = cond.strip()
                                    if cond[0] in ["(", ")"]:
                                        cond = cond[1:]
                                    if cond[-1] in ["(", ")"]:
                                        cond = cond[:-1]
                                    columns_from_rule = extract_columns(parse_expr(cond))
                                    for col in columns_from_rule:
                                        if col not in table_attributes_view:
                                            ok = False
                                            break
                                    if not ok:
                                        break

                                if ok:
                                    view_constraints[constraint_type].append(rule)

                        table = self.fill_dense_relational_table(
                            table_attributes_view,
                            table_types_view,
                            range_view,
                            units,
                            decimals,
                            new_value_col, #value_col,
                            id_col,
                            semantic_constraints=view_constraints,
                            col_to_keep=fk_column,
                            value_to_keep=value_to_keep,
                        )
                        new_table = table.copy()
                        new_table = new_table[table_attributes_view].drop_duplicates(keep="first")

                        if fk_column is not None and i != num_tables-1:
                            mask = new_table[fk_column].eq(value_to_keep)
                            idx = new_table.index[mask]
                            to_drop = idx[1:]
                            new_table = new_table.drop(to_drop)

                        #except Exception as e: # TODO: remove comments
                            #print(e)
                        #    errors.append("error in filling table")
                        #    good = False
                        #    break
                            # return None, None, "error in filling table"

                        # run sql loading and sql generation/execution
                        # data, full_mask = self.generate_label(table, table_name, value_col, id_col, method=method)
                        if i != num_tables-1:
                            method_single_table = "extractive"
                        else:
                            method_single_table = method_name

                        data, full_mask = self.generate_label(
                            new_table,
                            table_name,
                            new_value_col,
                            id_col,
                            method=method_single_table,
                            col_to_keep=fk_column,
                            value_to_keep=value_to_keep,
                            impose_target_for_extractive=True if method_single_table == "extractive" else False
                        )

                        if data is None:
                            good = False
                            errors.append(f"error in data generation")
                            break
                            #return None, None, f"error in data generation for the {i+1}-th table"

                        fk_column = data["target"] # column used later to connect to the next table
                        value_to_keep = data["label"]
                        cols_to_avoid.extend([col for col in new_table.columns if col != fk_column])

                        # adding a sample with all perturbations randomly applied with lower strength
                        # new_table = table.copy()
                        new_full_mask = full_mask
                        new_table_types = deepcopy(table_types)

                        if self.rnd.randint(0,1) == 1: # apply perturbation with 50% chance
                            attrs = {
                                "table": new_table,
                                "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                                "value_col": new_value_col,
                            }
                            new_table = self.perturber.null_perturbation(**attrs)

                        try:
                            new_table = self.substitute_column_names(new_table, table_attributes_long_view)
                            new_value_col = table_attributes_long[table_attributes.index(new_value_col)]
                            new_id_col = table_attributes_long[table_attributes.index(id_col)]
                        except Exception as e:
                            good = False
                            errors.append("column name substitution error")
                            break

                        for perturbation in self.perturber.pre_hct_perturbations:
                            if self.rnd.randint(0,1) == 1 and i != 0:
                                attrs = {
                                    "table": new_table,
                                    "value_col": new_value_col,
                                    "columns": data["columns"],
                                    "constraints": data["constraint"],
                                }

                                new_table, new_value_col = perturbation(**attrs)

                        try:
                            unit_in_cell = self.rnd.randint(0, 1)
                            new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                                 unit_in_cell=unit_in_cell)
                            table_before_pivot = new_table.copy()
                            table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table, new_value_col, new_id_col, new_table_types,
                                                                                aggr=aggr,
                                                                                unit_in_cell=unit_in_cell, full_mask=new_full_mask)
                        except:
                            good = False
                            errors.append("error pivoting")
                            break

                        if table_hct is None:
                            good = False
                            errors.append("table generation error (pivoting)")
                            continue
                            #return None, None, "table generation error (pivoting)"

                        table_hct = self.perturber.restore_needed_cells_after_value_merge(
                            table_before_pivot=table_before_pivot,
                            pivot_table=table_hct,
                            value_col=new_value_col,
                            full_mask=full_mask,
                            rows_chosen=rows_chosen,
                            cols_chosen=cols_chosen,
                        )

                        for perturbation in self.perturber.post_hct_perturbations:
                            if perturbation == self.perturber.insert_blank_columns and option == 0: # may cancel unit of measurement
                                continue
                            if self.rnd.randint(0,1) == 1:
                                attrs = {
                                    "table": table_hct
                                }

                                table_hct = perturbation(**attrs)

                        data["decimals"] = new_decimals[new_units[0]]
                        data["units"] = new_units[0]
                        list_of_table_hct.append(table_hct.copy())
                        list_of_data.append(deepcopy(data))

                    if good:
                        """if p > 0 and num_tables_idx == 0:
                            initial_offset += p"""
                        break

                if not good:
                    continue
            #for method_name in SQL_TEMPLATES:
                #if method_name in ["extractive", "comparative", "percentage_change"]:
                #    continue

                if method_name not in datasets[str(num_tables)]:
                    datasets[str(num_tables)][method_name] = {}

                data = self.generate_label_multitable_fk(list_of_data, method=method_name)
                if data is None:
                    errors.append("error in generating multitable data")
                    continue
                    #return None, None, "error in generating multitable data"

                #data["nl_question"] = "sample question"
                #data["nl_question"] = "sample question"

                data["nl_question"], tables_final = self.generate_question_multitable_fk(
                    list_of_table_hct,
                    data,
                    [d["decimals"] for d in list_of_data],
                    method_name,
                ) #list_of_data["decimals"])
                if data["nl_question"] is None:
                    errors.append("error in generating nl question")
                    continue
                    # return None, None, "error in generating nl question"

                #with pd.option_context("display.float_format", lambda x: f"{x:.{list_of_data['decimals']}f}"):
                data["nl_question"] = self.check_nlquestion_validity_fk(
                    data["nl_question"],
                    data, #["query"],
                    tables_final,
                    data["label"],
                    multi=True,
                    unit=list_of_data[0]["units"]
                )
                if data["nl_question"] is None:
                    errors.append("error in checking nl question")
                    continue
                    #return None, None, "error in checking nl question"

                data["decimals"] = [d["decimals"] for d in list_of_data]
                datasets[str(num_tables)][method_name]["sequential"] = (list_of_table_hct, data, constraints)
                #return list_of_table_hct, data

        return datasets, decimals[units[0]], "\n".join(errors) if len(errors) > 0 else None

    def run_ablations(self, num_tables: int = 1, num_samples: int = 1, domain: str | None = None, sequential: bool = False):
        if num_tables == 1:
            datasets, datasets_df, error_logs = {}, {}, []
            perturbations_to_apply = [self.perturber.null_perturbation] + \
                                     self.perturber.pre_hct_perturbations + \
                                     self.perturber.post_hct_perturbations + \
                                     self.intra_ambiguous_perturber.perturbations + \
                                     ["all_perturbations"]

            for method in SQL_TEMPLATES:
                datasets[method] = {}
                datasets_df[method] = {}
                for perturbation in perturbations_to_apply:
                    try:
                        perturbation_name = perturbation.__name__
                    except:
                        perturbation_name = perturbation

                    datasets[method][perturbation_name] = []

            for _ in tqdm(range(num_samples), desc="Generating samples..."):
                result, decimals, error_log = self.run_one_table_ablations(domain=domain)
                if result is None:
                    print(f"Error in sample generation: {error_log}")
                    error_logs.append(error_log)
                    continue
                else:
                    print(result)

                for method in SQL_TEMPLATES:
                    for perturbation in perturbations_to_apply:
                        try:
                            perturbation_name = perturbation.__name__
                        except:
                            perturbation_name = perturbation

                        try:
                            if method in result and perturbation_name in result[method]:
                                with pd.option_context("display.float_format", lambda x: f"{x:.{decimals}f}"):
                                    datasets[method][perturbation_name].append([
                                        result[method][perturbation_name][1]["nl_question"],
                                        result[method][perturbation_name][1]["query"],
                                        method,
                                        result[method][perturbation_name][0].to_html(index=True),
                                        result[method][perturbation_name][2], # constraints
                                        result[method][perturbation_name][1]["label"],
                                    ])
                        except:
                            continue

            for k1 in datasets:
                for k2 in datasets[k1]:
                    datasets_df[k1][k2] = pd.DataFrame(datasets[k1][k2], columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])

            return datasets_df, error_logs
        else:
            datasets, datasets_df, error_logs = {}, {}, {}
            if not sequential:
                perturbation_names = ["unit_converted", "not_unit_converted"]
                ranges = [2,3,5,10,20]
            else:
                perturbation_names = ["sequential"]
                ranges = [2,3,5]

            for num_tables in ranges:
                datasets[str(num_tables)] = {}
                datasets_df[str(num_tables)] = {}
                error_logs[str(num_tables)] = []

                for method in SQL_TEMPLATES:
                    if not sequential and method in ["extractive", "comparative", "percentage_change"]:
                        continue
                    datasets[str(num_tables)][method] = {}
                    datasets_df[str(num_tables)][method] = {}

                    for perturbation_name in perturbation_names:
                        datasets[str(num_tables)][method][perturbation_name] = []

            for _ in tqdm(range(num_samples), desc="Generating samples..."):
                #try:
                if sequential:
                    result, decimals, error_log = self.run_multi_table_ablations_fk(domain=domain, num_tables=num_tables)
                else:
                    result, decimals, error_log = self.run_multi_table_ablations(domain=domain, num_tables=num_tables, col_cardinality=20)
                #except:
                #    continue

                if result is None:
                    print(f"Error in sample generation: {error_log}")
                    error_logs[str(num_tables)].append(error_log)
                    continue
                else:
                    if error_log is not None:
                        print(f"Error in sample generation: {error_log}")
                        error_logs[str(num_tables)].append(error_log)
                    #print(result)

                for num_tables in ranges:
                    for method in SQL_TEMPLATES:
                        if not sequential and method in ["extractive", "comparative", "percentage_change"]:
                            continue

                        for perturbation_name in perturbation_names:
                            #try:
                                if method in result[str(num_tables)] and perturbation_name in result[str(num_tables)][method]:
                                    text = ""
                                    for i, table in enumerate(result[str(num_tables)][method][perturbation_name][0]):
                                        with pd.option_context("display.float_format", lambda x: f"{x:.{result[str(num_tables)][method][perturbation_name][1]['decimals'][i]}f}"):
                                            text += f"Table {i}\n\n{table.to_html(index=True)}\n\n"
                                    text = text.strip()

                                    datasets[str(num_tables)][method][perturbation_name].append([
                                        result[str(num_tables)][method][perturbation_name][1]["nl_question"],
                                        result[str(num_tables)][method][perturbation_name][1]["query"],
                                        method,
                                        text,
                                        #"\n\n".join([f"Table {i}\n\n{table.to_html(index=True)}" for i, table in enumerate(result[method][perturbation_name][0])]),
                                        result[str(num_tables)][method][perturbation_name][2],  # constraints
                                        result[str(num_tables)][method][perturbation_name][1]["label"],
                                    ])
                            #except:
                            #    continue
                print("Current lengths:")
                if 'unit_converted' in datasets['2']['average'] or 'not_unit_converted' in datasets['2']['average']:
                    print(f"\t 2 tables: {len(datasets['2']['average']['unit_converted'])}")
                    print(f"\t 3 tables: {len(datasets['3']['average']['unit_converted'])}")
                    print(f"\t 5 tables: {len(datasets['5']['average']['unit_converted'])}")
                    print(f"\t 10 tables: {len(datasets['10']['average']['unit_converted'])}")
                    print(f"\t 20 tables: {len(datasets['20']['average']['unit_converted'])}")
                else:
                    print(f"\t 2 tables: {len(datasets['2']['average']['sequential'])}")
                    print(f"\t 3 tables: {len(datasets['3']['average']['sequential'])}")
                    print(f"\t 5 tables: {len(datasets['5']['average']['sequential'])}")
                    #print(f"\t 10 tables: {len(datasets['10']['average']['sequential'])}")
                    #print(f"\t 20 tables: {len(datasets['20']['average']['sequential'])}")

            for num_tables in datasets:
                for k1 in datasets[str(num_tables)]:
                    for k2 in datasets[str(num_tables)][k1]:
                        datasets_df[str(num_tables)][k1][k2] = pd.DataFrame(datasets[str(num_tables)][k1][k2],
                                                           columns=["Question", "SQL Query", "Method", "Table",
                                                                    "Constraints", "Label"])

            return datasets_df, error_logs

    def run_bird_comparison(self, num_tables: int = 1, method: str = "average"):
        from spider_table_extraction import find_densest_pivot, extract_tables_from_sqlite_directories
        import pickle

        """with open("bird_tables/tables.pkl", "rb") as reader:
            tables = pickle.load(reader)

        with open("bird_tables/categorical_cols_per_table.pkl", "rb") as reader:
            categorical_cols_per_table = pickle.load(reader)

        with open("bird_tables/float_cols_per_table.pkl", "rb") as reader:
            float_cols_per_table = pickle.load(reader)"""

        tables_dataset = pd.read_csv("pivot_task/exports/qualifying_dense_pivots_verified_moredbs.csv")
        tables_dataset = pd.concat(
            [
                tables_dataset.loc[
                    ((tables_dataset["pivot_rows"] >= 10) | (tables_dataset["pivot_columns"] >= 10)) &
                    ((tables_dataset["num_row_attributes"] > 2) | (tables_dataset["num_column_attributes"] > 2)) &
                    ((tables_dataset["pivot_rows"] > 3) & (tables_dataset["pivot_columns"] > 3))
                    ]
                .groupby(["dataset", "relation"], group_keys=False)
                .apply(lambda g: g.sample(n=min(10, len(g)), random_state=42)),

                tables_dataset.loc[
                    ((tables_dataset["pivot_rows"] >= 10) | (tables_dataset["pivot_columns"] >= 10)) &
                    ((tables_dataset["num_row_attributes"] > 1) | (tables_dataset["num_column_attributes"] > 1)) &
                    ((tables_dataset["pivot_rows"] > 2) & (tables_dataset["pivot_columns"] > 2))
                    ]
                .groupby(["dataset", "relation"], group_keys=False)
                .apply(lambda g: g.sample(n=min(5, len(g)), random_state=42)),
            ],
            ignore_index=True,
        ).reset_index(drop=True)

        categorical_cols_per_table = tables_dataset["row_attributes"].apply(lambda x: [el.strip() for el in x.split(";")])
        categorical_cols_per_table += tables_dataset["column_attributes"].apply(lambda x: [el.strip() for el in x.split(";")])
        float_cols_per_table = tables_dataset["value_attribute"]
        cols_split = tables_dataset["row_attributes"].apply(lambda x: [[el.strip() for el in x.split(";")]])
        cols_split += tables_dataset["column_attributes"].apply(lambda x: [[el.strip() for el in x.split(";")]])
        tables = [pd.read_csv(path) for path in tables_dataset["table_csv_path"]]

        """root_dir = "./bird/train/train_databases/"
        tables, float_cols_per_table, categorical_cols_per_table = extract_tables_from_sqlite_directories(
            root_dir=root_dir,
            seed=123,
            target_n_tables=50,
            min_rows=200,
            min_categorical_cols=4,
            min_float_cols=1,
            drop_duplicates=True,
        )"""

        path = "datasets/bird/"
        os.makedirs(path, exist_ok=True)

        datasets = {}
        ccols, fcols = categorical_cols_per_table, float_cols_per_table

        tables_relational, tables_non_relational = [], []

        for num, (table, ccol, fcol, c_split) in tqdm(enumerate(zip(tables, ccols, fcols, cols_split))):
            if num == 9:
                continue
            # tables may have whitespaces and brackets, that could potentially disrupt SQL templates. For safety, we remove them
            cleaner = lambda x: x.replace(" ", "").replace("(", "").replace(")", "")
            table.columns = table.columns.map(cleaner)
            ccol = [cleaner(c) for c in ccol]
            fcol = cleaner(fcol)

            cols = c_split[0] + c_split[1]
            decimals = len(str(table[fcol].iloc[0]).split(".")[-1])

            table_relational = table.drop_duplicates(subset=cols, keep="first")
            """sub_results = []
            mask_total = pd.Series([False] * len(table_relational))
            for row_pair in best['row_pairs']:
                for col_pair in best['col_pairs']:
                    mask = pd.Series([True] * len(table_relational))
                    pairs = list(row_pair) + list(col_pair)
                    for pair, col in zip(pairs, cols):
                        mask &= (table_relational[col] == pair).reset_index(drop=True)
                    #filtered_table = table_relational[mask.tolist()]

                    mask_total |= mask
                    #sub_results.append(filtered_table)

            table_relational = table_relational[mask_total.tolist()]
            table_relational = table_relational[cols+[fcol]]
            table_relational = table_relational.drop_duplicates(subset=cols)"""
            #table_relational = pd.concat(sub_results)
            tables_relational.append(table_relational)
            new_table = table_relational.copy()
            new_value_col = fcol

            tables_unit_converted, tables_not_unit_converted = [], []
            data_unit_converted, data_not_unit_converted = [], []
            units, decimals = ["unit"], len(str(table[fcol].iloc[0]).split(".")[-1])
            #table[fcol] = table[fcol].apply(lambda x: round(x, decimals))

            if num_tables == 1:
                for j, method in enumerate(SQL_TEMPLATES):
                    if method not in ["average"]:
                        continue
                    #table_rel = table_rel[cols + [fcol]]
                    new_table = table_relational[cols + [fcol]].copy()
                    new_table = new_table.drop_duplicates(subset=cols, keep="first")
                    new_value_col = fcol
                    new_table[new_value_col] = new_table[new_value_col].apply(lambda x: round(x, decimals))

                    if method not in datasets:
                        datasets[method] = {}

                    try:
                        data, full_mask = self.generate_label(
                            new_table,
                            "sample_table",
                            new_value_col,
                            "",
                            method=method,
                        )
                    except:
                        continue

                    if data is None:
                        continue

                    if self.rnd.randint(0, 1) == 1:  # apply perturbation with 50% chance
                        attrs = {
                            "table": new_table,
                            "constraints": list(zip(data["columns"], data["constraint"])),
                            "value_col": new_value_col,
                        }
                        new_table = self.perturber.null_perturbation(**attrs)


                    for perturbation in self.perturber.pre_hct_perturbations:
                        if self.rnd.randint(0, 1) == 1:
                            attrs = {
                                "table": new_table,
                                "value_col": new_value_col
                            }

                            new_table, new_value_col = perturbation(**attrs)


                    try:
                        unit_in_cell = self.rnd.randint(0, 1)
                        new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col,
                                                                                             ["unit"],
                                                                                             unit_in_cell=unit_in_cell)
                        table_before_pivot = new_table.copy()
                        table_hct, rows_chosen, cols_chosen, _ = self.perturber.multiheader_perturbation(new_table,
                                                                                                         new_value_col,
                                                                                                         "",
                                                                                                         [],
                                                                                                         aggr="first",
                                                                                                         unit_in_cell=unit_in_cell,
                                                                                                         full_mask=full_mask)
                    except:
                        continue

                    if table_hct is None:
                        continue


                    table_hct = self.perturber.restore_needed_cells_after_value_merge(
                        table_before_pivot=table_before_pivot,
                        pivot_table=table_hct,
                        value_col=new_value_col,
                        full_mask=full_mask,
                        rows_chosen=rows_chosen,
                        cols_chosen=cols_chosen,
                    )

                    for perturbation in self.perturber.post_hct_perturbations:
                        if self.rnd.randint(0, 1) == 1:
                            attrs = {
                                "table": table_hct
                            }

                            table_hct = perturbation(**attrs)


                    data["nl_question"] = self.generate_question(table_hct, data, decimals, method=method)
                    if data["nl_question"] is None:
                        continue


                    with pd.option_context("display.float_format", lambda x: f"{x:.{decimals}f}"):
                        data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"],
                                                                             table_hct.to_html(index=False), data["label"])
                    if data["nl_question"] is None:
                        continue

                    if "non_relational" not in datasets[method]:
                        datasets[method]["non_relational"] = []
                    if "relational" not in datasets[method]:
                        datasets[method]["relational"] = []

                    data["decimals"] = decimals
                    datasets[method]["non_relational"].append((table_hct, data))
                    datasets[method]["relational"].append((table_relational, data))

                    #if j == 0:
                    print(f"Single table sample number {num} added to dataset")
            else:
                # columns, constraint_cols, column_to_vary, old_col_values, inter_table_contradiction, best_val = None, None, None, [], None, None
                columns, constraint_cols, column_to_vary, old_col_values, best_val, inter_table_contradiction = None, None, None, [], None, False
                old_constraint_cols = None
                good = True
                table = table.drop_duplicates(subset=cols)

                for i in range(num_tables):
                    table_rel = table.copy()
                    table_rel = table_rel[cols+[fcol]]
                    table_rel = table_rel.drop_duplicates(subset=cols)
                    table_rel[cols] = table_rel[cols].fillna("None")
                    table_rel = table_rel.dropna(subset=[fcol])
                    new_value_col = fcol

                    if i != 0:
                        try:
                            best_val = self.choose_constraint_on_new_table(
                                columns,
                                column_to_vary,
                                old_col_values,
                                table_rel,
                                inter_table_contradiction
                            )

                            if old_constraint_cols is not None:
                                constraint_cols = deepcopy(old_constraint_cols)
                                old_constraint_cols = None

                            if isinstance(constraint_cols, tuple):
                                constraint_cols = list(constraint_cols)

                            constraint_cols[column_to_vary] = best_val
                            if len(table_rel[(table_rel[columns] == constraint_cols).all(axis=1)]) == 0:
                                found = False
                                for col_to_change in [i for i in range(len(columns)) if i != column_to_vary]:
                                    #col_to_change = random.choice([i for i in range(len(columns)) if i != column_to_vary])
                                    columns_without_col_to_change = [c for j,c in enumerate(columns) if j != col_to_change]
                                    constraint_cols_without_col_to_change = [c for j,c in enumerate(constraint_cols) if j != col_to_change]
                                    solutions = table_rel[
                                        (table_rel[columns_without_col_to_change] == constraint_cols_without_col_to_change).all(axis=1)
                                    ].reset_index()
                                    old_constraint_cols = deepcopy(constraint_cols)
                                    if len(solutions) > 0:
                                        constraint_cols[col_to_change] = solutions.loc[0,columns[col_to_change]]
                                        found = True
                                        break

                                if not found:
                                    good = False
                                    break

                            """dup_mask = (table_rel[columns] == constraint_cols).all(axis=1)
    
                            match_idx = table_rel.index[dup_mask]
                            if len(match_idx) > 1:
                                table_rel = table_rel.drop(match_idx[1:])"""

                            # rows where the selected columns match the given values
                            mask = (table_rel[columns] == constraint_cols).all(axis=1)

                            # keep all non-matching rows, and deduplicate matching rows
                            table_rel = pd.concat([
                                table_rel[~mask],
                                table_rel[mask].drop_duplicates(subset=columns, keep="first")
                            ], ignore_index=True)
                        except:
                            good = False
                            break

                    try:
                        data, full_mask = self.generate_label(
                            table_rel,
                            "sample_table",
                            new_value_col,
                            id_col = "",
                            method="extractive",
                            # we force the method to be extractive, as the true method will be applied across tables later
                            columns=columns,
                            constraint=constraint_cols,
                            column_to_vary=column_to_vary,
                            old_col_values=old_col_values,
                            inter_table_contradiction=inter_table_contradiction,
                            best_val=best_val,
                            impose_target_for_extractive=True,
                            #all_cols=True
                        )
                    except:
                        good = False
                        break

                    if data is None:
                        good = False
                        break

                    if columns is None and constraint_cols is None:
                        columns = data["columns"]
                        constraint_cols = data["constraint"]

                        # I make sure that inter row constraints are kept even when dealing with multiple tables
                        # for that, in different tables, I vary an attribute that is not part of the inter-row constraints (if any)
                        columns_to_consider = [col for col in columns]
                        # the column to vary must not be the value_col
                        columns_to_consider = [col for col in columns_to_consider if col != new_value_col]
                        # the column to vary must have at least num_tables different values, to ensure that we can vary it across tables
                        columns_to_consider = [col for col in columns_to_consider if table_rel[col].nunique() >= num_tables]
                        if len(columns_to_consider) == 0:
                            good = False
                            break

                        max_val, best_col = -1, None
                        for col in columns_to_consider:
                            pos = columns.index(col)
                            num_samples = len(table_rel[table_rel[columns[pos]] == data["constraint"][pos]])
                            if num_samples > max_val:
                                max_val = num_samples
                                best_col = col

                        column_to_vary = columns.index(best_col)

                    old_col_values.append(data["constraint"][column_to_vary])

                    table_rel = table_rel[table_rel[columns[column_to_vary]] == data["constraint"][column_to_vary]]

                    # adding a sample with all perturbations randomly applied with lower strength
                    for apply_unit_conversions in [True, False]:
                        self.set_random_seed(num+i)
                        self.perturber.set_random_seed(num+i)
                        new_table = table_rel.copy()
                        new_full_mask = full_mask
                        new_value_col = fcol
                        #new_table_types = deepcopy(table_types)

                        #try:
                        if i != 0 and apply_unit_conversions:
                            needed_decimals = -1
                            for j in range(5):
                                surrogate_unit, target_unit = get_random_unit(units[0], "finance") #using financial domain to use millions, thousands etc. notation
                                highest_needed_decimals = -1
                                for k, row in new_table.iterrows():
                                    try:
                                        value_converted, needed_decimals = get_value(
                                            surrogate_unit,
                                            target_unit,
                                            "finance",
                                            round(row[new_value_col], 10), # avoid strange infinite decimals
                                            self.perturber.nan_fill_str
                                        )
                                    except:
                                        good = False
                                        break

                                    if needed_decimals is not None:
                                        if needed_decimals == -1:
                                            break
                                            # return None, None, "couldn't find rounding with the specified number of decimal values"

                                        # new_table.at[k, new_value_col] = round(value_converted, needed_decimals)
                                        new_table.at[k, new_value_col] = value_converted
                                        if needed_decimals > highest_needed_decimals:
                                            highest_needed_decimals = needed_decimals

                                if needed_decimals != -1:
                                    #good = False
                                    break

                            if needed_decimals == -1:
                                good = False
                                break

                            # data["label"] = get_value(surrogate_unit, target_unit, domain, data["label"], self.perturber.nan_fill_str)
                            if surrogate_unit != units[0]:
                                target_unit = f"{target_unit} {units[0].replace('[', '').replace(']', '')}"

                            new_units = [target_unit]
                            new_decimals = max(decimals, highest_needed_decimals) #{target_unit: max(decimals, highest_needed_decimals)}
                        else:
                            new_units = ["units"]
                            new_decimals = decimals
                        """except:
                            good = False
                            break""" # TODO: uncomment

                        if self.rnd.randint(0,1) == 1: # apply perturbation with 50% chance
                            attrs = {
                                "table": new_table,
                                "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                                "value_col": new_value_col,
                            }
                            new_table = self.perturber.null_perturbation(**attrs)

                        for perturbation in self.perturber.pre_hct_perturbations:
                            if self.rnd.randint(0,1) == 1 and i != 0:
                                attrs = {
                                    "table": new_table,
                                    "value_col": new_value_col
                                }

                                new_table, new_value_col = perturbation(**attrs)

                        try:
                            unit_in_cell = self.rnd.randint(0, 1)
                            new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                                 unit_in_cell=unit_in_cell)
                            table_before_pivot = new_table.copy()
                            table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table, new_value_col,
                                                                                "", [],
                                                                                aggr="first",
                                                                                unit_in_cell=unit_in_cell, full_mask=new_full_mask,
                                                                                rows_chosen=c_split[0], cols_chosen=c_split[1])
                        except:
                            good = False
                            break

                        if table_hct is None:
                            good = False
                            break

                        table_hct = self.perturber.restore_needed_cells_after_value_merge(
                            table_before_pivot=table_before_pivot,
                            pivot_table=table_hct,
                            value_col=new_value_col,
                            full_mask=full_mask,
                            rows_chosen=rows_chosen,
                            cols_chosen=cols_chosen,
                        )

                        for perturbation in self.perturber.post_hct_perturbations:
                            if perturbation == self.perturber.insert_blank_columns and option == 0: # may cancel unit of measurement
                                continue
                            if self.rnd.randint(0,1) == 1:
                                attrs = {
                                    "table": table_hct
                                }

                                table_hct = perturbation(**attrs)

                        data["decimals"] = new_decimals
                        data["units"] = new_units[0]
                        if apply_unit_conversions:
                            tables_unit_converted.append(table_hct.copy())
                            data_unit_converted.append(deepcopy(data))
                        else:
                            tables_not_unit_converted.append(table_hct.copy())
                            data_not_unit_converted.append(deepcopy(data))

                if not good:
                    continue

                first = True
                for num_question, (perturbation_name, list_of_table_hct, list_of_data) in enumerate(zip(
                        ["unit_converted", "not_unit_converted"],
                        [tables_unit_converted, tables_not_unit_converted],
                        [data_unit_converted, data_not_unit_converted]
                )):

                    for method in SQL_TEMPLATES:
                        if method not in ["average"]:
                            continue

                        if method not in datasets:
                            datasets[method] = {}

                        data = self.generate_label_multitable(list_of_data, method=method)
                        if data is None:
                            continue

                        data["nl_question"], tables_final = self.generate_question_multitable(
                            list_of_table_hct,
                            data,
                            [d["decimals"] for d in list_of_data],
                            method,
                            unit=list_of_data[0]["units"]
                        )
                        if data["nl_question"] is None:
                            continue

                        data["nl_question"] = self.check_nlquestion_validity(
                            data["nl_question"],
                            data["query"],
                            tables_final,
                            data["label"],
                            multi=True,
                            unit=list_of_data[0]["units"]
                        )
                        if data["nl_question"] is None:
                            continue

                        if perturbation_name not in datasets[method]:
                            datasets[method][perturbation_name] = []

                        data["decimals"] = [d["decimals"] for d in list_of_data]
                        datasets[method][perturbation_name].append((list_of_table_hct, data))

                        if first:
                            print(f"Multi table sample number {num} added to dataset")
                            first = False

        return datasets