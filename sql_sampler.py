import sqlite3
from collections import defaultdict
from itertools import combinations

import pandas as pd
import random

class UnionFind:
    def __init__(self):
        self.parent = dict()

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def clear(self):
        self.parent = dict()

class SQLSampler:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.uf = UnionFind()

    def clear_memory(self):
        self.conn.close()
        self.conn = sqlite3.connect(':memory:')

    def get_unique_row_value_combinations(self, df: pd.DataFrame, columns: list):
        """
        Given a DataFrame and list of columns, return the list of row-wise value combinations
        that are unique (i.e. only appear once across the entire table).

        Args:
            df: DataFrame
            columns: columns to consider for uniqueness

        Returns:
            List of unique value tuples
        """
        # with drop_duplicates, in this function, the rest of the code can be ignored, as the df by definition contains only unique rows
        counts = df.groupby(list(columns)).size()
        unique_combos = counts[counts == 1].index.tolist()
        return unique_combos

    def load_table(self, df: pd.DataFrame, table_name):
        df.to_sql(table_name, self.conn, index=False, if_exists='replace')

    def get_combination(self, df: pd.DataFrame, table_name, value_col, all_cols=False, col_to_keep=None, value_to_keep=None, **kwargs):
        #df.to_sql(table_name, self.conn, index=False, if_exists='replace')
        self.load_table(df, table_name)
        columns = list(df.columns)
        columns.pop(columns.index(value_col))
        #if id_col in columns:
        #    columns.pop(columns.index(id_col))

        all_combos = []
        if all_cols:
            all_combos.extend(combinations(columns, len(columns)))
        else:
            for r in range(2, 5): #len(columns) + 1): #capping to 5 to avoid combinatorial explosion
                all_combos.extend(combinations(columns, r))

        if col_to_keep is not None:
            if col_to_keep not in columns and col_to_keep != value_col:
                raise ValueError("Column {} not found in table".format(col_to_keep))
            all_combos = [combo for combo in all_combos if col_to_keep in combo]

        unique_combo = []
        combo = []
        for combo in all_combos:
            unique_combo = self.get_unique_row_value_combinations(df, combo)
            if col_to_keep is not None:
                col_pos = combo.index(col_to_keep)
                unique_combo = [el for el in unique_combo if el[col_pos] == value_to_keep]
            if len(unique_combo) > 0:
                break

        return unique_combo, list(combo)


    def get_extractive_clusters(self, table: pd.DataFrame, table_name: str, id_col: str):
        """
        trying to get value pairs that are bijective, so that when we ask a natural language question, the response is unique.
        we use union-find to detect the clusters of connected values (by "connected" we mean that the values are in a one-to-one relationship between each other)
        """
        table.to_sql(table_name, self.conn, index=False, if_exists='replace')
        cursor = self.conn.cursor()

        edges = []

        columns = table.columns.tolist()

        for i, col1 in enumerate(columns):
            if col1 == id_col:
                continue
            for col2 in columns[i + 1:]:
                if col2 == id_col:
                    continue
                # check for one-to-one mapping between col1 and col2
                query = f"""
                    SELECT {col1}, {col2}
                    FROM {table_name}
                    GROUP BY {col1}
                    HAVING COUNT(*) = 1
                """
                cursor.execute(query)
                forward = set(cursor.fetchall())

                # the opposite
                query = f"""
                    SELECT {col1}, {col2}
                    FROM {table_name}
                    GROUP BY {col2}
                    HAVING COUNT(*) = 1
                """
                cursor.execute(query)
                backward = set(cursor.fetchall())

                bijective = forward & backward
                for val1, val2 in bijective:
                    edges.append(((col1, val1), (col2, val2)))

        self.uf.clear()
        for a, b in edges:
            self.uf.union(a, b)

        clusters = defaultdict(list)
        for node in self.uf.parent:
            root = self.uf.find(node)
            clusters[root].append(node)

        return clusters

    def sample_from_cluster(self, clusters):
        cluster_values = list(clusters.values())
        good_clusters = [value for value in cluster_values if len(value) > 1]
        if len(good_clusters) == 0:
            return None, None

        rnd_cluster = random.randint(0, len(good_clusters)-1) # choosing random cluster
        num_attr = random.randint(2, len(good_clusters[rnd_cluster])) # choosing random amount of attributes for the query
        chosen_attr = random.sample(good_clusters[rnd_cluster], num_attr) # selecting the attributes (target + constraints)
        target_idx = random.randint(0, num_attr-1) # choosing which attribute will be requested in the select clause
        target = chosen_attr[target_idx]
        chosen_attr.pop(target_idx)

        return target, chosen_attr

    def execute(self, query):
        return pd.read_sql_query(query, self.conn)