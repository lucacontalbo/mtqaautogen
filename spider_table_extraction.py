import os
import random
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

from itertools import combinations

"""def find_densest_pivot(
    df: pd.DataFrame,
    categorical_cols: list[str],
    value_col: str,
    min_rows: int = 3,
    max_rows: int = 10,
    min_cols: int = 3,
    max_cols: int = 10,
    aggfunc="first",
):

    best = None

    # Need at least 4 categorical columns total
    if len(categorical_cols) < 4:
        return None

    # Choose 4 columns, then partition them into 2 row cols and 2 col cols
    for four_cols in combinations(categorical_cols, 4):
        four_cols = list(four_cols)

        # choose 2 for rows, remaining 2 for cols
        for row_cols in combinations(four_cols, 2):
            row_cols = list(row_cols)
            col_cols = [c for c in four_cols if c not in row_cols]

            # avoid symmetric duplicates by imposing an order
            if tuple(row_cols) > tuple(col_cols):
                continue

            needed = row_cols + col_cols + [value_col]
            tmp = df[needed].dropna(subset=needed).copy()
            if tmp.empty:
                continue

            # Count presence of each 4-way combination
            counts = (
                tmp.groupby(row_cols + col_cols, observed=False)
                .size()
                .reset_index(name="count")
            )

            if counts.empty:
                continue

            # Build binary presence matrix:
            # rows are tuples from row_cols, columns are tuples from col_cols
            presence_df = counts.pivot_table(
                index=row_cols,
                columns=col_cols,
                values="count",
                aggfunc="sum",
                fill_value=0,
            )

            presence = (presence_df > 0).astype(int)

            if presence.shape[0] < min_rows or presence.shape[1] < min_cols:
                continue

            current = presence.copy()

            # First trim to max size
            while current.shape[0] > max_rows or current.shape[1] > max_cols:
                row_fill = current.sum(axis=1) / current.shape[1]
                col_fill = current.sum(axis=0) / current.shape[0]

                remove_row_score = row_fill.min() if current.shape[0] > max_rows else np.inf
                remove_col_score = col_fill.min() if current.shape[1] > max_cols else np.inf

                if remove_row_score <= remove_col_score:
                    current = current.drop(index=row_fill.idxmin())
                else:
                    current = current.drop(columns=col_fill.idxmin())

            # Then greedily improve density by dropping sparse rows/cols
            improved = True
            while improved:
                improved = False
                base_density = current.values.mean()

                row_fill = current.sum(axis=1) / current.shape[1]
                col_fill = current.sum(axis=0) / current.shape[0]

                best_candidate = None

                if current.shape[0] > min_rows:
                    worst_row = row_fill.idxmin()
                    cand = current.drop(index=worst_row)
                    cand_density = cand.values.mean()
                    if cand_density > base_density:
                        best_candidate = ("row", worst_row, cand_density, cand)

                if current.shape[1] > min_cols:
                    worst_col = col_fill.idxmin()
                    cand = current.drop(columns=worst_col)
                    cand_density = cand.values.mean()
                    if cand_density > base_density:
                        if best_candidate is None or cand_density > best_candidate[2]:
                            best_candidate = ("col", worst_col, cand_density, cand)

                if best_candidate is not None:
                    current = best_candidate[3]
                    improved = True

            if not (min_rows <= current.shape[0] <= max_rows and min_cols <= current.shape[1] <= max_cols):
                continue

            # Current selected row/col keys
            selected_row_keys = list(current.index)
            selected_col_keys = list(current.columns)

            # Convert selected MultiIndex keys into masks
            row_mask = pd.Series(False, index=tmp.index)
            for key in selected_row_keys:
                if not isinstance(key, tuple):
                    key = (key,)
                cond = pd.Series(True, index=tmp.index)
                for c, v in zip(row_cols, key):
                    cond &= (tmp[c] == v)
                row_mask |= cond

            col_mask = pd.Series(False, index=tmp.index)
            for key in selected_col_keys:
                if not isinstance(key, tuple):
                    key = (key,)
                cond = pd.Series(True, index=tmp.index)
                for c, v in zip(col_cols, key):
                    cond &= (tmp[c] == v)
                col_mask |= cond

            filtered = tmp[row_mask & col_mask].copy()
            if filtered.empty:
                continue

            pivot = pd.pivot_table(
                filtered,
                index=row_cols,
                columns=col_cols,
                values=value_col,
                aggfunc=aggfunc,
            )

            # Reindex to preserve exactly the selected structure
            row_index = pd.MultiIndex.from_tuples(selected_row_keys, names=row_cols)
            col_index = pd.MultiIndex.from_tuples(selected_col_keys, names=col_cols)
            pivot = pivot.reindex(index=row_index, columns=col_index)

            n_non_null = int(pivot.notna().sum().sum())
            total = pivot.shape[0] * pivot.shape[1]
            density = n_non_null / total if total > 0 else 0.0

            result = {
                "row_cols": row_cols,
                "col_cols": col_cols,
                "row_keys": selected_row_keys,
                "col_keys": selected_col_keys,
                "pivot": pivot,
                "density": density,
                "n_non_null": n_non_null,
                "shape": pivot.shape,
            }

            if best is None:
                best = result
            else:
                # prefer higher density, then more filled cells, then larger pivot
                if (
                    result["density"] > best["density"]
                    or (
                        result["density"] == best["density"]
                        and result["n_non_null"] > best["n_non_null"]
                    )
                    or (
                        result["density"] == best["density"]
                        and result["n_non_null"] == best["n_non_null"]
                        and (result["shape"][0] * result["shape"][1] > best["shape"][0] * best["shape"][1])
                    )
                ):
                    best = result

    return best"""

import pandas as pd
import numpy as np
from itertools import combinations

def _rank_categorical_columns(df, categorical_cols, min_card=3, max_card=50, top_k=10):
    """
    Cheap heuristic to keep only promising categorical columns.
    """
    scored = []

    n = len(df)
    for col in categorical_cols:
        s = df[col]
        vc = s.value_counts(dropna=True)

        nunique = len(vc)
        if nunique < min_card:
            continue

        # Prefer moderate cardinality and non-extreme skew
        top_mass = vc.iloc[:10].sum() / n if n else 0
        score = 0.0

        # columns with too many unique values are usually bad for dense pivots
        if nunique <= max_card:
            score += 2.0
        else:
            score += max(0.0, 2.0 - nunique / 200.0)

        # prefer columns where a few values cover much of the data
        score += top_mass

        # prefer columns that can plausibly form 3-10 categories
        score += 1.0 if nunique >= 3 else 0.0

        scored.append((col, score, nunique))

    scored.sort(key=lambda x: (-x[1], x[2]))
    return [c for c, _, _ in scored[:top_k]]


def _encode_categoricals(df, categorical_cols):
    """
    Encode categorical columns once as int32 codes.
    Missing values -> -1
    """
    codes = {}
    categories = {}
    for col in categorical_cols:
        cat = pd.Categorical(df[col])
        codes[col] = cat.codes.astype(np.int32)
        categories[col] = cat.categories
    return codes, categories


def _make_pair_ids(a_codes, b_codes):
    """
    Create dense integer ids for pairs (a,b), ignoring rows with missing codes.
    Returns:
      valid_mask, pair_ids, unique_pairs
    """
    valid = (a_codes >= 0) & (b_codes >= 0)
    if not valid.any():
        return valid, np.array([], dtype=np.int32), np.empty((0, 2), dtype=np.int32)

    pairs = np.column_stack((a_codes[valid], b_codes[valid]))
    unique_pairs, pair_ids = np.unique(pairs, axis=0, return_inverse=True)
    return valid, pair_ids.astype(np.int32), unique_pairs.astype(np.int32)


def _presence_from_quad(row_ids, col_ids):
    """
    Build binary presence matrix from row-pair ids and col-pair ids.
    """
    if len(row_ids) == 0 or len(col_ids) == 0:
        return None

    n_rows = int(row_ids.max()) + 1
    n_cols = int(col_ids.max()) + 1

    flat = row_ids.astype(np.int64) * n_cols + col_ids.astype(np.int64)
    occupied = np.unique(flat)

    mat = np.zeros((n_rows, n_cols), dtype=np.uint8)
    rr = occupied // n_cols
    cc = occupied % n_cols
    mat[rr, cc] = 1
    return mat


def _trim_dense_submatrix(mat, min_rows=3, max_rows=10, min_cols=3, max_cols=10):
    """
    Greedily trim sparse rows/cols.
    Works on a binary numpy matrix.
    Returns selected row indices, selected col indices, density.
    """
    if mat is None:
        return None

    row_idx = np.arange(mat.shape[0])
    col_idx = np.arange(mat.shape[1])
    cur = mat

    if cur.shape[0] < min_rows or cur.shape[1] < min_cols:
        return None

    # trim down to max sizes
    while cur.shape[0] > max_rows or cur.shape[1] > max_cols:
        row_fill = cur.mean(axis=1)
        col_fill = cur.mean(axis=0)

        remove_row_score = row_fill.min() if cur.shape[0] > max_rows else np.inf
        remove_col_score = col_fill.min() if cur.shape[1] > max_cols else np.inf

        if remove_row_score <= remove_col_score:
            k = row_fill.argmin()
            keep = np.ones(cur.shape[0], dtype=bool)
            keep[k] = False
            row_idx = row_idx[keep]
            cur = cur[keep, :]
        else:
            k = col_fill.argmin()
            keep = np.ones(cur.shape[1], dtype=bool)
            keep[k] = False
            col_idx = col_idx[keep]
            cur = cur[:, keep]

    # improve density greedily
    improved = True
    while improved:
        improved = False
        base_density = cur.mean()

        best_gain = 0.0
        best_axis = None
        best_k = None

        if cur.shape[0] > min_rows:
            row_fill = cur.mean(axis=1)
            k = row_fill.argmin()
            cand = np.delete(cur, k, axis=0)
            gain = cand.mean() - base_density
            if gain > best_gain:
                best_gain = gain
                best_axis = 0
                best_k = k

        if cur.shape[1] > min_cols:
            col_fill = cur.mean(axis=0)
            k = col_fill.argmin()
            cand = np.delete(cur, k, axis=1)
            gain = cand.mean() - base_density
            if gain > best_gain:
                best_gain = gain
                best_axis = 1
                best_k = k

        if best_axis is not None:
            improved = True
            if best_axis == 0:
                keep = np.ones(cur.shape[0], dtype=bool)
                keep[best_k] = False
                row_idx = row_idx[keep]
                cur = cur[keep, :]
            else:
                keep = np.ones(cur.shape[1], dtype=bool)
                keep[best_k] = False
                col_idx = col_idx[keep]
                cur = cur[:, keep]

    if cur.shape[0] < min_rows or cur.shape[1] < min_cols:
        return None

    density = float(cur.mean())
    n_non_null = int(cur.sum())

    return {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "density": density,
        "n_non_null": n_non_null,
        "shape": cur.shape,
    }


def find_densest_pivot(
    df,
    categorical_cols,
    value_col,
    min_rows=3,
    max_rows=10,
    min_cols=3,
    max_cols=10,
    aggfunc="mean",
    top_k_cols=8,
):
    """
    Fast search:
      - shortlist columns
      - search using encoded integer ids and binary presence matrices
      - build the real pivot only once for the best candidate
    """

    candidate_cols = _rank_categorical_columns(
        df, categorical_cols, min_card=3, max_card=60, top_k=top_k_cols
    )


    work = df[candidate_cols + [value_col]].dropna(subset=[value_col]).copy()
    if work.empty:
        return None

    codes, categories = _encode_categoricals(work, candidate_cols)

    best = None

    for four in combinations(candidate_cols, 4):
        four = list(four)

        for row_cols in combinations(four, 2):
            row_cols = list(row_cols)
            col_cols = [c for c in four if c not in row_cols]

            if tuple(row_cols) > tuple(col_cols):
                continue

            # pair ids for row MultiIndex
            valid_r, row_ids, row_pairs = _make_pair_ids(codes[row_cols[0]], codes[row_cols[1]])
            if len(row_pairs) < min_rows:
                continue

            # pair ids for column MultiIndex
            valid_c, col_ids, col_pairs = _make_pair_ids(codes[col_cols[0]], codes[col_cols[1]])
            if len(col_pairs) < min_cols:
                continue

            valid = valid_r & valid_c
            if valid.sum() == 0:
                continue

            # recompute pair ids on the jointly valid rows only
            r_pairs = np.column_stack((codes[row_cols[0]][valid], codes[row_cols[1]][valid]))
            c_pairs = np.column_stack((codes[col_cols[0]][valid], codes[col_cols[1]][valid]))

            row_pairs_u, row_ids = np.unique(r_pairs, axis=0, return_inverse=True)
            col_pairs_u, col_ids = np.unique(c_pairs, axis=0, return_inverse=True)

            if len(row_pairs_u) < min_rows or len(col_pairs_u) < min_cols:
                continue

            # cheap cardinality guard
            if len(row_pairs_u) > 200 or len(col_pairs_u) > 200:
                continue

            mat = _presence_from_quad(row_ids, col_ids)
            trimmed = _trim_dense_submatrix(
                mat,
                min_rows=min_rows,
                max_rows=max_rows,
                min_cols=min_cols,
                max_cols=max_cols,
            )
            if trimmed is None:
                continue

            result = {
                "row_cols": row_cols,
                "col_cols": col_cols,
                "row_pairs_codes": row_pairs_u[trimmed["row_idx"]],
                "col_pairs_codes": col_pairs_u[trimmed["col_idx"]],
                "density": trimmed["density"],
                "n_non_null": trimmed["n_non_null"],
                "shape": trimmed["shape"],
                "score": trimmed["density"] + 0.01 * trimmed["n_non_null"],
            }

            if best is None or result["score"] > best["score"]:
                best = result

    if best is None:
        return None

    # Build the real pivot only once for the winner
    row_cols = best["row_cols"]
    col_cols = best["col_cols"]

    # Decode selected pairs back to category values
    selected_row_pairs = []
    for a, b in best["row_pairs_codes"]:
        selected_row_pairs.append((
            categories[row_cols[0]][a],
            categories[row_cols[1]][b],
        ))

    selected_col_pairs = []
    for a, b in best["col_pairs_codes"]:
        selected_col_pairs.append((
            categories[col_cols[0]][a],
            categories[col_cols[1]][b],
        ))

    needed = row_cols + col_cols + [value_col]
    tmp = work[needed].dropna()

    row_pair_set = set(selected_row_pairs)
    col_pair_set = set(selected_col_pairs)

    row_tuples = list(zip(tmp[row_cols[0]], tmp[row_cols[1]]))
    col_tuples = list(zip(tmp[col_cols[0]], tmp[col_cols[1]]))

    mask = np.array(
        [(r in row_pair_set) and (c in col_pair_set) for r, c in zip(row_tuples, col_tuples)],
        dtype=bool,
    )

    filtered = tmp.loc[mask]
    if filtered.empty:
        return None

    pivot = pd.pivot_table(
        filtered,
        index=row_cols,
        columns=col_cols,
        values=value_col,
        aggfunc=aggfunc,
    )

    row_index = pd.MultiIndex.from_tuples(selected_row_pairs, names=row_cols)
    col_index = pd.MultiIndex.from_tuples(selected_col_pairs, names=col_cols)
    pivot = pivot.reindex(index=row_index, columns=col_index)

    n_non_null = int(pivot.notna().sum().sum())
    total = pivot.shape[0] * pivot.shape[1]
    density = n_non_null / total if total else 0.0

    best["pivot"] = pivot
    best["density"] = density
    best["n_non_null"] = n_non_null
    best["shape"] = pivot.shape
    best["row_pairs"] = selected_row_pairs
    best["col_pairs"] = selected_col_pairs

    return best

def _is_float_dtype(sql_type: str) -> bool:
    """
    SQLite-style type affinity check for float-like columns.
    """
    t = (sql_type or "").strip().upper()
    float_keywords = ["REAL", "FLOAT", "DOUBLE", "DOUBLE PRECISION", "DECIMAL", "NUMERIC"]
    return any(k in t for k in float_keywords)


def _is_categorical_dtype(sql_type: str) -> bool:
    """
    Treat text-like / boolean / date-like columns as categorical.

    SQLite is permissive with types, so this is heuristic-based.
    """
    t = (sql_type or "").strip().upper()
    categorical_keywords = [
        "CHAR", "CLOB", "TEXT", "STRING", "VARCHAR",
        "BOOL", "BOOLEAN",
        "DATE", "DATETIME", "TIME", "TIMESTAMP",
    ]
    return any(k in t for k in categorical_keywords)


def _get_user_tables(conn: sqlite3.Connection) -> List[str]:
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
      AND name NOT LIKE 'sqlite_%'
    ORDER BY name
    """
    return pd.read_sql_query(q, conn)["name"].tolist()


def _get_table_schema(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f'PRAGMA table_info("{table_name}")', conn)


def _get_table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    q = f'SELECT COUNT(*) AS n FROM "{table_name}"'
    return int(pd.read_sql_query(q, conn)["n"].iloc[0])


def _find_sqlite_file(directory: Path) -> Optional[Path]:
    """
    Return the first .sqlite file found directly inside `directory`, or None.
    """
    sqlite_files = sorted(directory.glob("*.sqlite"))
    return sqlite_files[0] if sqlite_files else None


def extract_tables_from_sqlite_directories(
    root_dir: str,
    seed: int = 42,
    target_n_tables: int = 50,
    min_rows: int = 200,
    min_categorical_cols: int = 3,
    min_float_cols: int = 1,
    drop_duplicates: bool = True,
) -> Tuple[List[pd.DataFrame], List[List[str]], List[List[str]]]:
    """
    Traverse immediate subdirectories of `root_dir` in sorted order.
    In each subdirectory, if a .sqlite file exists, inspect its tables.

    Collect tables sequentially until `target_n_tables` valid tables are found.

    A valid table must have:
      - at least `min_rows` rows
      - at least `min_float_cols` float-like columns
      - at least `min_categorical_cols` categorical-like columns

    Returned tables keep ALL columns.

    Returns:
        result_tables: list of pandas DataFrames
        float_column_names_per_table: list of float-column-name lists
        categorical_column_names_per_table: list of categorical-column-name lists
    """
    random.seed(seed)
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root_dir}")

    result_tables: List[pd.DataFrame] = []
    float_column_names_per_table: List[List[str]] = []
    categorical_column_names_per_table: List[List[str]] = []

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for subdir in tqdm(subdirs):
        if len(result_tables) >= target_n_tables:
            break

        sqlite_file = _find_sqlite_file(subdir)
        if sqlite_file is None:
            continue

        conn = None
        try:
            conn = sqlite3.connect(str(sqlite_file))
            table_names = _get_user_tables(conn)

            for table_name in table_names:
                if len(result_tables) >= target_n_tables:
                    break

                try:
                    schema = _get_table_schema(conn, table_name)
                except Exception:
                    continue

                if schema.empty:
                    continue

                try:
                    row_count = _get_table_row_count(conn, table_name)
                except Exception:
                    continue

                if row_count < min_rows:
                    continue

                float_cols = []
                categorical_cols = []

                for _, row in schema.iterrows():
                    col_name = row["name"]
                    col_type = row["type"]

                    if _is_float_dtype(col_type):
                        float_cols.append(col_name)
                    elif _is_categorical_dtype(col_type):
                        categorical_cols.append(col_name)

                if len(float_cols) < min_float_cols:
                    continue
                if len(categorical_cols) < min_categorical_cols:
                    continue

                try:
                    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                except Exception:
                    continue

                if drop_duplicates:
                    df = df.drop_duplicates().reset_index(drop=True)

                # Enforce row count after duplicate removal too.
                # Remove this block if you only care about raw SQL row count.
                if len(df) < min_rows:
                    continue

                result_tables.append(df)
                float_column_names_per_table.append(float_cols)
                categorical_column_names_per_table.append(categorical_cols)

        except sqlite3.Error:
            continue
        finally:
            if conn is not None:
                conn.close()

    if len(result_tables) < target_n_tables:
        raise ValueError(
            f"Only found {len(result_tables)} valid tables before exhausting directories, "
            f"but {target_n_tables} were requested."
        )

    return result_tables, float_column_names_per_table, categorical_column_names_per_table


if __name__ == "__main__":
    root_dir = "./bird/train/train_databases/"

    tables, float_cols_per_table, categorical_cols_per_table = extract_tables_from_sqlite_directories(
        root_dir=root_dir,
        seed=123,
        target_n_tables=54,
        min_rows=200,
        min_categorical_cols=4,
        min_float_cols=1,
        drop_duplicates=True,
    )

    print(f"Number of returned tables: {len(tables)}")

    for i, (df, fcols, ccols) in enumerate(zip(tables[:3], float_cols_per_table[:3], categorical_cols_per_table[:3])):
        print(f"\n=== Table #{i} ===")
        print("Shape:", df.shape)
        print("Float columns:", fcols)
        print("Categorical columns:", ccols[:10])
        print(df.head())
        print(df)
        print(ccols)
        print(fcols)
        best = find_densest_pivot(df, ccols, fcols[0])

        if best is None:
            print("No valid 2x2 MultiIndex pivot found")
        else:
            print("Row column:", best["row_cols"])
            print("Column column:", best["col_cols"])
            print("Selected row values:", best["row_pairs"])
            print("Selected column values:", best["col_pairs"])
            print("Shape:", best["shape"])
            print("Density:", best["density"])
            print(best["pivot"])
        print()
        print()


    path = "bird_tables/"
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "tables.pkl"), "wb") as writer:
        pickle.dump(tables, writer)

    with open(os.path.join(path, "float_cols_per_table.pkl"), "wb") as writer:
        pickle.dump(float_cols_per_table, writer)

    with open(os.path.join(path, "categorical_cols_per_table.pkl"), "wb") as writer:
        pickle.dump(categorical_cols_per_table, writer)


