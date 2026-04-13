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
    min_rows=5,
    max_rows=10,
    min_cols=5,
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
    if len(categorical_cols) < 4:
        return None

    candidate_cols = _rank_categorical_columns(
        df, categorical_cols, min_card=3, max_card=60, top_k=top_k_cols
    )

    if len(candidate_cols) < 4:
        return None

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