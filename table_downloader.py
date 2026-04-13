#!/usr/bin/env python3

import itertools
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path("pivot_task")
DOWNLOAD_DIR = BASE_DIR / "downloads"
DB_DIR = BASE_DIR / "dbs"
EXPORT_DIR = BASE_DIR / "exports"
TABLE_CSV_DIR = EXPORT_DIR / "tables_csv"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_CSV_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS = 3
MIN_COLS = 3
MIN_DENSITY = 0.95

MAX_ROW_ATTRS = 4
MAX_COL_ATTRS = 4
MAX_UNIQUE_PER_AXIS_ATTR = 60

MAX_RESULTS_PER_RELATION = None

# Small speed guard so the search does not explode on very large cartesian products.
MAX_ESTIMATED_CARTESIAN_CELLS = 20000


@dataclass
class DatasetSpec:
    dataset_name: str
    db_filename: str
    sql_sources: List[Tuple[str, str]]
    post_build_sql: Optional[str] = None


BIKESTORES_POST_BUILD_SQL = """
DROP VIEW IF EXISTS sales_enriched;
CREATE VIEW sales_enriched AS
SELECT
    oi.order_id,
    oi.item_id,
    oi.quantity,
    oi.list_price,
    oi.discount,
    o.order_status,
    substr(o.order_date, 1, 7) AS order_month,
    o.store_id,
    o.staff_id,
    c.city AS customer_city,
    c.state AS customer_state,
    st.store_name,
    st.city AS store_city,
    st.state AS store_state,
    p.product_name,
    p.model_year,
    b.brand_name,
    cat.category_name
FROM order_items oi
JOIN orders o
    ON oi.order_id = o.order_id
JOIN products p
    ON oi.product_id = p.product_id
JOIN brands b
    ON p.brand_id = b.brand_id
JOIN categories cat
    ON p.category_id = cat.category_id
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN stores st
    ON o.store_id = st.store_id;
"""

BOOTCAMP_POST_BUILD_SQL = """
DROP VIEW IF EXISTS sales_enriched;
CREATE VIEW sales_enriched AS
SELECT
    f.sales_id,
    d.month_name,
    d.month,
    d.quarter,
    d.year,
    d.is_weekend,
    c.gender,
    c.country AS customer_country,
    c.state AS customer_state,
    c.city AS customer_city,
    p.category,
    p.brand,
    p.product_name,
    s.region,
    s.country AS store_country,
    s.city AS store_city,
    f.quantity_sold,
    f.unit_price,
    f.discount,
    f.total_amount
FROM fact_sales f
JOIN dim_date d
    ON f.date_key = d.date_key
JOIN dim_customer c
    ON f.customer_key = c.customer_key
JOIN dim_product p
    ON f.product_key = p.product_key
JOIN dim_store s
    ON f.store_key = s.store_key;
"""


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        dataset_name="northwind",
        db_filename="northwind.db",
        sql_sources=[
            (
                "northwind_create.sql",
                "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/src/create.sql",
            )
        ],
    ),
    DatasetSpec(
        dataset_name="sakila",
        db_filename="sakila.db",
        sql_sources=[
            (
                "sakila_schema.sql",
                "https://github.com/bradleygrant/sakila-sqlite3/raw/refs/heads/main/source/sqlite-sakila-schema.sql",
            ),
            (
                "sakila_insert.sql",
                "https://github.com/bradleygrant/sakila-sqlite3/raw/refs/heads/main/source/sqlite-sakila-insert-data.sql",
            ),
        ],
    ),
    DatasetSpec(
        dataset_name="bikestores",
        db_filename="bikestores.db",
        sql_sources=[
            (
                "bike_create.sql",
                "https://raw.githubusercontent.com/asynched/bike-stores-sqlite/main/sql/create.sql",
            ),
            (
                "bike_insert.sql",
                "https://raw.githubusercontent.com/asynched/bike-stores-sqlite/main/sql/insert.sql",
            ),
        ],
        post_build_sql=BIKESTORES_POST_BUILD_SQL,
    ),
    DatasetSpec(
        dataset_name="sql_bootcamp",
        db_filename="sql_bootcamp.db",
        sql_sources=[
            (
                "source.sql",
                "https://raw.githubusercontent.com/anshlambagit/SQL_Bootcamp/main/Source_Setup/source.sql",
            )
        ],
        post_build_sql=BOOTCAMP_POST_BUILD_SQL,
    ),
]


def download_file(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def build_sqlite_db_from_sql(
    db_path: Path,
    sql_files: List[Path],
    post_build_sql: Optional[str] = None,
) -> None:
    if db_path.exists():
        print(f"Database already exists: {db_path}")
        return

    print(f"Building database from SQL: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        for sql_file in sql_files:
            print(f"  Applying {sql_file.name}")
            sql_text = sql_file.read_text(encoding="utf-8", errors="ignore")
            cur.executescript(sql_text)
            con.commit()

        if post_build_sql:
            print("  Applying post-build SQL")
            cur.executescript(post_build_sql)
            con.commit()
    finally:
        con.close()


def ensure_dataset(spec: DatasetSpec) -> Path:
    db_path = DB_DIR / spec.db_filename

    sql_paths: List[Path] = []
    for filename, url in spec.sql_sources:
        dst = DOWNLOAD_DIR / spec.dataset_name / filename
        download_file(url, dst)
        sql_paths.append(dst)

    build_sqlite_db_from_sql(db_path, sql_paths, post_build_sql=spec.post_build_sql)
    return db_path


def get_relation_names(con: sqlite3.Connection) -> List[Tuple[str, str]]:
    q = """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
    """
    rows = pd.read_sql_query(q, con)
    return list(rows.itertuples(index=False, name=None))


def read_relation(con: sqlite3.Connection, relation_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{relation_name}"', con)


def numeric_conversion(series: pd.Series) -> Tuple[pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    coverage = float(numeric.notna().mean())
    return numeric, coverage


def is_float_attribute(series: pd.Series) -> bool:
    numeric, coverage = numeric_conversion(series)
    numeric = numeric.dropna()
    if coverage < 0.95 or numeric.empty:
        return False
    return bool(((numeric % 1) != 0).any())


def is_integer_attribute(series: pd.Series) -> bool:
    numeric, coverage = numeric_conversion(series)
    numeric = numeric.dropna()
    if coverage < 0.95 or numeric.empty:
        return False
    return bool(np.allclose(numeric, np.round(numeric)))


def is_axis_attribute(series: pd.Series) -> bool:
    non_null = series.dropna()
    nunique = non_null.nunique()

    if nunique < 3:
        return False
    if nunique > MAX_UNIQUE_PER_AXIS_ATTR:
        return False

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if is_integer_attribute(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True

    return False


def normalize_axis_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    return v


def export_relation_csv(df: pd.DataFrame, dataset_name: str, relation_name: str) -> str:
    safe_name = f"{dataset_name}__{relation_name.replace(' ', '_').replace('/', '_')}.csv"
    out_path = TABLE_CSV_DIR / safe_name
    if not out_path.exists():
        df.to_csv(out_path, index=False)
    return str(out_path)


def candidate_axis_combinations(cols: List[str], max_size: int) -> List[Tuple[str, ...]]:
    combos: List[Tuple[str, ...]] = []
    for size in range(1, min(max_size, len(cols)) + 1):
        combos.extend(itertools.combinations(cols, size))
    return combos


def disjoint(a: Sequence[str], b: Sequence[str]) -> bool:
    return set(a).isdisjoint(set(b))


def pivot_density(pivot: pd.DataFrame) -> float:
    return float(pivot.notna().to_numpy().mean())


def estimated_cartesian_cells(
    df: pd.DataFrame,
    row_attrs: Sequence[str],
    col_attrs: Sequence[str],
) -> int:
    row_prod = 1
    for c in row_attrs:
        row_prod *= max(1, int(df[c].dropna().nunique()))

    col_prod = 1
    for c in col_attrs:
        col_prod *= max(1, int(df[c].dropna().nunique()))

    return row_prod * col_prod


def score_result(result: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        result["non_nan_density"],
        result["num_row_attributes"] + result["num_column_attributes"],
        min(result["num_row_attributes"], result["num_column_attributes"]),
        result["pivot_cells"],
        result["pivot_rows"],
        result["pivot_columns"],
    )


def find_all_valid_pivots(
    df: pd.DataFrame,
    dataset_name: str,
    relation_name: str,
) -> List[Dict[str, Any]]:
    if df.shape[0] < 9 or df.shape[1] < 3:
        return []

    work_df = df.copy()

    value_candidates: List[str] = []
    axis_candidates: List[str] = []

    for col in work_df.columns:
        s = work_df[col]
        non_null = s.dropna()
        if non_null.empty or non_null.nunique() < 3:
            continue

        if is_float_attribute(s):
            value_candidates.append(col)
        elif is_axis_attribute(s):
            axis_candidates.append(col)

    if not value_candidates or len(axis_candidates) < 2:
        return []

    row_combos = candidate_axis_combinations(axis_candidates, MAX_ROW_ATTRS)
    col_combos = candidate_axis_combinations(axis_candidates, MAX_COL_ATTRS)

    results: List[Dict[str, Any]] = []

    for value_col in value_candidates:
        for row_attrs in row_combos:
            for col_attrs in col_combos:
                if not disjoint(row_attrs, col_attrs):
                    continue

                est_cells = estimated_cartesian_cells(work_df, row_attrs, col_attrs)
                if est_cells > MAX_ESTIMATED_CARTESIAN_CELLS:
                    continue

                needed_cols = list(row_attrs) + list(col_attrs) + [value_col]
                sub = work_df[needed_cols].copy()

                for c in row_attrs:
                    sub[c] = sub[c].map(normalize_axis_value)
                for c in col_attrs:
                    sub[c] = sub[c].map(normalize_axis_value)

                sub = sub.dropna(subset=list(row_attrs) + list(col_attrs) + [value_col])
                if sub.empty:
                    continue

                try:
                    pivot = sub.pivot_table(
                        index=list(row_attrs),
                        columns=list(col_attrs),
                        values=value_col,
                        aggfunc="mean",
                        observed=False,
                    )
                except Exception:
                    continue

                if pivot.shape[0] < MIN_ROWS or pivot.shape[1] < MIN_COLS:
                    continue

                density = pivot_density(pivot)
                if density < MIN_DENSITY:
                    continue

                results.append(
                    {
                        "dataset": dataset_name,
                        "relation": relation_name,
                        "row_attributes": list(row_attrs),
                        "column_attributes": list(col_attrs),
                        "value_attribute": value_col,
                        "num_row_attributes": len(row_attrs),
                        "num_column_attributes": len(col_attrs),
                        "pivot_rows": int(pivot.shape[0]),
                        "pivot_columns": int(pivot.shape[1]),
                        "pivot_cells": int(pivot.shape[0] * pivot.shape[1]),
                        "non_nan_density": density,
                        "source_rows": int(df.shape[0]),
                        "source_columns": int(df.shape[1]),
                    }
                )

    deduped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in results:
        key = (
            r["dataset"],
            r["relation"],
            tuple(r["row_attributes"]),
            tuple(r["column_attributes"]),
            r["value_attribute"],
        )
        old = deduped.get(key)
        if old is None or score_result(r) > score_result(old):
            deduped[key] = r

    out = list(deduped.values())
    out.sort(key=score_result, reverse=True)

    if MAX_RESULTS_PER_RELATION is not None:
        out = out[:MAX_RESULTS_PER_RELATION]

    return out


def main() -> None:
    built_dbs: List[Tuple[str, Path]] = []

    for spec in DATASETS:
        try:
            db_path = ensure_dataset(spec)
            built_dbs.append((spec.dataset_name, db_path))
        except Exception as e:
            print(f"SKIP dataset {spec.dataset_name}: {e}")

    summary_rows: List[Dict[str, Any]] = []

    for dataset_name, db_path in built_dbs:
        print(f"Scanning dataset: {dataset_name}")
        con = sqlite3.connect(db_path)
        try:
            relations = get_relation_names(con)
            valid_for_dataset = 0

            for relation_name, relation_type in relations:
                try:
                    df = read_relation(con, relation_name)
                except Exception as e:
                    print(f"  SKIP relation {relation_name}: {e}")
                    continue

                results = find_all_valid_pivots(df, dataset_name, relation_name)

                if results:
                    valid_for_dataset += 1
                    csv_path = export_relation_csv(df, dataset_name, relation_name)
                    for result in results:
                        result["relation_type"] = relation_type
                        result["table_csv_path"] = csv_path
                        summary_rows.append(result)
                        print(
                            f"  PASS {relation_type} {relation_name}: "
                            f"rows={result['row_attributes']} "
                            f"cols={result['column_attributes']} "
                            f"value={result['value_attribute']} "
                            f"shape={result['pivot_rows']}x{result['pivot_columns']} "
                            f"levels=({result['num_row_attributes']},{result['num_column_attributes']}) "
                            f"density={result['non_nan_density']:.3f}"
                        )

            if valid_for_dataset == 0:
                print(f"  -> dataset {dataset_name} produced no valid extracted tables")
            else:
                print(f"  -> dataset {dataset_name} produced {valid_for_dataset} valid source relations")
        finally:
            con.close()

    if not summary_rows:
        print("No qualifying relations found.")
        return

    summary_df = pd.DataFrame(summary_rows)

    summary_df["row_attributes"] = summary_df["row_attributes"].apply(lambda x: "; ".join(x))
    summary_df["column_attributes"] = summary_df["column_attributes"].apply(lambda x: "; ".join(x))
    summary_df["total_header_levels"] = (
        summary_df["num_row_attributes"] + summary_df["num_column_attributes"]
    )
    summary_df["min_header_side_levels"] = summary_df[
        ["num_row_attributes", "num_column_attributes"]
    ].min(axis=1)

    summary_df = summary_df.sort_values(
        by=[
            "non_nan_density",
            "total_header_levels",
            "min_header_side_levels",
            "pivot_cells",
            "pivot_rows",
            "pivot_columns",
            "dataset",
            "relation",
            "value_attribute",
            "row_attributes",
            "column_attributes",
        ],
        ascending=[False, False, False, False, False, False, True, True, True, True, True],
    ).reset_index(drop=True)

    summary_csv = EXPORT_DIR / "qualifying_dense_pivots_verified_moredbs.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_per_relation = (
        summary_df.sort_values(
            by=[
                "non_nan_density",
                "total_header_levels",
                "min_header_side_levels",
                "pivot_cells",
                "pivot_rows",
                "pivot_columns",
            ],
            ascending=[False, False, False, False, False, False],
        )
        .groupby(["dataset", "relation"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_per_relation_csv = EXPORT_DIR / "best_qualifying_dense_pivots_per_relation_moredbs.csv"
    best_per_relation.to_csv(best_per_relation_csv, index=False)

    working_dbs = summary_df[["dataset"]].drop_duplicates()["dataset"].tolist()

    print()
    print("Saved summary to:", summary_csv)
    print("Saved best-per-relation summary to:", best_per_relation_csv)
    print("Databases that actually produced at least one valid extracted table:")
    for db in working_dbs:
        print(" -", db)
    print(f"Total qualifying pivots: {len(summary_df)}")
    print(
        "Total source relations with at least one valid pivot: "
        f"{summary_df[['dataset', 'relation']].drop_duplicates().shape[0]}"
    )
    print()
    print("Best valid pivot per relation:")
    print(best_per_relation.to_string(index=False))


if __name__ == "__main__":
    main()
