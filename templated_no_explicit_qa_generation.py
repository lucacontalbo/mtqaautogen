import argparse
import ast
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

warnings.filterwarnings("ignore")


# -----------------------------
# SQL parsing
# -----------------------------

_SQL_IDENT = r'(?:`[^`]+`|"[^"]+"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)'


@dataclass
class SQLCondition:
    field: str
    op: str
    value: Any


@dataclass
class SQLQuery:
    select_field: str
    table_name: Optional[str]
    where: List[SQLCondition]


def _strip_identifier(x: str) -> str:
    x = x.strip()
    if (x.startswith("`") and x.endswith("`")) or \
       (x.startswith('"') and x.endswith('"')) or \
       (x.startswith("[") and x.endswith("]")):
        return x[1:-1]
    return x


def _parse_sql_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.upper() == "NULL":
        return None
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return _strip_identifier(raw)


def _split_where_clause(where_clause: str) -> List[str]:
    parts = []
    cur = []
    in_single = False
    in_double = False
    depth = 0

    i = 0
    while i < len(where_clause):
        ch = where_clause[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            cur.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            cur.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(depth - 1, 0)

            if depth == 0 and where_clause[i:i + 3].upper() == "AND":
                prev_ok = (i == 0 or where_clause[i - 1].isspace())
                next_ok = (i + 3 == len(where_clause) or where_clause[i + 3].isspace())
                if prev_ok and next_ok:
                    parts.append("".join(cur).strip())
                    cur = []
                    i += 3
                    continue

        cur.append(ch)
        i += 1

    if cur:
        parts.append("".join(cur).strip())

    return [p for p in parts if p]


def parse_simple_sql(sql: str) -> SQLQuery:
    """
    Supports:
      SELECT <field> FROM <table> WHERE <cond1> AND <cond2> ...
    with operators: =, !=, <>, >, >=, <, <=, IN
    """
    sql = " ".join(sql.strip().rstrip(";").split())

    m = re.match(
        rf"(?is)^SELECT\s+(?P<select>{_SQL_IDENT})\s+FROM\s+(?P<table>{_SQL_IDENT})(?:\s+WHERE\s+(?P<where>.*))?$",
        sql
    )
    if not m:
        raise ValueError(
            f"Unsupported SQL format. Expected: SELECT <field> FROM <table> [WHERE ...], got: {sql}"
        )

    select_field = _strip_identifier(m.group("select"))
    table_name = _strip_identifier(m.group("table"))
    where_raw = m.group("where")

    conditions: List[SQLCondition] = []
    if where_raw:
        for piece in _split_where_clause(where_raw):
            m_in = re.match(rf"(?is)^\s*({_SQL_IDENT})\s+IN\s*\((.*)\)\s*$", piece)
            if m_in:
                field = _strip_identifier(m_in.group(1))
                values_raw = m_in.group(2)
                values = []
                cur = []
                in_single = False
                in_double = False
                for ch in values_raw:
                    if ch == "'" and not in_double:
                        in_single = not in_single
                    elif ch == '"' and not in_single:
                        in_double = not in_double
                    if ch == "," and not in_single and not in_double:
                        values.append(_parse_sql_value("".join(cur).strip()))
                        cur = []
                    else:
                        cur.append(ch)
                if cur:
                    values.append(_parse_sql_value("".join(cur).strip()))
                conditions.append(SQLCondition(field, "IN", values))
                continue

            m_cmp = re.match(
                rf"(?is)^\s*({_SQL_IDENT})\s*(=|!=|<>|>=|<=|>|<)\s*(.+?)\s*$",
                piece
            )
            if not m_cmp:
                raise ValueError(f"Unsupported WHERE condition: {piece}")

            field = _strip_identifier(m_cmp.group(1))
            op = m_cmp.group(2)
            value = _parse_sql_value(m_cmp.group(3))
            conditions.append(SQLCondition(field, op, value))

    return SQLQuery(select_field=select_field, table_name=table_name, where=conditions)


# -----------------------------
# Fuzzy normalization + scoring
# -----------------------------

def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.casefold()
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9.\- %]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_sort_string(s: str) -> str:
    toks = s.split()
    toks.sort()
    return " ".join(toks)


def _string_similarity(a: Any, b: Any) -> float:
    import difflib

    a = _normalize_text(a)
    b = _normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    r1 = difflib.SequenceMatcher(None, a, b).ratio()
    r2 = difflib.SequenceMatcher(None, _token_sort_string(a), _token_sort_string(b)).ratio()

    sub = 0.0
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        sub = shorter / max(longer, 1)

    return max(r1, r2, sub)


def _best_match_score(query: Any, candidates: Sequence[Any]) -> float:
    if not candidates:
        return 0.0
    return max(_string_similarity(query, c) for c in candidates)


# -----------------------------
# Table path helpers
# -----------------------------

def _index_paths(index: pd.Index) -> List[Tuple[Any, ...]]:
    if isinstance(index, pd.MultiIndex):
        return [tuple(x) for x in index.tolist()]
    return [(x,) for x in index.tolist()]


def _column_paths(columns: pd.Index) -> List[Tuple[Any, ...]]:
    if isinstance(columns, pd.MultiIndex):
        return [tuple(x) for x in columns.tolist()]
    return [(x,) for x in columns.tolist()]


def _flatten_path(path: Tuple[Any, ...]) -> List[str]:
    out = []
    for x in path:
        s = _normalize_text(x)
        if s:
            out.append(s)
    return out


# -----------------------------
# Constraint / target scoring
# -----------------------------

def _constraint_score_on_path(cond: SQLCondition, path: Tuple[Any, ...]) -> float:
    labels = _flatten_path(path)
    if not labels:
        return 0.0

    field_score = _best_match_score(cond.field, labels)

    if cond.op == "IN":
        value_score = max((_best_match_score(v, labels) for v in cond.value), default=0.0)
    else:
        value_score = _best_match_score(cond.value, labels)

    score = 0.35 * field_score + 0.65 * value_score

    if field_score >= 0.70 and value_score >= 0.70:
        score += 0.15

    return min(score, 1.0)


def _select_score_on_path(select_field: str, path: Tuple[Any, ...]) -> float:
    labels = _flatten_path(path)
    if not labels:
        return 0.0
    return _best_match_score(select_field, labels)


# -----------------------------
# Main solver
# -----------------------------

@dataclass
class CellMatchResult:
    row_iloc: int
    col_iloc: int
    row_index_label: Tuple[Any, ...]
    col_index_label: Tuple[Any, ...]
    answer: Any
    score: float
    debug: Dict[str, Any]


def find_cell_from_sql(
    df: pd.DataFrame,
    sql: str,
    *,
    select_axis_bias: str = "columns",
    min_score: float = 0.35,
) -> CellMatchResult:
    q = parse_simple_sql(sql)

    row_paths = _index_paths(df.index)
    col_paths = _column_paths(df.columns)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Matching dataframe is empty; could not search for answer cell.")

    best = None

    for ri, rpath in enumerate(row_paths):
        row_select = _select_score_on_path(q.select_field, rpath)

        for ci, cpath in enumerate(col_paths):
            col_select = _select_score_on_path(q.select_field, cpath)

            cond_debug = []
            cond_total = 0.0
            for cond in q.where:
                rs = _constraint_score_on_path(cond, rpath)
                cs = _constraint_score_on_path(cond, cpath)
                cond_total += max(rs, cs)
                cond_debug.append({
                    "field": cond.field,
                    "op": cond.op,
                    "value": cond.value,
                    "row_score": rs,
                    "col_score": cs,
                    "used": "row" if rs >= cs else "col",
                })

            select_score = max(row_select, col_select)

            bias = 0.0
            if select_axis_bias == "columns" and col_select >= row_select:
                bias = 0.05
            elif select_axis_bias == "rows" and row_select >= col_select:
                bias = 0.05

            total_score = cond_total + 1.25 * select_score + bias

            candidate = CellMatchResult(
                row_iloc=ri,
                col_iloc=ci,
                row_index_label=rpath,
                col_index_label=cpath,
                answer=df.iat[ri, ci],
                score=total_score,
                debug={
                    "select_field": q.select_field,
                    "row_select_score": row_select,
                    "col_select_score": col_select,
                    "conditions": cond_debug,
                },
            )

            if best is None or candidate.score > best.score:
                best = candidate

    if best is None or best.score < min_score:
        raise ValueError(
            f"Could not confidently identify a cell. Best score={None if best is None else best.score:.3f}"
        )

    return best


# -----------------------------
# HTML table mapping
# -----------------------------

@dataclass
class HtmlCellRef:
    tag: Tag
    source_row: int
    source_col_in_row: int
    is_header: bool


@dataclass
class HtmlTableMap:
    soup: BeautifulSoup
    table_tag: Tag
    grid_text: List[List[str]]
    grid_refs: List[List[Optional[HtmlCellRef]]]
    nrows: int
    ncols: int


@dataclass
class MatchingTable:
    html: str
    table_map: HtmlTableMap
    match_df: pd.DataFrame
    header_rows: int
    header_cols: int


def _get_cell_text(tag: Tag) -> str:
    return tag.get_text(" ", strip=True)


def build_html_table_map(html: str) -> HtmlTableMap:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> found in HTML.")

    trs = table.find_all("tr")
    grid_text: List[List[str]] = []
    grid_refs: List[List[Optional[HtmlCellRef]]] = []

    # col -> (remaining_rows, ref, text)
    active_rowspans: Dict[int, Tuple[int, HtmlCellRef, str]] = {}
    max_cols = 0

    for r, tr in enumerate(trs):
        row_text: List[str] = []
        row_refs: List[Optional[HtmlCellRef]] = []
        c = 0

        def fill_carried():
            nonlocal c
            while c in active_rowspans:
                remaining, ref, txt = active_rowspans[c]
                row_text.append(txt)
                row_refs.append(ref)
                if remaining > 1:
                    active_rowspans[c] = (remaining - 1, ref, txt)
                else:
                    del active_rowspans[c]
                c += 1

        fill_carried()

        source_cells = tr.find_all(["th", "td"], recursive=False)
        for source_col, cell in enumerate(source_cells):
            fill_carried()

            txt = _get_cell_text(cell)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            ref = HtmlCellRef(
                tag=cell,
                source_row=r,
                source_col_in_row=source_col,
                is_header=(cell.name.lower() == "th"),
            )

            for offset in range(colspan):
                row_text.append(txt)
                row_refs.append(ref)
                if rowspan > 1:
                    active_rowspans[c + offset] = (rowspan - 1, ref, txt)
            c += colspan

            fill_carried()

        fill_carried()

        max_cols = max(max_cols, len(row_text))
        grid_text.append(row_text)
        grid_refs.append(row_refs)

    for i in range(len(grid_text)):
        while len(grid_text[i]) < max_cols:
            grid_text[i].append("")
            grid_refs[i].append(None)

    return HtmlTableMap(
        soup=soup,
        table_tag=table,
        grid_text=grid_text,
        grid_refs=grid_refs,
        nrows=len(grid_text),
        ncols=max_cols,
    )


def infer_header_depths(table_map: HtmlTableMap) -> Tuple[int, int]:
    """
    Header rows:
      consecutive top rows where every occupied cell is a <th>.

    Header columns:
      after removing header rows, consecutive left columns where every occupied cell
      in the remaining rows is a <th>.
    """
    header_rows = 0
    for r in range(table_map.nrows):
        refs = [ref for ref in table_map.grid_refs[r] if ref is not None]
        if refs and all(ref.is_header for ref in refs):
            header_rows += 1
        else:
            break

    header_cols = 0
    body_row_start = header_rows

    for c in range(table_map.ncols):
        refs = []
        for r in range(body_row_start, table_map.nrows):
            ref = table_map.grid_refs[r][c]
            if ref is not None:
                refs.append(ref)
        if refs and all(ref.is_header for ref in refs):
            header_cols += 1
        else:
            break

    return header_rows, header_cols


def build_matching_table(html: str) -> MatchingTable:
    table_map = build_html_table_map(html)
    header_rows, header_cols = infer_header_depths(table_map)

    raw_df = pd.DataFrame(table_map.grid_text)

    data = raw_df.iloc[header_rows:, header_cols:].copy()

    if header_rows > 0 and data.shape[1] > 0:
        col_arrays = [raw_df.iloc[i, header_cols:].tolist() for i in range(header_rows)]
        data.columns = pd.MultiIndex.from_arrays(col_arrays) if header_rows > 1 else pd.Index(col_arrays[0])

    if header_cols > 0 and data.shape[0] > 0:
        row_arrays = [raw_df.iloc[header_rows:, j].tolist() for j in range(header_cols)]
        data.index = pd.MultiIndex.from_arrays(row_arrays) if header_cols > 1 else pd.Index(row_arrays[0])

    return MatchingTable(
        html=html,
        table_map=table_map,
        match_df=data,
        header_rows=header_rows,
        header_cols=header_cols,
    )


def df_pos_to_grid_pos(tbl: MatchingTable, row_iloc: int, col_iloc: int) -> Tuple[int, int]:
    return row_iloc + tbl.header_rows, col_iloc + tbl.header_cols


def find_cell_from_html_table(html: str, sql: str, *, min_score: float = 0.35) -> CellMatchResult:
    tbl = build_matching_table(html)
    return find_cell_from_sql(tbl.match_df, sql, min_score=min_score)


# -----------------------------
# Previous value location + masking
# -----------------------------

@dataclass
class PreviousValueLocation:
    kind: str  # "body", "row_header", "col_header", "header", "not_found"
    score: float
    body_matches: List[Tuple[int, int, float]]
    row_matches: List[Tuple[int, float]]
    col_matches: List[Tuple[int, float]]

def extract_inside_outside(s: str):
    if s == "":
        return [""]
    inside = [x.strip() for x in re.findall(r'\((.*?)\)', s)]
    outside = re.sub(r'\(.*?\)', ' ', s)
    outside = ' '.join(outside.split())
    inside.append(outside)
    return inside

def _find_previous_value_locations(
    df: pd.DataFrame,
    previous_value: Any,
    *,
    body_threshold: float = 0.88,
    header_threshold: float = 0.88,
) -> PreviousValueLocation:
    row_paths = _index_paths(df.index)
    col_paths = _column_paths(df.columns)

    body_matches: List[Tuple[int, int, float]] = []
    row_matches: List[Tuple[int, float]] = []
    col_matches: List[Tuple[int, float]] = []

    best_body = 0.0
    best_row = 0.0
    best_col = 0.0

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            texts = extract_inside_outside(df.iloc[i, j])

            for text in texts:
                s = _string_similarity(text, previous_value)
                if s >= body_threshold:
                    body_matches.append((i, j, s))
                if s > best_body:
                    best_body = s

    for i, rpath in enumerate(row_paths):
        labels = _flatten_path(rpath)
        s = max((_string_similarity(lbl, previous_value) for lbl in labels), default=0.0)
        if s >= header_threshold:
            row_matches.append((i, s))
        if s > best_row:
            best_row = s

    for j, cpath in enumerate(col_paths):
        labels = _flatten_path(cpath)
        s = max((_string_similarity(lbl, previous_value) for lbl in labels), default=0.0)
        if s >= header_threshold:
            col_matches.append((j, s))
        if s > best_col:
            best_col = s

    best_header = max(best_row, best_col)

    if best_body == 0.0 and best_header == 0.0:
        kind = "not_found"
        score = 0.0
    elif best_body >= best_header:
        kind = "body"
        score = best_body
    else:
        if best_row > 0.0 and best_col > 0.0:
            kind = "header"
        elif best_row > 0.0:
            kind = "row_header"
        else:
            kind = "col_header"
        score = best_header

    return PreviousValueLocation(
        kind=kind,
        score=score,
        body_matches=body_matches,
        row_matches=row_matches,
        col_matches=col_matches,
    )


def _set_tag_text(tag: Tag, text: str) -> None:
    tag.clear()
    tag.append(text)


def mask_second_table_from_previous_result(
    html_second: str,
    sql_second: str,
    previous_result: Any,
    *,
    body_threshold: float = 0.88,
    header_threshold: float = 0.88,
    min_query_score: float = 0.35,
) -> Tuple[str, Dict[str, Any]]:
    tbl = build_matching_table(html_second)
    df_second = tbl.match_df

    target = find_cell_from_sql(df_second, sql_second, min_score=min_query_score)

    prev_loc = _find_previous_value_locations(
        df_second,
        previous_result,
        body_threshold=body_threshold,
        header_threshold=header_threshold,
    )

    target_grid = df_pos_to_grid_pos(tbl, target.row_iloc, target.col_iloc)
    changed_tag_ids = set()
    masked_positions = []

    def mask_grid_position(grid_r: int, grid_c: int, reason: str, score: Optional[float] = None):
        if not (0 <= grid_r < tbl.table_map.nrows and 0 <= grid_c < tbl.table_map.ncols):
            return
        if (grid_r, grid_c) == target_grid:
            return

        ref = tbl.table_map.grid_refs[grid_r][grid_c]
        if ref is None:
            return

        tag_id = id(ref.tag)
        if tag_id in changed_tag_ids:
            return

        _set_tag_text(ref.tag, "n.a.")
        changed_tag_ids.add(tag_id)

        entry = {
            "grid_row": grid_r,
            "grid_col": grid_c,
            "reason": reason,
        }
        if score is not None:
            entry["score"] = score
        masked_positions.append(entry)

    if prev_loc.kind == "body":
        for i, j, s in prev_loc.body_matches:
            grid_r, grid_c = df_pos_to_grid_pos(tbl, i, j)
            mask_grid_position(grid_r, grid_c, "body_match", s)

    elif prev_loc.kind in {"row_header", "col_header", "header"}:
        rows_to_mask = {i for i, _ in prev_loc.row_matches}
        cols_to_mask = {j for j, _ in prev_loc.col_matches}

        for i in rows_to_mask:
            grid_r = i + tbl.header_rows
            for grid_c in range(tbl.table_map.ncols):
                mask_grid_position(grid_r, grid_c, "matched_row_header")

        for j in cols_to_mask:
            grid_c = j + tbl.header_cols
            for grid_r in range(tbl.table_map.nrows):
                mask_grid_position(grid_r, grid_c, "matched_col_header")

    info = {
        "target_cell": {
            "row_iloc": target.row_iloc,
            "col_iloc": target.col_iloc,
            "row_index_label": target.row_index_label,
            "col_index_label": target.col_index_label,
            "answer": target.answer,
            "score": target.score,
        },
        "second_query_result": target.answer,
        "second_query_row_iloc": target.row_iloc,
        "second_query_col_iloc": target.col_iloc,
        "previous_result": previous_result,
        "previous_result_location": {
            "kind": prev_loc.kind,
            "score": prev_loc.score,
            "body_matches": prev_loc.body_matches,
            "row_matches": prev_loc.row_matches,
            "col_matches": prev_loc.col_matches,
        },
        "masked_positions": masked_positions,
    }

    return str(tbl.table_map.table_tag), info


# -----------------------------
# HTML extraction / replacement
# -----------------------------

def extract_html_tables_with_spans(html: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"<table\b.*?</table>", html, flags=re.IGNORECASE | re.DOTALL)]


def replace_tables_in_html(original_html: str, new_tables: List[str]) -> str:
    spans = extract_html_tables_with_spans(original_html)
    if len(spans) != len(new_tables):
        raise ValueError(
            f"Mismatch when replacing tables: found {len(spans)} original tables, got {len(new_tables)} replacements."
        )

    pieces = []
    last = 0
    for (start, end, _old), new_table in zip(spans, new_tables):
        pieces.append(original_html[last:start])
        pieces.append(new_table)
        last = end
    pieces.append(original_html[last:])
    return "".join(pieces)


def extract_html_tables(html: str) -> List[str]:
    return [x[2] for x in extract_html_tables_with_spans(html)]


# -----------------------------
# Question generation
# -----------------------------

def get_new_question(sql_queries: List[str]) -> Optional[str]:
    """
    Placeholder.
    Replace this with your own question-generation logic.
    Returning None leaves the Question column unchanged if it already exists.
    """
    return None


# -----------------------------
# CSV processing
# -----------------------------

def _load_sql_queries(cell_value: Any) -> List[str]:
    if isinstance(cell_value, list):
        return cell_value
    if not isinstance(cell_value, str):
        raise ValueError(f"Unsupported SQL Query cell type: {type(cell_value)}")
    parsed = ast.literal_eval(cell_value)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("The 'SQL Query' column must contain a Python list of SQL strings.")
    return parsed


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    for i, row in df.iterrows():
        sql_queries = _load_sql_queries(row["SQL Query"])
        original_table_html = row["Table"]

        if not isinstance(original_table_html, str):
            raise ValueError(f"Row {i}: 'Table' must be a string containing HTML.")

        templated_question = get_new_question(sql_queries)
        if templated_question is not None:
            df_copy.at[i, "Question"] = templated_question

        tables = extract_html_tables(original_table_html)
        if len(tables) != len(sql_queries):
            raise ValueError(
                f"Row {i}: number of tables does not match number of SQL queries "
                f"({len(tables)} tables vs {len(sql_queries)} SQL queries)."
            )

        if not tables:
            df_copy.at[i, "Table"] = original_table_html
            continue

        updated_tables = list(tables)
        info = None

        for j in range(len(sql_queries) - 2):
            if j == 0:
                first_result = find_cell_from_html_table(updated_tables[j], sql_queries[j])
                previous_answer = first_result.answer
            else:
                previous_answer = info["second_query_result"]

            new_html, info = mask_second_table_from_previous_result(
                html_second=updated_tables[j + 1],
                sql_second=sql_queries[j + 1],
                previous_result=previous_answer,
            )
            updated_tables[j + 1] = new_html

        df_copy.at[i, "Table"] = replace_tables_in_html(original_table_html, updated_tables)

    return df_copy


# -----------------------------
# CLI
# -----------------------------

def derive_output_csv_path(input_csv: str | Path) -> Path:
    input_path = Path(input_csv)
    if input_path.suffix.lower() != ".csv":
        raise ValueError("Input file must have a .csv extension.")
    return input_path.with_name(f"{input_path.stem}_templated_no_explicit.csv")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process chained SQL queries over HTML tables, preserve the original table HTML, "
            "and mask repeated intermediate values with 'n.a.' while keeping all other table structure unchanged."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help='Path to the input CSV file. Must contain "SQL Query" and "Table" columns.',
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_csv = Path(args.input_path)
    output_csv = derive_output_csv_path(input_csv)

    df = pd.read_csv(input_csv)

    required = {"SQL Query", "Table"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = process_dataframe(df)
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()