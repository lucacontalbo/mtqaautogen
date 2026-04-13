import re
import pandas as pd
import argparse

DEPENDENCY_PLACEHOLDER = "this value depends on the previous instruction"


def camel_to_words(name: str) -> str:
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)
    return " ".join(w.lower() for w in words)


def clean_value(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def extract_select_expr(sql: str) -> str:
    m = re.search(r"SELECT\s+(.*?)\s+FROM\b", sql, flags=re.I | re.S)
    if not m:
        raise ValueError(f"Could not extract SELECT expression from:\n{sql}")
    return m.group(1).strip()


def extract_where_clause(sql: str) -> str | None:
    m = re.search(r"\bWHERE\b\s+(.*?)(?:;|\bUNION\b|\)\s*$|$)", sql, flags=re.I | re.S)
    return m.group(1).strip() if m else None


def split_conditions(where_clause: str) -> list[str]:
    parts = []
    current = []
    in_single = False
    in_double = False
    i = 0
    n = len(where_clause)

    while i < n:
        ch = where_clause[i]

        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if where_clause[i:i+3].upper() == "AND":
                prev_ok = i == 0 or where_clause[i - 1].isspace()
                next_ok = i + 3 >= n or where_clause[i + 3].isspace()
                if prev_ok and next_ok:
                    part = "".join(current).strip()
                    if part:
                        parts.append(part)
                    current = []
                    i += 3
                    while i < n and where_clause[i].isspace():
                        i += 1
                    continue

        current.append(ch)
        i += 1

    part = "".join(current).strip()
    if part:
        parts.append(part)

    return parts


def parse_condition(cond: str) -> tuple[str, str, str] | None:
    m = re.match(
        r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(=|!=|<>|>=|<=|>|<)\s*(.+?)\s*$',
        cond,
        flags=re.S,
    )
    if not m:
        return None
    attr, op, value = m.groups()
    return attr.strip(), op.strip(), clean_value(value.strip())


def operator_to_text(op: str) -> str:
    return {
        "=": "is",
        "!=": "is not",
        "<>": "is not",
        ">": "is greater than",
        "<": "is less than",
        ">=": "is greater than or equal to",
        "<=": "is less than or equal to",
    }[op]


def condition_to_text(cond: str) -> str | None:
    parsed = parse_condition(cond)
    if not parsed:
        return None
    attr, op, value = parsed
    return f'{camel_to_words(attr)} {operator_to_text(op)} "{value}"'


def is_dependency_condition(cond: str) -> bool:
    parsed = parse_condition(cond)
    return bool(parsed and parsed[2].lower() == DEPENDENCY_PLACEHOLDER)


def parse_last_query(last_sql: str) -> tuple[str, str, list[str]]:
    select_expr = extract_select_expr(last_sql)

    agg_match = re.match(
        r"^(AVG|SUM|MIN|MAX|COUNT)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)(?:\s+AS\s+([A-Za-z_][A-Za-z0-9_]*))?$",
        select_expr,
        flags=re.I,
    )

    if agg_match:
        agg_func, inner_col, alias = agg_match.groups()
        question_type = agg_func.lower()
        value_name = camel_to_words(alias or inner_col)
    else:
        question_type = "exact"
        value_name = camel_to_words(select_expr.split()[-1])

    subquery_wheres = re.findall(
        r"SELECT\s+.*?\s+FROM\s+.*?\bWHERE\b\s+(.*?)(?=\bUNION\s+ALL\b|\)\s*;?\s*$)",
        last_sql,
        flags=re.I | re.S,
    )

    across_items = []
    for where_clause in subquery_wheres:
        parts = split_conditions(where_clause)
        texts = []
        for p in parts:
            if is_dependency_condition(p):
                continue
            t = condition_to_text(p)
            if t:
                texts.append(t)
        if texts:
            across_items.append(", ".join(["the "+t for t in texts[:-1]])+" and the "+texts[-1])

    if question_type == "avg":
        question_type = "average"
    if question_type == "sum":
        question_type = "summation"

    return question_type, value_name, across_items


def sql_queries_to_templated_question(sql_queries: list[str]) -> str:
    if not sql_queries:
        raise ValueError("sql_queries cannot be empty")

    constraints = []
    seen = set()

    for sql in sql_queries[:-1]:
        where_clause = extract_where_clause(sql)
        if not where_clause:
            continue

        for cond in split_conditions(where_clause):
            if is_dependency_condition(cond):
                continue
            text = condition_to_text(cond)
            if text and text not in seen:
                seen.add(text)
                constraints.append(text)

    question_type, value_name, across_items = parse_last_query(sql_queries[-1])

    lines = ["Considering the following constraints:"]
    for i, c in enumerate(constraints, start=1):
        lines.append(f"{i}) {c}")

    lines.append("")
    if across_items:
        across_text = "; ".join(f"({i}) {item}" for i, item in enumerate(across_items, start=1))
        if question_type == "summation":
            lines.append(f"what is the {question_type} of {value_name} where {across_text}?")
        else:
            lines.append(f"what is the {question_type} value of {value_name} where {across_text}?")
    else:
        lines.append(f"what is the {question_type} value of {value_name}?")

    return "\n".join(lines)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, required=True, help='path to QA dataset to process')

    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":
    sql_queries = [
        'SELECT OperationalLossEventCount FROM CounterpartyStressTable WHERE FXExposureBucket = "major currency concentrated long bucket" AND BaselRiskWeightedAssetClass = "SFTs and repos netting bucket" AND PrepaymentAssumptionCurve = "accelerated prepay stress scenario";',
        'SELECT CovenantBreachFlag FROM CounterpartyStressTable WHERE InstrumentISIN = "US02079K3059" AND PortfolioVintageYear = "2010" AND CounterpartyName = "Deutsche Bank AG" AND OperationalLossEventCount = "this value depends on the previous instruction";',
        'SELECT CreditRatingAgencyConsensus FROM CounterpartyStressTable WHERE YieldCurveSteepnessBps = "1200" AND CovenantBreachFlag = "this value depends on the previous instruction" AND VolatilitySkewRatio = "0" AND LegalProvisionReserveBucket = "settlement provision awaiting mediation outcome";',
        'SELECT CollateralEligibilityTier FROM CounterpartyStressTable WHERE TraderDeskCode = "EMEA rates — London execution desk intrinsic" AND CreditRatingAgencyConsensus = "this value depends on the previous instruction" AND SettlementVelocityIndex = "1" AND PriceDiscoveryDepth = "intermittent discovery driven by block trades";',
        'SELECT AVG(StressLossEstimate) AS StressLossEstimate FROM (SELECT StressLossEstimate AS StressLossEstimate FROM CounterpartyStressTable WHERE CollateralEligibilityTier = "this value depends on the previous instruction" AND MarketRegimeLabel = "low-volatility complacent state" AND LiquidityCoverageAdj = "0" AND OptionGammaExposure = "500000" UNION ALL SELECT StressLossEstimate AS StressLossEstimate FROM CounterpartyStressTable WHERE CollateralEligibilityTier = "this value depends on the previous instruction" AND MarketRegimeLabel = "idiosyncratic credit shock window" AND LiquidityCoverageAdj = "5000" AND OptionGammaExposure = "500000");',
    ]

    args = get_args()
    df = pd.read_csv(args["input_path"])
    df_copy = df.copy()
    for i, row in df.iterrows():
        question = sql_queries_to_templated_question(eval(row["SQL Query"]))
        df_copy.at[i, "Question"] = question

    new_path = args["input_path"].replace(".csv", "_templated.csv")
    df_copy.to_csv(new_path, index=False)