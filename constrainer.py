from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Tuple, Union
from utils import replace_many, is_float
import re
import pandas as pd
import numpy as np
import operator
from z3solver import Z3Solver
from itertools import chain
import ast


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)      # auto
pd.set_option("display.max_colwidth", None)

# -----------------------------
# AST node definitions
# -----------------------------
@dataclass(frozen=True)
class Compare:
    col: str
    op: str   # "==" or "!="
    lit: Any  # literal value


@dataclass(frozen=True)
class BoolOp:
    op: str   # "and" or "or"
    left: "Expr"
    right: "Expr"

op_bool_map = {
    "==": operator.eq,
    "!=": operator.ne,
    "<":  operator.lt,
    "<=": operator.le,
    ">":  operator.gt,
    ">=": operator.ge,
}

op_math_map = {
    "+":  operator.add,
    "-":  operator.sub,
    "*":  operator.mul,
    "/":  operator.truediv,
    "//": operator.floordiv,
    "%":  operator.mod,
    "**": operator.pow,
}

Expr = Union[Compare, BoolOp]


# -----------------------------
# Tokenizer
# -----------------------------
_TOKEN_RE = re.compile(
    r"""
    \s*(
        ==|!=|<=|>=|<|>|\+|-|\*|/|//|%|\*\*|\(|\)             # operators/parens
        |\band\b|\bor\b                   # boolean keywords
        |"(?:\\.|[^"])*"                  # double-quoted string
        |'(?:\\.|[^'])*'                  # single-quoted string
        |[A-Za-z_][A-Za-z0-9_]*           # identifiers
        |-?\d+(?:\.\d+)?                  # numbers (int/float)
    )\s*
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _tokenize(s: str) -> List[str]:
    tokens = [m.group(1) for m in _TOKEN_RE.finditer(s)]
    joined = "".join(re.sub(r"\s+", "", t) for t in tokens)
    if re.sub(r"\s+", "", s) != joined:
        # Some character wasn't tokenized (e.g. unsupported operator)
        raise ValueError(f"Could not fully tokenize expression: {s!r}")

    # unifying expressions like "-1000", which would parsed into the ["-", "1000"] tokens, into a unique token
    tokens_unified, add_last = [tokens[0]], True
    for i in range(1,len(tokens)-1):
        if tokens[i] in ["-", "+"]:
            if is_float(tokens[i+1]) and not is_float(tokens[i-1]):
                tokens_unified.append(tokens[i]+tokens[i+1])
                i+=1
                add_last = False
            else:
                tokens_unified.append(tokens[i])
                add_last = True
        else:
            tokens_unified.append(tokens[i])
            add_last = True

    if add_last:
        tokens_unified.append(tokens[-1])
    # normalize and/or
    tokens = [t.lower() if t.lower() in ("and", "or") else t for t in tokens_unified]
    return tokens


# -----------------------------
# Recursive descent parser
# Grammar:
#   expr    := or_expr
#   or_expr := and_expr ("or" and_expr)*
#   and_expr:= atom ("and" atom)*
#   atom    := "(" expr ")" | comparison
#   comparison := IDENT ( "==" | "!=" ) LITERAL
# LITERAL may be identifier-like (e.g., COGS) or quoted string or number
# -----------------------------
class _Parser:
    def __init__(self, tokens: List[str]) -> None:
        self.toks = tokens
        self.i = 0

    def _peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, expected: str | None = None) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Expr:
        node = self._parse_or()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token: {self._peek()!r}")
        return node

    def _parse_or(self) -> Expr:
        node = self._parse_and()
        while self._peek() == "or":
            self._eat("or")
            rhs = self._parse_and()
            node = BoolOp("or", node, rhs)
        return node

    def _parse_and(self) -> Expr:
        node = self._parse_atom()
        while self._peek() == "and":
            self._eat("and")
            rhs = self._parse_atom()
            node = BoolOp("and", node, rhs)
        return node

    def _parse_atom(self) -> Expr:
        if self._peek() == "(":
            self._eat("(")
            node = self._parse_or()
            self._eat(")")
            return node
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        col = self._eat()
        op = self._eat()
        if op not in op_bool_map:
            raise ValueError(f"Unsupported operator {op!r} (only {list(op_bool_map.keys())})")
        lit_tok = self._eat()
        lit = _parse_literal(lit_tok)
        return Compare(col=col, op=op, lit=lit)

def parse_expr(s: str) -> Expr:
    return _Parser(_tokenize(s)).parse()

def extract_columns(expr: Expr) -> list[str]:
    """
    Return all column names referenced in an expression AST produced by parse_expr().
    """
    cols: list[str] = []

    def walk(node: Expr) -> None:
        if isinstance(node, Compare):
            cols.append(node.col)
            return
        if isinstance(node, BoolOp):
            walk(node.left)
            walk(node.right)
            return
        raise TypeError(f"Unknown Expr node type: {type(node)!r}")

    walk(expr)

    # removing duplicates while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _fix_mojibake(s: str) -> str:
    # Repair common UTF-8→cp1252/latin1 mojibake (â\x80\x94, Ã©, etc.)
    if "â" in s or "Ã" in s:
        try:
            return s.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    return s

def _parse_literal(tok: str) -> Any:
    # quoted string
    if (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
        q = tok[0]  # quote char
        inner = tok[1:-1]
        inner = bytes(inner, "utf-8").decode("unicode_escape")
        inner = _fix_mojibake(inner)
        return f"{q}{inner}{q}"
    # number
    if re.fullmatch(r"[+-]?\d+", tok):
        return int(tok)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", tok):
        return float(tok)
    # bare identifier
    return tok


# -----------------------------
# Evaluation on a DataFrame -> boolean mask
# -----------------------------
def eval_expr(expr: Expr, df: pd.DataFrame) -> pd.Series:
    if isinstance(expr, Compare):
        if expr.col not in df.columns:
            raise KeyError(f"Column {expr.col!r} not found in DataFrame")
        s = df[expr.col]
        if expr.op in op_bool_map:
            if isinstance(expr.lit, str) and ((expr.lit.startswith('"') and expr.lit.endswith('"')) or (expr.lit.startswith("'") and expr.lit.endswith("'"))):
                val = expr.lit[1:-1]
            else:
                val = expr.lit
            return op_bool_map[expr.op](s, val) #expr.lit)
        """if expr.op == "==":
            return s == expr.lit
        if expr.op == "!=":
            return s != expr.lit"""

        raise ValueError(f"Unknown op {expr.op!r}")
    else:
        left = eval_expr(expr.left, df)
        right = eval_expr(expr.right, df)
        if expr.op == "and":
            return left & right
        if expr.op == "or":
            return left | right
        raise ValueError(f"Unknown bool op {expr.op!r}")


def expr_to_str(node: Expr) -> str:
    if isinstance(node, Compare):
        lit = node.lit
        # quote strings that contain spaces/special chars if you want
        if isinstance(lit, str) and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lit):
            lit_s = repr(lit)
        else:
            lit_s = str(lit)
        return f"{node.col} {node.op} {lit_s}"

    if isinstance(node, BoolOp):
        return f"({expr_to_str(node.left)} {node.op} {expr_to_str(node.right)})"

    raise TypeError(f"Unknown Expr node type: {type(node)!r}")

def enforce_expr(expr: Expr, cond: Expr, df_constraints: pd.DataFrame, rows: pd.Series) -> bool:
    """
    Mutates df in place for selected rows to make `expr` evaluate True.
    """
    if rows is None or rows.sum() == 0:
        return True

    if isinstance(expr, Compare):
        col = expr.col
        if col not in df_constraints.columns:
            return True # we can't apply any constraints (e.g. due to table views reducing the number of attributes)

        cond_cols = extract_columns(cond)
        #if col not in df_constraints.columns:
        #    raise KeyError(f"Column {col!r} not found in DataFrame")
        if not all([cond_col in df_constraints.columns for cond_col in cond_cols]):
            return True # we can't apply any constraints (e.g. due to table views reducing the number of attributes)
            #raise KeyError(f"Some columns in condition ({cond_cols}) are not inside the df columns {df_constraints.columns}")

        if expr.op in op_bool_map:
            for r in df_constraints.index[rows]:
                value = f"x_{col}_{r} {expr.op} {expr.lit}"
                cond_cols_new = [f"x_{col_name}_{r}" for col_name in cond_cols]
                cond_str = expr_to_str(cond)
                cond_str = replace_many(cond_str, {k:v for k,v in zip(cond_cols, cond_cols_new)})

                df_constraints.at[r, col].append((cond_str, value))
            return True # ok

        #raise ValueError(f"Unknown op {expr.op!r}")

    #raise ValueError(f"Unknown bool op {expr.op!r}")
    return False

def enforce_expr_inter(expr: Expr, df_constraints: pd.DataFrame, link: List[int], rule: str, value_col: str) -> None:
    if link is None:
        return

    if isinstance(expr, Compare):
        if value_col not in df_constraints.columns:
            raise KeyError(f"Column {value_col!r} not found in DataFrame")

        if expr.op in op_bool_map:
            for i in range(len(link)):
                df_constraints.at[link[i], value_col].append(rule)
            return

        raise ValueError(f"Unknown op {expr.op!r}")

    raise ValueError(f"Unknown bool op {expr.op!r}")

# -----------------------------
# Main function
# -----------------------------


"""def mapper_inter_rules(df: pd.DataFrame, cond_masks: List[List[pd.Series]], cond_cols: List[List[str]], value_col: str, id_col: str):
    same_columns = True
    if len(cond_cols) == 0:
        return [], []

    for i in range(len(cond_cols[0])):
        col = cond_cols[0][i]
        for j in range(1, len(cond_cols)):
            if col != cond_cols[j][i]:
                same_columns = False
                break
        if not same_columns:
            break
    if not same_columns:
        return None, None

    df_tmp = df.copy(deep=True).drop(cond_cols[0] + [value_col, id_col], axis="columns")
    links, to_remove = [], set()
    for r in df_tmp.index[cond_masks[0]]:
        link = [r]
        added = False
        for cond_mask in cond_masks[1:]:
            added = False
            for r2 in df_tmp.index[cond_mask]:
                equal = all(df_tmp.loc[r] == df_tmp.loc[r2])
                if equal and not added:
                    link.append(r2)
                    added = True
                elif equal and added:
                    to_remove.add(r2)
            if not added:
                break

        if added:
            links.append(link)

    return links, to_remove"""

def mapper_inter_rules(df: pd.DataFrame, cond_masks: List[List[pd.Series]], cond_cols: List[List[str]], value_col: str, id_col: str):
    same_columns = True
    if len(cond_cols) == 0:
        return [], []

    for i in range(len(cond_cols[0])):
        col = cond_cols[0][i]
        for j in range(1, len(cond_cols)):
            if col != cond_cols[j][i]:
                same_columns = False
                break
        if not same_columns:
            break
    if not same_columns:
        return None, None

    df_tmp = df.copy(deep=True).drop(cond_cols[0] + [value_col], axis="columns") #[value_col, id_col], axis="columns")
    if len(df_tmp.columns) == 0:
        row_keys = pd.Series([()] * len(df_tmp), index=df_tmp.index)
    else:
        row_keys = pd.Series(list(map(tuple, df_tmp.to_numpy())), index=df_tmp.index)

    links, to_remove = [], set()
    keyed_masks = []
    for cond_mask in cond_masks:
        bucket = {}
        for r in df_tmp.index[cond_mask]:
            k = row_keys.at[r]
            bucket.setdefault(k, []).append(r)
        keyed_masks.append(bucket)

    for r in df_tmp.index[cond_masks[0]]:
        link = [r]
        added = False
        key = row_keys.at[r]
        for bucket in keyed_masks[1:]:
            added = False
            matches = bucket.get(key, [])
            if len(matches) > 0:
                link.append(matches[0])
                added = True
                for r2 in matches[1:]:
                    to_remove.add(r2)
            if not added:
                break

        if added:
            links.append(link)

    return links, to_remove

"""def get_inter_bounds(
    df: pd.DataFrame,
    selections: List[List[str]],
    rules: List[str],
    value_col: str,
    id_col: str,
    *,
    inplace: bool = False,
) -> pd.DataFrame:

    pat = re.compile(r"\(.*?\)\.\S+")

    def replace_atoms(expr, repl):
        it = iter(repl)
        return pat.sub(lambda _: next(it), expr)

    out = df if inplace else df.copy(deep=False)
    df_constraints = df.copy(deep=True).astype(object)
    df_constraints = df_constraints.map(lambda _: [])

    for i, conditions in enumerate(selections):
        cond_asts, cond_masks = [], []
        for cond_str in conditions:
            p = parse_expr(cond_str)
            if not all([col in out.columns for col in extract_columns(p)]):
                continue
            cond_asts.append(p)
            cond_masks.append(eval_expr(cond_asts[-1], out))
        cond_cols = [extract_columns(cond_ast) for cond_ast in cond_asts]
        links, to_remove = mapper_inter_rules(out, cond_masks, cond_cols, value_col, id_col)

        if links is None:
            return None
        if to_remove is None or len(to_remove) > 0:
            #in this case, there are duplicate equal rows that are included in a inter-row constraint.
            #E.g., if we want to set the gross income > net income, and there are two rows indicating the net income
            #for the same set of attributes, then in that case the table contains ambiguous contents, and cannot be
            #further processed.
            
            raise ValueError("Inter-row constraints lead to ambiguous linking.")
        

        for j in range(len(links)):
            rule = replace_atoms(rules[i], [f"x_{value_col}_{el}" for el in links[j]])
            # applying rule with replace to adhere to parsing
            rule_tmp = deepcopy(rule)
            for op in op_math_map:
                rule_tmp = rule_tmp.replace(f"{op}", "")
            enforce_expr_inter(parse_expr(rule_tmp.replace(" ","")), df_constraints, links[j], rule, value_col)

    return df_constraints"""

def get_inter_bounds(
    df: pd.DataFrame,
    selections: List[List[str]],
    rules: List[str],
    value_col: str,
    id_col: str,
    *,
    inplace: bool = False,
):
    pat = re.compile(r"\(.*?\)\.\S+")
    to_remove_total = []

    def replace_atoms(expr, repl):
        it = iter(repl)
        return pat.sub(lambda _: next(it), expr)

    out = df if inplace else df.copy(deep=False)
    df_constraints = df.copy(deep=True).astype(object)
    df_constraints = df_constraints.map(lambda _: [])

    for i, conditions in enumerate(selections):
        cond_asts, cond_masks = [], []
        for cond_str in conditions:
            p = parse_expr(cond_str)
            if not all([col in out.columns for col in extract_columns(p)]):
                continue
            cond_asts.append(p)
            cond_masks.append(eval_expr(cond_asts[-1], out))

        cond_cols = [extract_columns(cond_ast) for cond_ast in cond_asts]
        links, to_remove = mapper_inter_rules(out, cond_masks, cond_cols, value_col, id_col)

        if links is None:
            return None, None

        if to_remove is not None and len(to_remove) > 0:
            to_remove_total.extend(to_remove)
            out = out.drop(index=list(to_remove))
            df_constraints = df_constraints.drop(index=list(to_remove))

            # recompute masks/links on the reduced dataframe
            cond_masks = [eval_expr(cond_ast, out) for cond_ast in cond_asts]
            links, _ = mapper_inter_rules(out, cond_masks, cond_cols, value_col, id_col)

            if links is None:
                return None, None

        for j in range(len(links)):
            rule = replace_atoms(rules[i], [f"x_{value_col}_{el}" for el in links[j]])
            rule_tmp = deepcopy(rule)
            for op in op_math_map:
                rule_tmp = rule_tmp.replace(f"{op}", "")
            enforce_expr_inter(parse_expr(rule_tmp.replace(" ","")), df_constraints, links[j], rule, value_col)

    return out, df_constraints, list(set(to_remove_total))

class Constrainer:
    def __init__(self):
        self.PATTERN_intra = re.compile(
            r'^\s*if\s*\(\s*(?P<condition>.+?)\s*\)\s*then\s*(?P<expr>.+?)\s*$',
            re.IGNORECASE
        )

        self.ATOM_NC = r"""
            \(
                \s*(.*?)\s*
            \)
            \s*\.\s*
            ([A-Za-z_][A-Za-z0-9_]*)
        """

        self.ATOM_EXTRACT = r"""
            \(
                \s*(?P<cond>.*?)\s*
            \)
            \s*\.\s*
            (?P<value_col>[A-Za-z_][A-Za-z0-9_]*)
        """

        self.NUMBER = r"""[+-]?\d+(?:\.\d+)?"""
        self.TERM = rf"""(?:{self.ATOM_NC}|{self.NUMBER})"""
        self.MATH_OP = r"""(?:\*\*|[+\-*/%])"""
        self.BOOL_OP = r"""(?:==|!=|<=|>=|<|>|=)"""

        self.EXPR = rf"""{self.TERM}(?:\s*{self.MATH_OP}\s*{self.TERM})*"""

        self.pattern = re.compile(
            rf"""^\s*(?P<lhs>{self.EXPR})\s*(?P<bool_op>{self.BOOL_OP})\s*(?P<rhs>{self.EXPR})\s*$""",
            re.VERBOSE | re.DOTALL
        )

        self.variable_pattern = r"x_(?P<colname>[^\s_]+)_(?P<i>\d+)"

        self.z3_solver = Z3Solver()

    def get_intra_bounds(
            self,
            df: pd.DataFrame,
            rules: List[Tuple[str, str]],
            *,
            inplace: bool = False,
    ) -> pd.DataFrame:
        out = df if inplace else df.copy(deep=False)
        df_constraints = df.copy(deep=True).astype(object)
        df_constraints = df_constraints.map(lambda _: [])

        # Parse once
        parsed_rules = []
        for cond_str, expr_str in rules:
            cond_ast = parse_expr(cond_str)
            expr_ast = parse_expr(expr_str)
            cond_cols = set(extract_columns(cond_ast))
            expr_cols = set(extract_columns(expr_ast))
            parsed_rules.append({
                "cond_str": cond_str,
                "expr_str": expr_str,
                "cond_ast": cond_ast,
                "expr_ast": expr_ast,
                "cond_cols": cond_cols,
                "expr_cols": expr_cols,
            })

        n = len(parsed_rules)

        # Build dependency graph:
        # i -> j if rule i can modify a column that appears in rule j's condition
        deps = {i: set() for i in range(n)}
        rev_deps = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if parsed_rules[i]["expr_cols"] & parsed_rules[j]["cond_cols"]:
                    deps[i].add(j)
                    rev_deps[j].add(i)

        # Precompute current condition / expression masks
        cond_masks = []
        expr_masks = []
        for r in parsed_rules:
            cond_masks.append(eval_expr(r["cond_ast"], out))
            expr_masks.append(eval_expr(r["expr_ast"], out))

        # For each rule j, collect all upstream rules that can influence it (transitively)
        def get_upstream(rule_idx: int) -> set[int]:
            seen = set()
            stack = list(rev_deps[rule_idx])
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(rev_deps[x] - seen)
            return seen

        for j, r in enumerate(parsed_rules):
            # rows that currently violate the rule
            violating = cond_masks[j] #& ~expr_masks[j]

            # rows that may become relevant because upstream rules can alter this condition
            propagated = pd.Series(False, index=out.index)
            for i in get_upstream(j):
                propagated |= cond_masks[i]

            candidate_rows = violating | propagated

            if not candidate_rows.any():
                continue

            success = enforce_expr(r["expr_ast"], r["cond_ast"], df_constraints, candidate_rows)
            if not success:
                return None

        return df_constraints

    # """
    # we apply no violation, as considering as constrained only those rows whose cond is currently satisfied does not
    # completely take into account multi-hop scenarios, i.e. when column A changes B, B changes C etc.
    # This will likely increase runtime, but we will see how to manage that.
    # TODO: fix this problem to improve runtime
    # """

    """def get_intra_bounds(
            self,
            df: pd.DataFrame,
            rules: List[Tuple[str, str]],
            *,
            inplace: bool = False,
    ) -> pd.DataFrame:

        out = df if inplace else df.copy(deep=False)
        df_constraints = df.copy(deep=True).astype(object)
        df_constraints = df_constraints.map(lambda _: [])

        dependencies = [[] * len(out)]

        for i, (cond_str, expr_str) in enumerate(rules):
            cond_ast = parse_expr(cond_str)
            expr_ast = parse_expr(expr_str)

            cond_cols = extract_columns(cond_ast)
            expr_cols = extract_columns(expr_ast)

            # cond_mask = eval_expr(cond_ast, out)
            # expr_mask = eval_expr(expr_ast, out)

            # violating = cond_mask  # & ~expr_mask

            violating = pd.Series(True, index=out.index)
            success = enforce_expr(expr_ast, cond_ast, df_constraints, violating)
            if not success:
                return None

            #out = self.apply_intra(out, df_constraints, value_col="Value", id_col="Id", domains=domains)

        return df_constraints"""

    def strip_outer_parens(self, s: str):
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            return s[1:-1].strip()
        return s


    def parse_intra_rules(self, intra_row_constraints):
        intra_rules = []

        for constr in intra_row_constraints:
            m = self.PATTERN_intra.match(constr)
            if not m:
                return None

            condition = "if".join(constr.split("if")[1:]).split(" then ")[0].strip()
            expr = constr.split(" then ")[1].strip()

            """if " and " or " or " in expr:
                return None"""

            condition = self.strip_outer_parens(condition)
            expr = self.strip_outer_parens(expr)

            intra_rules.append((condition, expr))

        return intra_rules

    def parse_inter_rules(self, inter_row_constraints):
        inter_rules = []
        for constr in inter_row_constraints:
            m = self.pattern.match(constr)
            if not m:
                raise ValueError(f"Inter-row constraint {constr} does not match expected pattern.")

            atoms = [mm.group("cond") for mm in re.compile(self.ATOM_EXTRACT, re.VERBOSE | re.DOTALL).finditer(constr)]
            inter_rules.append(atoms)

        return inter_rules

    def get_bounds(self, df, semantic_constraints, value_col, id_col, domains, random_state=0):
        df_tmp = df.copy(deep=True)
        intra_row_constraints = semantic_constraints.get("intra_row_constraints", None)
        inter_row_constraints = semantic_constraints.get("inter_row_constraints", None)

        if intra_row_constraints is None or inter_row_constraints is None:
            return None, None

        intra_rules = self.parse_intra_rules(intra_row_constraints)
        if intra_rules is None:
            return None, None

        inter_rules = self.parse_inter_rules(inter_row_constraints)
        if inter_rules is None:
            return None, None

        df_constraints_intra = self.get_intra_bounds(df_tmp, intra_rules)
        if df_constraints_intra is None: # error in rule application
            return None, None #df, None

        df_tmp, df_constraints_intra, to_remove_intra = self.apply_intra(df_tmp, df_constraints_intra, value_col=value_col, id_col=id_col, domains=domains)
        if df_tmp is None: # unsat
            return None, None #df, None
        # removing duplicate rows after intra application
        key_cols = df_tmp.columns.difference([value_col]) #, id_col])

        perm = df_tmp.sample(frac=1, random_state=random_state).index
        df_tmp = df_tmp.loc[perm].reset_index(drop=True)
        df_constraints_intra = df_constraints_intra.loc[perm].reset_index(drop=True)
        mask = ~df_tmp.duplicated(subset=key_cols, keep="first")
        df_tmp = df_tmp.loc[mask].reset_index(drop=True)
        df_constraints_intra = df_constraints_intra.loc[mask].reset_index(drop=True)

        #df_constraints_intra = df_constraints_intra.loc[~df_tmp.duplicated(subset=key_cols, keep="first")]
        #df_tmp = df_tmp.loc[~df_tmp.duplicated(subset=key_cols, keep="first")]
        #print(df_constraints_intra)
        #print(df_tmp)

        df_tmp, df_constraints_inter, to_remove = get_inter_bounds(df_tmp, inter_rules, inter_row_constraints, value_col, id_col)
        df_constraints_intra = df_constraints_intra.drop(index=list(to_remove)) # dropping inter ambiguous rows from intra
        #idx = df_constraints_intra.index[df_constraints_intra[value_col].map(len) > 0] # I get the rows where an intra-constraint is applied to the value_col

        for i,_ in df_constraints_inter.iterrows():
            matches = []
            for text in df_constraints_inter.at[i, value_col]:
                matches.extend([int(m.group("i")) for m in re.finditer(self.variable_pattern, text)])
            matches = [el for el in list(set(matches)) if el != i]

            # in this way, intra rules targeting row y are also added if the row y is mentioned in another row x
            for match in matches:
                #df_constraints_inter.at[i, value_col] += df_constraints_intra.at[i, value_col]
                df_constraints_inter.at[i, value_col] += df_constraints_intra.at[match, value_col]

            df_constraints_inter.at[i, value_col] += df_constraints_intra.at[i, value_col]

        if df_constraints_inter is None:
            return None, None #df, None

        """for i in idx:
            value = f"x_{value_col}_{i}"
            for j, _ in df_constraints_inter.iterrows():
                joined_constraints_str = " ; ".join(str(df_constraints_inter.at[j, value_col]))
                if joined_constraints_str != "[ ; ]":
                    c = 5
                    c = c+1

                if value in joined_constraints_str or i == j:
                    df_constraints_inter.at[j, value_col] += df_constraints_intra.at[i, value_col]"""
        #print(df_constraints_inter)
        df_tmp, to_remove_inter = self.apply_inter(df_tmp, df_constraints_inter, value_col)
        to_remove = list(set(to_remove_inter + to_remove_intra))
        df_tmp = df_tmp.drop(index=to_remove).reset_index(drop=True)
        df_constraints = df_constraints_intra + df_constraints_inter
        df_constraints = df_constraints.map(lambda x: list(set(x)))
        df_constraints = df_constraints.drop(index=to_remove).reset_index(drop=True)

        if df_tmp is None: # unsat
            return None, None #df, None

        return df_tmp, df_constraints

    def get_value_col_types(self, df):
        value_col_types = {col: str(df[col].dtype) for col in df.columns}
        for k, v in value_col_types.items():
            if "float" in v:
                value_col_types[k] = "float"
            elif "int" in v:
                value_col_types[k] = "int"
            else:
                value_col_types[k] = "str"

        return value_col_types

    def apply_intra(self, df, df_constraints, value_col, id_col, domains):
        #df_tmp = df.copy(deep=True).drop([value_col, id_col], axis="columns")
        #df_constraints_tmp = df_constraints.copy(deep=True).drop([value_col, id_col], axis="columns")
        self.z3_solver.init_solver()

        value_col_types = self.get_value_col_types(df)
        df_tmp = df.copy(deep=True)
        to_remove = []
        for i, _ in df_tmp.iterrows():
            #for j, col in enumerate(df_tmp.columns):
                """if col in [value_col, id_col]:
                    continue"""
                # matches = [(m.group("colname"), int(m.group("i"))) for col in df_tmp.columns for m in re.finditer(self.variable_pattern, " , ".join([el[0] + " " + el[1] for el in df_constraints.at[i, col]])) if col != value_col]

                matches = [
                    (m.group("colname"), int(m.group("i")))
                    for col in df_tmp.columns
                    if col != value_col
                    for m in re.finditer(
                        self.variable_pattern,
                        " , ".join(f"{a} {b}" for (a, b) in df_constraints.at[i, col])
                    )
                ]
                variable_names = [f"x_{m[0]}_{m[1]}" for m in matches]
                rules = list(chain.from_iterable(df_constraints.loc[i, col] for col in df_tmp.columns if col != value_col))
                solutions = self.z3_solver.intra_chooser(rules, domains, variable_names, value_col_types, list(df_constraints.columns), value_col)
                if solutions is not None:
                    for solution in solutions:
                        variable_parts = solution[0].split("_")
                        colname, index = variable_parts[-2], variable_parts[-1]
                        df.at[int(index), colname] = solution[1]
                else: # unsat
                    to_remove.append(i)
                    #return None

        """df = df.drop(index=to_remove)
        df_constraints = df_constraints.drop(index=to_remove)"""
        return df, df_constraints, to_remove

    def apply_inter(self, df, df_constraints, value_col):
        """TODO: first apply changes from intra, only for non-value columns, since it may change some selector values for inter
        then apply inter by finding the correct amount of values that can be randomized, without making the constraints unsatisfiable
        to randomly get the solutions, we can:
        1) add the intra value constraints into each row that uses that variable (e.g. x_Value_0 >= 300 in 2nd row)
        2) for each constraint, we define a variable ordering and select one variable
            a) we calculate the min/max for that possible variable, and we sample from that range
            b) we fix that variable to the sampled value, and check if the system is still satisfiable (e.g. for non-continuous solution spaces)
                i) if the system is not satisfiable, we resample for maximum N times, or apply the random z3 solution
            c) we repeat until all variables are fixed
        3) we apply the changes to the dataframe
        """

        all_variables = {} # keeping track of already assigned values to variables
        value_col_types = self.get_value_col_types(df)

        for i, row in df.iterrows():
            for column in df.columns:
                if column == value_col:
                    continue
                # we set the variables to pass to z3
                all_variables[f"x_{column}_{i}"] = row[column]

        to_remove = []
        for i, row in df.iterrows():
            rules = df_constraints.at[i, value_col]
            matches = [(m.group("colname"), int(m.group("i"))) for m in re.finditer(self.variable_pattern, ", ".join([str(r) for r in rules]))]
            variable_names = sorted([f"x_{m[0]}_{m[1]}" for m in matches]) #if m[0] != value_col

            tmp_all_variables, tmp_rules = self.z3_solver.inter_chooser(rules, variable_names, value_col, value_col_types, all_variables, index=i)
            if tmp_all_variables is None: # unsat
                to_remove.append(i)
            else:
                all_variables = tmp_all_variables
                rules = tmp_rules

        for variable_name in all_variables:
            variable_split = variable_name.split("_")
            colname, index = variable_split[-2], variable_split[-1]
            if colname != value_col:
                continue # inter constraints do not alter non value columns, so we skip an useless assignment

            df.at[int(index), colname] = all_variables[variable_name]

        #df = df.drop(index=to_remove)
        return df, to_remove

if __name__ == "__main__":
    c = Constrainer()
    """df = pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "TypeOfMoney": ["COGS", "COGS", "YEY", "COGS"],
        "Scenario":    ["Forecast - 2", "Actual - 2", "Forecast - 2", "Forecast - 2"],
        "Bla": ["A", "B", "A", "D"],
        "Value": [100, 200, 300, 400],
    })

    intra_rules = [
        "if (TypeOfMoney == \"COGS\" or Bla == \"A\") then (Value >= 300)",
        "if (Scenario == \"Forecast - 2\") then (Bla != \"A\")",
    ]
    inter_rules = [
        "(TypeOfMoney == \"COGS\").Value + (TypeOfMoney == \"YEY\").Value >= 500.0",
    ]

    print(df)
    domains = [[1, 2, 3, 4], ["COGS", "YEY"], ["Forecast - 2", "Actual - 2"], ["A", "B", "D"], [100, 600]]"""

    df = pd.DataFrame({
        "PortfolioLedger": [
            "BlackRock Global Unconstrained Credit",
            "PIMCO Income Fund",
            "Goldman Sachs Strategic Income Portfolio",
            "J.P. Morgan Income Opportunities Fund",
            "Vanguard Short-Term Corporate Bond Account",
            "PIMCO Income Fund",
        ],
        "BilateralExposureBand": [
            "senior secured tranche - repo eligible",
            "senior unsecured - investment grade",
            "senior unsecured - high yield",
            "subordinated mezzanine layer",
            "junior residual / equity slice",
            "synthetic CDO tranche - non-vanilla",
        ],
        "EffectiveDurationMonths": [60, 1, 90, 180, 48, 360],
        "LiquidityAdjustedNPV": [5000000.00, 2000000.00, 1000000.00, -500000.00, -2000000.00, -700000.00],
    })

    intra_rules = [
        "if (PortfolioLedger == \"Vanguard Short-Term Corporate Bond Account\") then (EffectiveDurationMonths <= 36)",
        "if (PortfolioLedger == \"BlackRock Global Unconstrained Credit\") then (EffectiveDurationMonths >= 36)",
        "if (BilateralExposureBand == \"senior secured tranche - repo eligible\") then (LiquidityAdjustedNPV >= 0)",
        "if (BilateralExposureBand == \"junior residual / equity slice\") then (LiquidityAdjustedNPV <= 0)",
        "if (EffectiveDurationMonths >= 120) then (LiquidityAdjustedNPV <= 0)",
    ]

    inter_rules = [
        "(BilateralExposureBand == \"senior secured tranche - repo eligible\").LiquidityAdjustedNPV >= (BilateralExposureBand == \"senior unsecured - investment grade\").LiquidityAdjustedNPV",
        "(BilateralExposureBand == \"senior unsecured - investment grade\").LiquidityAdjustedNPV >= (BilateralExposureBand == \"senior unsecured - high yield\").LiquidityAdjustedNPV",
        "(BilateralExposureBand == \"senior unsecured - high yield\").LiquidityAdjustedNPV >= (BilateralExposureBand == \"subordinated mezzanine layer\").LiquidityAdjustedNPV",
        "(BilateralExposureBand == \"subordinated mezzanine layer\").LiquidityAdjustedNPV >= (BilateralExposureBand == \"junior residual / equity slice\").LiquidityAdjustedNPV",
        "(BilateralExposureBand == \"synthetic CDO tranche - non-vanilla\").LiquidityAdjustedNPV <= (BilateralExposureBand == \"subordinated mezzanine layer\").LiquidityAdjustedNPV",
        "(EffectiveDurationMonths == 360).LiquidityAdjustedNPV <= (EffectiveDurationMonths == 1).LiquidityAdjustedNPV",
    ]

    print(df)

    domains = [
        [
            "BlackRock Global Unconstrained Credit",
            "PIMCO Income Fund",
            "Goldman Sachs Strategic Income Portfolio",
            "J.P. Morgan Income Opportunities Fund",
            "Vanguard Short-Term Corporate Bond Account",
        ],
        [
            "senior secured tranche - repo eligible",
            "senior unsecured - investment grade",
            "senior unsecured - high yield",
            "subordinated mezzanine layer",
            "junior residual / equity slice",
            "synthetic CDO tranche - non-vanilla",
        ],
        [1, 360],
        [-10000000.0, 100000000.0],
    ]

    """df = pd.DataFrame({
        "InstrumentMnemonic": [
            "JPM — JPMorgan Chase & Co. (common equity, NYSE:JPM)",
            "AAPL — Apple Inc. (common equity, NASDAQ:AAPL)",
            "HSBA — HSBC Holdings plc (ordinary shares, LSE:HSBA)",
            "BABA — Alibaba Group Holding Ltd (ADR, NYSE:BABA)",
            "BP — BP plc (ordinary shares, LSE:BP)",
        ],
        "AccountingLedger": [
            "Core treasury general ledger — mark-to-market adjustments",
            "Valuation reserve ledger — credit valuation adjustments",
            "Hedge accounting ledger — cashflow hedge subledger",
            "Proprietary ALM ledger — duration-matching module",
            "Consolidation ledger — intercompany eliminations",
        ],
        "ConcentrationTranche": [
            "Top-10 single-name unsecured exposures (credit concentration)",
            "Interbank overnight placements — system liquidity corridor",
            "Sectoral: commodity producers exposure tranche (energy/metals)",
            "Regional sovereign bucket — emerging Asia local-currency debt",
            "Collateralised repo counterparties — secured funding tranche",
        ],
        "PrincipalValuation": [1500000000.00, -200000000.00, 25000000.00, 8000000.00, 12000000.00],
        "AsOfDate": ["2023-12-31", "2023-12-31", "2023-09-30", "2022-12-31", "2021-12-31"],
    })

    intra_rules = [
        "if (AccountingLedger == \"Valuation reserve ledger — credit valuation adjustments\") then (PrincipalValuation <= 0)",
        "if (ConcentrationTranche == \"Top-10 single-name unsecured exposures (credit concentration)\") then (PrincipalValuation >= 1000000)",
        "if (ConcentrationTranche == \"Interbank overnight placements — system liquidity corridor\") then (PrincipalValuation <= 500000000)",
    ]

    inter_rules = [
        "(AccountingLedger == \"Core treasury general ledger — mark-to-market adjustments\").PrincipalValuation >= (AccountingLedger == \"Valuation reserve ledger — credit valuation adjustments\").PrincipalValuation",
        "(ConcentrationTranche == \"Top-10 single-name unsecured exposures (credit concentration)\").PrincipalValuation >= (ConcentrationTranche == \"Interbank overnight placements — system liquidity corridor\").PrincipalValuation",
    ]

    print(df)

    domains = [
        [
            "JPM — JPMorgan Chase & Co. (common equity, NYSE:JPM)",
            "AAPL — Apple Inc. (common equity, NASDAQ:AAPL)",
            "HSBA — HSBC Holdings plc (ordinary shares, LSE:HSBA)",
            "BABA — Alibaba Group Holding Ltd (ADR, NYSE:BABA)",
            "BP — BP plc (ordinary shares, LSE:BP)",
        ],
        [
            "Core treasury general ledger — mark-to-market adjustments",
            "Hedge accounting ledger — cashflow hedge subledger",
            "Proprietary ALM ledger — duration-matching module",
            "Consolidation ledger — intercompany eliminations",
            "Valuation reserve ledger — credit valuation adjustments",
        ],
        [
            "Top-10 single-name unsecured exposures (credit concentration)",
            "Sectoral: commodity producers exposure tranche (energy/metals)",
            "Regional sovereign bucket — emerging Asia local-currency debt",
            "Collateralised repo counterparties — secured funding tranche",
            "Interbank overnight placements — system liquidity corridor",
        ],
        [0.0, 2500000000.0],
        ["2023-12-31", "2023-09-30", "2022-12-31", "2022-06-30", "2021-12-31"],
    ]"""

    df, df_constraints = c.get_bounds(df, {"intra_row_constraints": intra_rules, "inter_row_constraints": inter_rules}, "LiquidityAdjustedNPV", "PortfolioLedger", domains)
    #df = c.apply_intra(df, df_constraints, value_col="Value", id_col="Id", domains=domains)
    #print(df_constraints)
    print(df)