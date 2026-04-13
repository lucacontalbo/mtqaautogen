from z3 import Solver, Optimize, EnumSort, Const, is_bool, sat, Real, Int, Implies, Not, And, Or, Context, is_const, is_app, Z3_OP_UNINTERPRETED
import random
import re

import ast

from utils import is_float, z3_num_to_float
from tqdm import tqdm
from functools import lru_cache


class BoolOpToZ3(ast.NodeTransformer):
    """
    Rewrites Python boolean ops, e.g:
    "(a or b) and not c" -> "And(Or(a, b), Not(c))"
    """

    # TODO: this class is redundant, as python bool -> z3 bool could technically be achieved with the parser in constrainer
    # TODO [cont.]: for the time being, I leave this as it is
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Or):
            return ast.Call(func=ast.Name(id="Or", ctx=ast.Load()),
                            args=node.values, keywords=[])
        if isinstance(node.op, ast.And):
            return ast.Call(func=ast.Name(id="And", ctx=ast.Load()),
                            args=node.values, keywords=[])
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.Call(func=ast.Name(id="Not", ctx=ast.Load()),
                            args=[node.operand], keywords=[])
        return node

DASH_TOKEN = "DASHDASHDASH"
#_dash_ident = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:--[A-Za-z0-9_]+)+)")

@lru_cache(maxsize=None)
def python_bool_to_z3_call(expr_str: str) -> str:
    """
    Returns a rewritten source string, e.g.
      "(a or b) and not c" -> "And(Or(a, b), Not(c))"
    """

    #escaped = _dash_ident.sub(lambda m: m.group(1).replace("--", DASH_TOKEN), expr_str)

    tree = ast.parse(expr_str, mode="eval")
    tree = BoolOpToZ3().visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)

    return out.replace("\"", "").replace("'", "") #out.replace(DASH_TOKEN, "--")

class Z3Solver:
    def __init__(self):
        self.solver = None
        self.optimizer = None
        self.variable_pattern = r"x_(?P<colname>[^\s_]+)_(?P<i>\d+)"


    def init_solver(self):
        self.ctx = Context()
        self.solver = Solver(ctx=self.ctx)
        self.optimizer = Optimize(ctx=self.ctx)
        #self.var_to_value = {}
        self.xs, self.vars_by_name, self.enum_by_name = [], {}, {}
        self.cellTypes, self.z3_domains = {}, {}
        self.value_col_domain = None
        self.placeholders = {}
        self._opt_cache = {}

    def rules_from_strings(self, rule_strs, vars_by_name, enum_by_name):
        env = {}
        env.update(vars_by_name)
        env.update(enum_by_name)
        env.update({"Or": Or, "And": And, "Not": Not})

        out = []
        try:
            for s in rule_strs:
                s2 = python_bool_to_z3_call(s)
                expr = eval(s2, {"__builtins__": {}}, env)  # we disable builtins, and add our own env info
                if not is_bool(expr):
                    raise TypeError(f"Rule did not evaluate to BoolRef: {s}  ->  {s2}  ->  {expr}")
                out.append(expr)
        except:
            pass
        return out

    def preprocess_rule(self, rule: str) -> str:
        """
        1) Replace whitespace runs inside quoted substrings (single or double quotes)
           with the same number of underscores.
        2) Replace each quoted substring with a placeholder: __v1__, __v2__, ...
           and store a mapping in self.placeholders:
               self.placeholders["__v1__"] = <original inner text with underscores, no quotes>

        Notes:
        - Only text inside quotes is turned into placeholders.
        - Quote characters are removed in the stored value.
        - Escaped characters inside quotes are preserved as-is (e.g., \", \').
        """
        if rule is None:
            return rule


        def _substitute(match: re.Match) -> str:
            inner = match.group("inner")

            # Replace each whitespace run with the same number of underscores
            # inner = re.sub(r"\s+", lambda m: "_" * len(m.group(0)), inner)
            inner = inner.replace("\"", "").replace("'", "")

            if inner not in self.placeholders:
                ph = f"__v{len(self.placeholders)}__"
                if is_float(inner):
                    ph = inner
                self.placeholders[inner] = ph
                self.placeholders[ph] = ph
            else:
                ph = self.placeholders[inner]

            return ph

        # Match either "..." or '...', allowing escaped chars inside.
        pattern = r"(?P<q>['\"])(?P<inner>(?:\\.|(?!\1).)*?)\1"
        return re.sub(pattern, _substitute, rule)

    def intra_chooser(self, rules, domains, variable_names, variable_types, columns, value_col, max_num_solutions=5):
        for i, domain in enumerate(domains):
            if i == columns.index(value_col):
                # it feels hacky to set the domain for value_col inside the intra function, since it will be applied in inter
                # however, in our setting, inter_chooser always follows intra_chooser, so the issue should only regard code style
                self.value_col_domain = ["{} >= "+str(domain[0]), "{} <= "+str(domain[1])]
            if columns[i] in self.cellTypes:
                continue

            cellType, z3_domain = EnumSort(f"cellType_{columns[i]}", [self.preprocess_rule(f"\"{str(d)}\"") for d in domain], ctx=self.ctx)
            self.enum_by_name |= {f"{str(name)}": val for name, val in zip(domain, z3_domain)}
            self.enum_by_name |= {str(val): val for name, val in zip(domain, z3_domain)}

            self.cellTypes[columns[i]] = cellType
            self.z3_domains[columns[i]] = z3_domain

        if len(rules) == 0:
            return []

        if not isinstance(variable_names, list):
            variable_names = [variable_names]

        xs_local, to_optimize, colnames = [], [], []
        for variable_name in variable_names:
            if variable_name in self.vars_by_name:
                continue
            colname = variable_name.split("_")[-2]
            #pos_colname = columns.index(colname)
            #if pos_colname == -1:
            #    return None
            if variable_types[colname] != "int":
                xs_local.append(Const(variable_name, self.cellTypes[colname]))
                to_optimize.append(False)
            else:
                xs_local.append(Int(variable_name, ctx=self.ctx))
                to_optimize.append(True)

            colnames.append(colname)
            self.xs.append(xs_local[-1]) #Const(variable_name, self.cellTypes[colname]))
            self.vars_by_name[variable_name] = xs_local[-1]

            #self.enum_by_name |= {name: val for name, val in zip(domains[pos_colname], z3_domains[pos_colname])}

        """xs = [Const(v, cellType) for v in variable_names]
        vars_by_name = {v: x for v, x in zip(variable_names, xs)}
        enum_by_name = {name: val for name, val in zip(domain, z3_domain)}"""

        if isinstance(rules[0][0], str):
            conds, exprs = [self.preprocess_rule(rule[0]) for rule in rules], [self.preprocess_rule(rule[1]) for rule in rules]

            cond_rules = self.rules_from_strings(conds, self.vars_by_name, self.enum_by_name)
            expr_rules = self.rules_from_strings(exprs, self.vars_by_name, self.enum_by_name)

            if len(cond_rules) > 0 and len(expr_rules) > 0:
                rules = [Implies(cond_rule, expr_rule) for cond_rule, expr_rule in zip(cond_rules, expr_rules)]
            else:
                rules = []
            #rules = self.rules_from_strings(rules, vars_by_name, enum_by_name)

        self.solver.add(*rules)

        for x, is_to_opt, colname in zip(xs_local, to_optimize, colnames):
            if not is_to_opt:
                continue

            self.optimizer = Optimize(ctx=self.ctx)
            max_v = self._opt_bound(rules, x, "max")
            self.optimizer = Optimize(ctx=self.ctx)
            min_v = self._opt_bound(rules, x, "min")

            if max_v is None or min_v is None:
                return None, None

            max_v = z3_num_to_float(max_v)
            min_v = z3_num_to_float(min_v)
            if max_v is None or min_v is None:
                return None, None

            if variable_types[colname] == "float":
                value = random.uniform(float(min_v), float(max_v))
            elif variable_types[colname] == "int":
                value = random.randint(int(min_v), int(max_v))
            else:
                return None, None

            self.solver.add(x == value)


        solutions = []

        while self.solver.check() == sat and len(solutions) < max_num_solutions:
            solution = []
            m = self.solver.model()
            rules_to_add = []
            for x, is_to_opt in zip(xs_local, to_optimize):
                val = m.eval(x, model_completion=True)
                if is_to_opt:
                    solution.append((x.decl().name(), val.as_long()))
                else:
                    solution.append((x.decl().name(), val.decl().name()))
                # block this value and continue
                rules_to_add.append(x == val)

            self.solver.add(Not(And(rules_to_add)))
            solutions.append(solution)

        self.solver = Solver(ctx=self.ctx)

        if len(solutions) > 0:
            solution = random.choice(solutions)
        else:
            return None

        new_solution = []
        placeholder_to_original = {v:k for k, v in self.placeholders.items() if k != v}
        for var, val in solution:
            #var = var.replace(DASH_TOKEN, " ") if isinstance(var, str) else var
            #val = val.replace(DASH_TOKEN, " ") if isinstance(val, str) else val
            var = placeholder_to_original[var] if var in placeholder_to_original else var
            val = placeholder_to_original[val] if val in placeholder_to_original else val
            new_solution.append((var, val))

        #solution = [(var.replace(DASH_TOKEN, " "), val.replace(DASH_TOKEN, " ")) for var, val in solution]
        return new_solution

    def _canonicalize_rule_texts(self, rules, x_name):
        var_pat = re.compile(r"\bx_[A-Za-z0-9_]+_\d+\b")
        mapping = {}

        def norm_token(tok):
            if tok == x_name:
                return "__target__"
            if tok not in mapping:
                mapping[tok] = f"__k{len(mapping)}__"
            return mapping[tok]

        canon = []
        for rule in sorted(map(str, rules)):
            canon.append(var_pat.sub(lambda m: norm_token(m.group(0)), rule))
        return tuple(canon), mapping

    def _canonicalize_rule_texts_no_target(self, rules):
        var_pat = re.compile(r"\bx_[A-Za-z0-9_]+_\d+\b")
        mapping = {}

        def norm_token(tok):
            if tok not in mapping:
                mapping[tok] = f"__k{len(mapping)}__"
            return mapping[tok]

        canon = []
        for rule in sorted(map(str, rules)):
            canon.append(var_pat.sub(lambda m: norm_token(m.group(0)), rule))
        return tuple(canon), mapping

    def _opt_cache_key(self, rules, kind, x=None):
        if x is None:
            canon, mapping = self._canonicalize_rule_texts_no_target(rules)
            return (canon, kind), mapping
        canon, mapping = self._canonicalize_rule_texts(rules, x.decl().name())
        return (canon, kind), mapping

    def _vars_in_expr(self, expr):
        out = set()

        def walk(e):
            if is_app(e):
                if e.decl().kind() == Z3_OP_UNINTERPRETED and e.num_args() == 0:
                    out.add(e.decl().name())
                else:
                    for ch in e.children():
                        walk(ch)

        walk(expr)
        return out

    def _build_rule_index(self, rules):
        rule_vars = []
        var_to_rules = {}

        for i, r in enumerate(rules):
            vs = self._vars_in_expr(r)
            rule_vars.append(vs)
            for v in vs:
                var_to_rules.setdefault(v, set()).add(i)

        return rule_vars, var_to_rules

    def _relevant_rules_for_var(self, rules, x, rule_vars=None, var_to_rules=None):
        x_name = x.decl().name()
        if rule_vars is None and var_to_rules is None:
            rule_vars, var_to_rules = self._build_rule_index(rules)

        seen_vars = set()
        seen_rules = set()
        stack = [x_name]

        while stack:
            v = stack.pop()
            if v in seen_vars:
                continue
            seen_vars.add(v)

            for ridx in var_to_rules.get(v, ()):
                if ridx in seen_rules:
                    continue
                seen_rules.add(ridx)

                for v2 in rule_vars[ridx]:
                    if v2 not in seen_vars:
                        stack.append(v2)

        return [rules[i] for i in sorted(seen_rules)]

    def _opt_bound(self, rules, x, kind="max", rule_vars=None, var_to_rules=None):
        filtered_rules = self._relevant_rules_for_var(rules, x, rule_vars, var_to_rules) # taking only the rules needed for variable x
        #key = self._opt_cache_key(filtered_rules, kind, x) # per formula caching

        #if key in self._opt_cache:
        #    return self._opt_cache[key]

        self.optimizer.add(*filtered_rules)
        h = self.optimizer.maximize(x) if kind == "max" else self.optimizer.minimize(x)
        res = self.optimizer.check()
        if res != sat:
        #    self._opt_cache[key] = None
            return None

        val = str(self.optimizer.upper(h)) if kind == "max" else str(self.optimizer.lower(h))
        #self._opt_cache[key] = val
        return val

    def _solv_bound(self, rules, x, rule_vars=None, var_to_rules=None, gap=50):
        filtered_rules = self._relevant_rules_for_var(rules, x, rule_vars, var_to_rules) # taking only the rules needed for variable x

        # first value
        self.solver.add(*filtered_rules)
        res = self.solver.check()
        if res == sat:
            m = self.solver.model()
            min_v = m.eval(x)
        else:
            return None, None

        # second value
        self.solver.push()
        self.solver.add(x >= min_v + gap)
        res = self.solver.check()
        if res == sat:
            m = self.solver.model()
            max_v = m.eval(x)
        else:
            self.solver.pop()
            self.solver.push()
            self.solver.add(x != min_v)
            res = self.solver.check()
            if res == sat:
                m = self.solver.model()
                max_v = m.eval(x)
                if z3_num_to_float(str(max_v)) < z3_num_to_float(str(min_v)):
                    tmp = max_v
                    max_v = min_v
                    min_v = tmp
            else:
                self.solver.pop()
                return None, None

        self.solver.pop()
        return str(min_v), str(max_v)

    def find_external_variables(self, variables, index):
        return list(set([v for v in variables if str(index) not in v.split("_")[-1]]))

    def is_variable_dependant(self, var_name, rules, var_to_value):
        for rule in rules:
            vars_found = [m.group(0) for m in re.finditer(self.variable_pattern, str(rule))]
            if var_name not in vars_found:
                continue

            if len(vars_found) > 1:
                for variable in vars_found:
                    if var_name == variable:
                        continue
                    if variable in var_to_value:
                        return True

        return False


    def sample_from_maxmin(self, rules, variables, value_col, variable_types, var_to_value, index=None):
        """
        following `variables` ordering, get the maximum and minimum value of the variable, and randomly sample from there
        the process is repeated for each variable, until we get a solution
        """
        if index is not None:
            external_variables = self.find_external_variables(variables, index)
        else:
            external_variables = []

        max_signature, max_mapping = self._opt_cache_key(rules, "max")
        min_signature, min_mapping = self._opt_cache_key(rules, "min")

        self.optimizer = Optimize(ctx=self.ctx)

        if len(rules) == 0:
            return var_to_value, rules
        #self.init_solver()

        xs, allowed_variables, non_value_variables, non_value_xs = [], [], [], []
        for v in set(variables): # + list(var_to_value.keys())):
            colname = v.split("_")[-2]

            if variable_types[colname] != "float" and variable_types[colname] != "int":
                xs.append(Const(v, self.cellTypes[colname]))
                non_value_variables.append(v)
                non_value_xs.append(xs[-1])
            else:
                xs.append(Real(v, ctx=self.ctx) if variable_types[colname].lower().strip() == "float" else Int(v, ctx=self.ctx))
            allowed_variables.append(v)

        #xs = [Real(v) if value_col_type.lower().strip() == "float" else Int(v) for v in variables] + self.xs
        vars_by_name = {var_name: value for var_name, value in zip(allowed_variables, xs) if var_name not in self.vars_by_name} | self.vars_by_name

        rules_str = [rule for rule in rules if isinstance(rule, str)]
        rules_cond = list(set(rules) - set(rules_str))

        rules = self.rules_from_strings([self.preprocess_rule(r) for r in rules_str], vars_by_name, {})
        old_rules = [self.preprocess_rule(f"{k} == \"{v}\"") for k,v in var_to_value.items() if k in allowed_variables]

        old_rules = self.rules_from_strings(old_rules, vars_by_name, self.enum_by_name)
        if len(rules) > 0 and len(old_rules) > 0:
            rules += old_rules
        elif len(rules) > 0 and len(old_rules) == 0:
            pass
        elif len(rules) == 0 and len(old_rules) > 0:
            rules = old_rules
        else:
            rules = []

        if len(rules_cond) > 0:
            conds, exprs = [self.preprocess_rule(rule[0]) for rule in rules_cond], [self.preprocess_rule(rule[1]) for rule in rules_cond]
            # add equality constraints

            cond_rules = self.rules_from_strings(conds, vars_by_name, self.enum_by_name)
            expr_rules = self.rules_from_strings(exprs, vars_by_name, self.enum_by_name)

            if len(cond_rules) > 0 and len(expr_rules) > 0:
                rules_cond = [Implies(cond_rule, expr_rule) for cond_rule, expr_rule in zip(cond_rules, expr_rules)]
            else:
                rules_cond = []

            rules += rules_cond

        #for x, v_name in zip(xs, non_value_variables):
        #result = _pick_discrete_model(non_value_xs, rules)
        #var_to_value |= result
        #rules += self.rules_from_strings([f"{k} == {v}" for k,v in result.items()], vars_by_name, self.enum_by_name)

        domain_rules = []
        for x, v_name in zip(xs, allowed_variables):
            if v_name in non_value_variables or value_col not in v_name:
                continue

            for bound in self.value_col_domain:
                domain_rules.append(bound.format(v_name))

        domain_rules = self.rules_from_strings(domain_rules, vars_by_name, self.enum_by_name)
        rules += domain_rules
        #self.optimizer.add(*rules)

        """dependant = False
        for x, v_name in zip(xs, allowed_variables):
            #if v_name in var_to_value and v_name.split("_")[-1] != str(index):
            if v_name.split("_")[-1] != str(index):
                dependant = True
                break"""

        for x, v_name in zip(xs, allowed_variables):
            if v_name in non_value_variables or v_name in var_to_value:
                continue

            dependant = self.is_variable_dependant(v_name, rules, var_to_value)

            #if not dependant and max_signature in self._opt_cache and max_mapping[v_name] in self._opt_cache[max_signature]:
            #    max_v = self._opt_cache[max_signature][max_mapping[v_name]] # this is the error: in subsequent iterations, it may use bounds that are not updated wrt. other variables
            #    min_v = self._opt_cache[min_signature][min_mapping[v_name]]
            #else:
            rule_vars, var_to_rules = self._build_rule_index(rules)
            if True: #max_signature in self._opt_cache and max_mapping[v_name] in self._opt_cache[max_signature]:
                self.solver = Solver(ctx=self.ctx)
                min_v, max_v = self._solv_bound(rules, x, rule_vars, var_to_rules)
            else:
                self.optimizer = Optimize(ctx=self.ctx)
                max_v = self._opt_bound(rules, x, "max", rule_vars, var_to_rules)
                if max_signature not in self._opt_cache:
                    self._opt_cache[max_signature] = {}
                self._opt_cache[max_signature][max_mapping[v_name]] = max_v

                self.optimizer = Optimize(ctx=self.ctx)
                min_v = self._opt_bound(rules, x, "min", rule_vars, var_to_rules)
                if min_signature not in self._opt_cache:
                    self._opt_cache[min_signature] = {}
                self._opt_cache[min_signature][min_mapping[v_name]] = min_v

            if max_v is None or min_v is None:
                return None, None

            max_v = z3_num_to_float(max_v)
            min_v = z3_num_to_float(min_v)
            if max_v is None or min_v is None:
                return None, None

            col_name = v_name.split("_")[-2]
            if variable_types[col_name] == "float": #variable_types[value_col] == "float":
                value = random.uniform(float(min_v), float(max_v))
            elif variable_types[col_name] == "int": #variable_types[value_col] == "int":
                value = random.randint(int(min_v), int(max_v))
            else:
                return None, None
                #raise ValueError(f"unknown type {value_col_type} for mathematical optimization")

            var_to_value[v_name] = value
            #self.optimizer.add(x == value)
            #rules.append(f"{v_name} == {value}")
            rules += self.rules_from_strings([f"{v_name} == {value}"], vars_by_name, {})

        new_vars = {}
        placeholder_to_original = {v:k for k, v in self.placeholders.items() if k != v}
        for var, val in var_to_value.items():
            #var = var.replace(DASH_TOKEN, " ") if isinstance(var, str) else var
            #val = val.replace(DASH_TOKEN, " ") if isinstance(val, str) else val
            var = placeholder_to_original[var] if var in placeholder_to_original else var
            val = placeholder_to_original[val] if val in placeholder_to_original else val
            new_vars[var] = val

        #var_to_value = {var.replace(DASH_TOKEN, " "): val.replace(DASH_TOKEN, " ") for var, val in var_to_value.items()}

        return new_vars, rules


    def inter_chooser(self, rules, variables, value_col, variable_types, var_to_value, index=None):
        if any(["**" in rule or "%" in rule for rule in rules]):
            # solution space is not continuous: we apply randomization by adding random constraints
            return None, None
        else:
            # solution space is continuous (linear): we get the max and min of the solution following the variable ordering
            # we sample from those bounds, and repeat the process for each variable

            var_to_value, rules = self.sample_from_maxmin(rules, variables, value_col, variable_types, var_to_value, index=index)

        return var_to_value, rules