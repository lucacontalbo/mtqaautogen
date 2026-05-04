import argparse
import re
import math

from fractions import Fraction

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--question_type', type=str, default='extractive', help='Type of question to generate')
    parser.add_argument('--domain', type=str, default=None, choices=["finance", "healthcare", "products", "environmental"], help='Domain of the tables')
    parser.add_argument('--num_tables', type=int, default=1, help='Number of tables to generate')
    parser.add_argument('--sequential', action="store_true", help="Apply generation of sequential multi-table data")
    parser.add_argument('--generate_ablations', action="store_true", help='Number of tables to generate')

    args = vars(parser.parse_args())

    return args

def get_args_test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--tatqa', action="store_true", help='Whether the dataset is tatqa or not')
    parser.add_argument('--bird', action="store_true", help='Execute BIRD ablations')
    parser.add_argument('--griqa', action="store_true", help='Whether the dataset is griqa or not')

    args = vars(parser.parse_args())
    if args['tatqa'] and args['bird']:
        raise ValueError("Cannot use both tatqa and bird")
    return args

def replace_many(text: str, repl: dict[str, str]) -> str:
    "replace multiple substrings simultaneously"
    pattern = re.compile("|".join(map(re.escape, sorted(repl, key=len, reverse=True))))
    return pattern.sub(lambda m: repl[m.group(0)], text)

def is_float(nbr):
    try:
        if isinstance(nbr, (int, float)):
            return True
        nbr = nbr.replace(",", "")
        float(nbr)
        return True
    except:
        return False


def z3_num_to_float(s: str, default_bound: int = 10**6) -> float:
    _OO_FORMS = re.compile(r"^\s*([+-]?\d+)\s*\*\s*oo\s*$")  # e.g. -1*oo, 1*oo
    _EPSILON_FORMS = re.compile(r"([+\-]?\s*\d*\s*\*?\s*epsilon)") # e.g. -1*epsilon, 1*epsilon, epsilon...

    s = s.strip()
    if s.endswith("?"):
        s = s[:-1].strip()

    if s == "oo":
        return default_bound #math.inf
    if s == "-oo":
        return -default_bound #-math.inf

    m = _OO_FORMS.match(s)
    if m:
        k = int(m.group(1))
        if k == 0:
            return None
        return math.copysign(default_bound, k)

    splits = s.split(" ")
    remove = None
    for i in range(len(splits)):
        if "epsilon" in splits[i]:
            remove = i
            break

    if remove is not None and remove > 1:
        splits = splits[:remove-1] + splits[remove+1:]
        s = " ".join(splits)
    elif remove is not None:
        s = "0"
        #return float(s)

    return float(Fraction(s))