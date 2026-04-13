import pandas as pd
from itertools import combinations
import random
import numpy as np

class Perturber:
    def __init__(self):
        self.num_columns_range = [2,10]
        self.num_rows_range = [2,10]
        self.num_blanks_range = [1,3] #num blank rows to insert
        self.nan_fill_str = [
            "n.a.",
            "-",
            "not available",
            "",
            "\\",
            "N/A"
        ]

        self.PARENTHETICAL_NOTES = [
            "{value} (see notes)",
            "{value} (est.)",
            "{value} (latest update)",
            "{value} (source: official reports)",
        ]

        self.FOOTNOTE_NOTES = [
            "{value}¹",
            "{value}²",
            "{value}³",
            "{value}⁴",
            "{value}⁵",
            "{value}*",
            "{value}**",
            "{value} (see footnote {number})",
            "{value} (see [{number}])",
            "{value} (cf. appendix)"
        ]

        self.CONTEXTUAL_PHRASES = [
            "{value} - according to official data",
            "{value} - as listed on the website",
            "{value} - noted in the catalog",
            "{value} - additional details are provided in the documentation",
            "{value} - Note: recorded in [{number}]",
            "{value}\nAccording to official data",
            "{value}\nAs listed on the website",
            "{value}\nNoted in the catalog",
            "{value}\nAdditional details are provided in the documentation",
            "{value}\nNote: recorded in [{number}]",
            "according to official data: {value}",
            "as listed on the website: {value}",
        ]

        self.pre_hct_perturbations = [
            # self.null_perturbation,  # cell level perturbation
            self.column_merging_perturbation,  # schema/row level perturbation #pre
            # self.row_merging_perturbation,  # schema/row level perturbation #pre
        ]

        self.post_hct_perturbations = [
            self.typos_insertion,  # cell level perturbation
            self.insert_blank_rows,  # schema/row level perturbation
            self.insert_blank_columns,  # schema/row level perturbation
            self.add_sentences_inside_cells,  # cell level perturbation #pre
        ]

        self.rng = random.Random(-1)

    def set_random_seed(self, i):
        self.rng = random.Random(i)

    # UTILITY FUNCTIONS
    def shrink_pivot(
            self,
            pivot: pd.DataFrame,
            keep_rows=None,
            keep_cols=None,
            random_state: int = 42,
    ):
        rng = np.random.default_rng(random_state)
        reduce_rows, reduce_cols = True, True

        if self.num_rows_range[1] > len(pivot.index):
            if self.num_rows_range[0] > len(pivot.index):
                #print("Pivot has less rows than the minimum required!")
                return None
            else:
                reduce_rows = False

        if self.num_columns_range[1] > len(pivot.columns):
            if self.num_columns_range[0] > len(pivot.columns):
                #print("Pivot has less columns than the minimum required!")
                return None
            else:
                reduce_cols = False

        n_rows = self.rng.randint(self.num_rows_range[0], self.num_rows_range[1])
        n_cols = self.rng.randint(self.num_columns_range[0], self.num_columns_range[1])

        if keep_rows is None:
            keep_rows = []
        if keep_cols is None:
            keep_cols = []

        # Rows
        if reduce_rows:
            keep_rows_idx = pivot.index.intersection(keep_rows)
            n_keep_rows = len(keep_rows_idx)

            if n_keep_rows > self.num_rows_range[1]:
                #print("Too many needed rows!")
                return None  # number of needed rows is higher than the rows we want to keep

            if n_keep_rows > self.num_rows_range[0]:
                n_rows = self.rng.randint(n_keep_rows, self.num_rows_range[1])

            n_rows = self.num_rows_range[1] # TODO: remove

            """if n_keep_rows > n_rows:
                print("Too many needed rows!")
                return None # number of needed rows is higher than the rows we want to keep
                #rows_to_keep = keep_rows_idx
            else:"""
            candidates = pivot.index.difference(keep_rows_idx)
            n_to_sample = max(0, n_rows - n_keep_rows)
            if n_to_sample > len(candidates):
                sampled = candidates
            else:
                #sampled = pd.Index(
                #    rng.choice(candidates.to_list(), size=n_to_sample, replace=False)
                #)
                sampled_pos = rng.choice(len(candidates), size=n_to_sample, replace=False)
                sampled = candidates.take(sampled_pos)

            rows_to_keep = keep_rows_idx.union(sampled)
            pivot = pivot.loc[rows_to_keep]

        # Columns
        if reduce_cols:
            keep_cols_idx = pivot.columns.intersection(keep_cols)
            n_keep_cols = len(keep_cols_idx) #len(keep_cols_idx.names) #len(keep_cols_idx)
            if n_keep_cols > self.num_columns_range[1]:
                #print("Too many needed columns!")
                return None

            if n_keep_cols > self.num_columns_range[0]:
                n_cols = self.rng.randint(n_keep_cols, self.num_columns_range[1])

            n_cols = self.num_cols_range[1] # TODO: remove


            """if n_keep_cols > n_cols:
                print("Too many needed columns!")
                return None
                #cols_to_keep = keep_cols_idx
            else:"""
            candidates = pivot.columns.difference(keep_cols_idx)
            n_to_sample = max(0, n_cols - n_keep_cols)
            if n_to_sample > len(candidates):
                sampled = candidates
            else:
                #sampled = pd.Index(
                #    rng.choice(candidates.to_list(), size=n_to_sample, replace=False)
                #)
                sampled_pos = rng.choice(len(candidates), size=n_to_sample, replace=False)
                sampled = candidates.take(sampled_pos)

            cols_to_keep = keep_cols_idx.union(sampled)

            pivot = pivot.loc[:, cols_to_keep]

        return pivot

    def insert_unit_of_measurement(self, table: pd.DataFrame, value_col: str, units_of_measurement: list, unit_in_cell: bool = False):
        if not unit_in_cell and len(set([unit for unit in units_of_measurement if unit is not None and unit != "None"])) != 1:
            unit_in_cell = True

        if len(units_of_measurement) == 0 or units_of_measurement[0] is None or units_of_measurement[0] == "None":
            return table, value_col

        if unit_in_cell:
            # insert the unit inside every cell
            new_value_col = value_col
            for i, (idx, row) in enumerate(table.iterrows()):
                val = str(row[value_col])
                if val.strip() not in self.nan_fill_str and pd.notna(row[value_col]):
                    table.at[idx, value_col] = f"{val} {units_of_measurement[0]}"
        else:
            # insert the unit inside the column text if the unit is always the same
            unit = [unit for unit in units_of_measurement if unit is not None and unit != "None"][0]
            new_value_col = f"{value_col} ({unit})"
            table.rename(columns={value_col: new_value_col}, inplace=True)

        return table, new_value_col

    # SCHEMA LEVEL PERTURBATIONS
    def column_merging_perturbation(self, table: pd.DataFrame, value_col: str, columns: list[str] = None, constraints: list[str] = None, **kwargs):
        """
        merges two random columns into one
        """

        def avoid_answer_collapse(columns: list[str], constraints: list[str], col_to_avoid: str):
            pos = columns.index(col_to_avoid)
            constraints_tmp = [[constraints[i][c] for c in range(len(constraints[i])) if c != pos] for i in range(len(constraints))]
            return any(constraints_tmp.count(x) > 1 for x in constraints_tmp)

        if len(table.columns) < 2:
            return table

        cols = list(table.columns)
        col1, col2 = self.rng.sample(cols, 2)
        if columns is not None and constraints is not None:
            if col1 == value_col:
                if col2 in columns and avoid_answer_collapse(columns, constraints, col2):
                    return table, value_col
            elif col2 == value_col:
                if col1 in columns and avoid_answer_collapse(columns, constraints, col1):
                    return table, value_col

        new_col_name = f"{col1} ({col2})"
        table[new_col_name] = table[col1].astype(str) + " (" + table[col2].astype(str) + ")"

        table.drop(columns=[col1, col2], inplace=True)

        if value_col == col1 or value_col == col2:
            value_col = new_col_name

        return table, value_col

    def row_merging_perturbation(self, table: pd.DataFrame):
        """
        merges two random rows into one
        """

        if len(table) < 2:
            return table

        row_indices = list(table.index)
        row1_idx, row2_idx = self.rng.sample(row_indices, 2)

        new_row = {}
        for col in table.columns:
            val1 = str(table.at[row1_idx, col])
            val2 = str(table.at[row2_idx, col])
            new_row[col] = f"{val1} ({val2})"

        table = table.drop(index=[row1_idx, row2_idx])
        new_row_df = pd.DataFrame([new_row], index=[f"{row1_idx}_{row2_idx}"])
        table = pd.concat([table, new_row_df], ignore_index=False)

        return table

    def insert_blank_rows(self, table: pd.DataFrame, **kwargs):
        """
        Insert a random number of blank rows (within num_blanks_range) into the table.

        - The table's structure (including MultiIndex on rows / nested columns) is preserved.
        - All cells in the inserted rows are "".
        - All index labels for the inserted rows are "" (for every level if MultiIndex).
        """

        n_blanks = self.rng.randint(self.num_blanks_range[0], self.num_blanks_range[1])
        result = table.copy()

        for _ in range(n_blanks):
            pos = self.rng.randint(0, len(result))

            blank_row = pd.DataFrame(
                [[""] * len(result.columns)],
                columns=result.columns,
            )

            if isinstance(result.index, pd.MultiIndex):
                # tuple with the correct number of levels, all ""
                blank_index = pd.MultiIndex.from_tuples(
                    [tuple([""] * result.index.nlevels)],
                    names=result.index.names,
                )
            else:
                blank_index = pd.Index([""], name=result.index.name)

            blank_row.index = blank_index

            top = result.iloc[:pos]
            bottom = result.iloc[pos:]
            result = pd.concat([top, blank_row, bottom])

        return result

    def insert_blank_columns(self, table: pd.DataFrame, **kwargs):
        """
        Insert a random number of blank columns (within num_blanks_range) into the table.

        - The table's structure (including MultiIndex columns) is preserved.
        - All cells in inserted columns are "".
        - Column labels for inserted columns are "" (for every level if MultiIndex).
        """

        n_blanks = self.rng.randint(self.num_blanks_range[0], self.num_blanks_range[1])
        result = table.copy()

        for _ in range(n_blanks):
            pos = self.rng.randint(0, len(result.columns))

            if isinstance(result.columns, pd.MultiIndex):
                blank_col = tuple([""] * result.columns.nlevels)

                blank_col_df = pd.DataFrame(
                    {blank_col: [""] * len(result)},
                    index=result.index,
                )

            else:
                blank_col = ""
                blank_col_df = pd.DataFrame(
                    {blank_col: [""] * len(result)},
                    index=result.index,
                )

            left = result.iloc[:, :pos]
            right = result.iloc[:, pos:]
            result = pd.concat([left, blank_col_df, right], axis=1)

        return result

    def _choose_best_pivot_split(self, table: pd.DataFrame, candidate_columns: list[str], value_col: str, aggr='first'):
        """
        Randomly choose among the splits that produce the densest pivot.
        Prefer splits with no NaNs. If none exist, choose randomly among the highest-density splits.
        Returns: (rows_chosen, cols_chosen, pivot, density)
        """
        if len(candidate_columns) < 2:
            return None, None, None

        candidates = []

        num_tries = 1000
        for num_col_indices in list(range(1, 5))[::-1]: # len(candidate_columns) # capping at 5 to avoid combinatorial explosion
            if num_tries == 0:
                break

            # all ways to choose columns for the pivot columns axis
            for cols_tuple in combinations(candidate_columns, num_col_indices):
                if num_tries == 0:
                    break
                cols_chosen = list(cols_tuple)
                rows_chosen = [c for c in candidate_columns if c not in cols_chosen]

                try:
                    num_tries -= 1
                    pivot = pd.pivot_table(
                        table,
                        values=value_col,
                        index=rows_chosen,
                        columns=cols_chosen,
                        aggfunc=aggr
                    )
                except Exception:
                    continue

                if pivot is None or pivot.empty:
                    continue
                if 1 in pivot.shape: # removing collapsing pivots
                    continue

                total_cells = pivot.shape[0] * pivot.shape[1]
                if total_cells == 0:
                    continue

                non_null = int(pivot.notna().sum().sum())
                density = non_null / total_cells
                has_no_nans = (non_null == total_cells)

                candidates.append({
                    "rows": rows_chosen,
                    "cols": cols_chosen,
                    "density": density,
                    "has_no_nans": has_no_nans,
                })

        if not candidates:
            return None, None, None

        perfect = [c for c in candidates if c["has_no_nans"]]
        if perfect:
            chosen = self.rng.choice(perfect)
            return chosen["rows"], chosen["cols"], chosen["density"]

        best_density = max(c["density"] for c in candidates)
        best = [c for c in candidates if c["density"] == best_density]
        chosen = self.rng.choice(best)

        return chosen["rows"], chosen["cols"], chosen["density"]

    def restore_needed_cells_after_value_merge(
            self,
            table_before_pivot: pd.DataFrame,
            pivot_table: pd.DataFrame,
            value_col: str,
            full_mask: pd.Series,
            rows_chosen: list[str],
            cols_chosen: list[str],
    ):
        """
        After pivoting, restore the needed cells from the pre-pivot table into the pivoted table.

        For every row marked True in full_mask:
          - compute the pivot row key from rows_chosen
          - compute the pivot column key from cols_chosen
          - if that coordinate exists in the pivot_table, overwrite that cell with the exact
            value from table_before_pivot[value_col]
        """
        if pivot_table is None or table_before_pivot is None or full_mask is None:
            #print("*********************************************************** failure 1")
            return pivot_table

        if value_col not in table_before_pivot.columns:
            #print("*********************************************************** failure 2")
            return pivot_table

        needed_indices = full_mask[full_mask].index
        if len(needed_indices) == 0:
            #print("*********************************************************** failure 3")
            return pivot_table

        out = pivot_table.copy()

        def make_key(row: pd.Series, cols: list[str]):
            if len(cols) == 1:
                return row[cols[0]]
            return tuple(row[c] for c in cols)

        for idx in needed_indices:
            if idx not in table_before_pivot.index:
                continue

            src_row = table_before_pivot.loc[idx]

            row_key = make_key(src_row, rows_chosen)
            col_key = make_key(src_row, cols_chosen)
            new_value = src_row[value_col]

            # only patch if the shrunk pivot still contains that coordinate
            if row_key in out.index and col_key in out.columns:
                out.at[row_key, col_key] = new_value

        return out

    def multiheader_perturbation(self,
                                 table: pd.DataFrame,
                                 value_col: str,
                                 id_col: str,
                                 table_types: list,
                                 aggr='first',
                                 unit_in_cell: bool = False,
                                 full_mask: pd.Series | None = None,
                                 rows_chosen=None,
                                 cols_chosen=None):

        columns = list(table.columns)
        #table_types.pop(columns.index(value_col))
        columns.pop(columns.index(value_col))
        #if id_col in columns:
        #    table_types.pop(columns.index(id_col))
        #    columns.pop(columns.index(id_col))
        #columns = [col for i, col in enumerate(columns) if table_types[i] == 'categorical']
        if len(columns) < 2:
            return None, None, None, None

        counter, patience, pivot_small = 0, 10, None
        while counter < patience:
            counter += 1
            num_col_indices = self.rng.randint(1, len(columns)-1)
            #cols_chosen = random.sample(columns, num_col_indices)
            #rows_chosen = [col for col in columns if col not in cols_chosen]
            if rows_chosen is None and cols_chosen is None or counter > 1:
                rows_chosen, cols_chosen, density = self._choose_best_pivot_split(table, columns, value_col, aggr=aggr)
            if rows_chosen is None:
                return None, None, None, None

            pivot = pd.pivot_table(
                table,
                values=value_col,
                index=rows_chosen,
                columns=cols_chosen,
                aggfunc=aggr
            )

            pivot_mask = pd.pivot_table(
                table.assign(__mask__=full_mask.astype(int)),
                values="__mask__",
                index=rows_chosen,
                columns=cols_chosen,
                aggfunc="max",
            ).fillna(0).astype(int).astype(bool)

            rows_must_keep = pivot_mask.any(axis=1)
            cols_must_keep = pivot_mask.any(axis=0)

            keep_rows = pivot.index[rows_must_keep]
            keep_cols = pivot.columns[cols_must_keep]

            try:
                pivot_small = self.shrink_pivot( #shrinking the pivoted table to the desired row/column size
                    pivot,
                    keep_rows=keep_rows,
                    keep_cols=keep_cols,
                )
                if pivot_small is not None:
                    break
            except:
                pass
            #pivot_small = pivot

        if pivot_small is None:
            return None, None, None, None

        option = -1
        if not unit_in_cell:
            """
            if the unit is not inside each cell, then it is in the value_col name, which is gone after pivoting.
            In this case, we add it back in one of many possible ways.
            """
            only_unit = self.rng.randint(0,1) # choose whether to add only the unit, or the whole value_col name
            if only_unit:
                new_value_col = value_col.split('(')[-1].replace(')', '').strip()
            else:
                new_value_col = value_col

            option = self.rng.randint(1,1) #2) # TODO: change options
            if option == 0:
                # added in the upper left side corner of the pivoted table
                pivot_small.columns.name = new_value_col
            elif option == 1:
                # added as a new row on top of the pivoted table
                filled = self.rng.randint(0,1) # choose whether the new_value_col is added only for the first column, or repeated for all columns
                if not filled:
                    new_row = [[new_value_col] + [''] * (len(pivot_small.columns)-1)]
                else:
                    new_row = [[new_value_col] * len(pivot_small.columns)]

                if isinstance(pivot_small.index, pd.MultiIndex):
                    header_idx = tuple([new_value_col] * pivot_small.index.nlevels)
                    new_index = pd.MultiIndex.from_tuples([header_idx],names=pivot_small.index.names)
                else:
                    new_index = [new_value_col]

                new_row = pd.DataFrame(new_row, columns=pivot_small.columns, index=new_index)
                pivot_small = pd.concat([new_row, pivot_small], axis=0)
            #elif option == 2:
                # concatenates the new_value_col to each column name
            #    cols = [col if not isinstance(col, tuple) else " ".join([str(el) for el in list(col)]) for col in pivot_small.columns]
            #    pivot_small.columns = [f"{new_value_col} - {col}" if not only_unit else f"{col} ({new_value_col})" for col in cols]
            """else:
                # add an additional multi-header level with the new_value_col
                pivot.columns = pd.MultiIndex.from_product([[new_value_col], pivot.columns])"""

        return pivot_small, rows_chosen, cols_chosen, option

    # CELL LEVEL PERTURBATIONS
    def null_perturbation(self, table: pd.DataFrame, constraints: list, value_col: str):
        """
        inserts null values into the table
        """

        def get_first_repetition(lst):
            seen = set()
            for i,x in enumerate(lst):
                if x in seen:
                    return i
                seen.add(x)
            return -1

        # in comparison we have repeated flattened columns inside constraints
        # the constraints do not apply for a single value, but they search for multiple values
        # we get the boundaries of each searched value in the following lines
        first_repetition = get_first_repetition([col for col, _ in constraints]) #constraints) #.keys()))
        masks = []
        for i, (col, val) in enumerate(constraints):
            if i == 0 or i % first_repetition == 0:
                if i != 0:
                    if first_repetition != -1:
                        masks.append(mask) # if repetition is found, we store all the masks

                if not (first_repetition == -1 and i != 0):
                    mask = pd.Series(True, index=table.index) # we compute a new mask for the new value we search

            mask &= table[col] == val # extracting all rows satisfying the constraints

        masks.append(mask)

        # we apply an OR operator so that all rows that are needed for an SQL query are kept
        mask = pd.concat(masks, axis=1).any(axis=1)

        unmasked_indices = table.index[~mask]

        nan_fill_nbr = self.rng.randint(0,5)
        nan_fill_str_chosen = self.nan_fill_str[nan_fill_nbr]

        for idx in unmasked_indices:
            num_cols_to_nan = self.rng.randint(0,1)  # randomly choose how many columns to nan
            if num_cols_to_nan == 1:
                table.loc[idx, value_col] = nan_fill_str_chosen #np.nan

        # TODO: insert random row full of nan

        return table

    def add_sentences_inside_cells(self, table: pd.DataFrame, strength: int = 20) -> pd.DataFrame:
        """
        For each of PARENTHETICAL_NOTES, FOOTNOTE_NOTES, CONTEXTUAL_PHRASES,
        I randomly decide whether to use that category. For each chosen category,
        I select one template. Then I apply one of the chosen templates to ~strength% of
        the (non-null) cells, at most one template per cell.
        """
        if strength < 0 or strength > 100:
            raise ValueError("add_sentences_inside_cells strength must be between 0 and 100")

        df = table.copy()

        available_categories = [
            "PARENTHETICAL_NOTES",
            "FOOTNOTE_NOTES",
            "CONTEXTUAL_PHRASES",
        ]

        chosen_categories = []
        for cat in available_categories:
            if self.rng.random() < 0.5: # 50% chance to choose each category
                chosen_categories.append(cat)

        # at least one category must be chosen
        if not chosen_categories:
            chosen_categories.append(self.rng.choice(available_categories))

        # pick one template from each chosen category
        chosen_templates = []
        for cat in chosen_categories:
            templates = getattr(self, cat)
            chosen_templates.append(self.rng.choice(templates))

        n_rows, n_cols = df.shape
        candidate_coords = []

        for i in range(n_rows):
            for j in range(n_cols):
                val = df.iat[i, j]
                # perturbing only values that are not null, or not in the nan_fill_str list
                if pd.notna(val) and str(val).strip() not in self.nan_fill_str:
                    candidate_coords.append((i, j))

        if not candidate_coords:
            return df

        num_to_perturb = max(1, int(len(candidate_coords) * strength / 100))
        num_to_perturb = min(num_to_perturb, len(candidate_coords))

        coords_to_perturb = self.rng.sample(candidate_coords, k=num_to_perturb)

        for i, j in coords_to_perturb:
            original_value = df.iat[i, j]
            original_str = str(original_value)

            template = self.rng.choice(chosen_templates)
            number = self.rng.randint(1, 5)

            new_value = template.format(value=original_str, number=number)
            df.iat[i, j] = new_value

        return df

    def typos_insertion(self, table: pd.DataFrame, strength: int = 20) -> pd.DataFrame:
        """
        Apply character swaps or insertion of Unicode '\\u' characters to ~strength% of
        cells (including headers and indices).

        - Can touch cells that contain numbers, but MUST NOT swap digits.
          Swaps/insertions are applied only to textual (non-digit) characters.
        - For each chosen cell:
            * If swaps are chosen: apply 1–2 nearby swaps on non-digit chars.
            * If Unicode insertion is chosen: insert exactly one Unicode char,
              positioned next to non-digit characters (not inside a pure digit block).
        - Works on data cells, row index labels, and column labels.
        - Preserves MultiIndex structure.

        the strength parameter is an integer from 0 to 100 that indicates the percentage of candidate cells to perturb,
        and the percentage of character swaps or unicode insertions to add to each selected cell.
        """
        if strength < 0 or strength > 100:
            raise ValueError("typo insertion strength must be between 0 and 100")

        df = table.copy()

        unicode_noise_chars = [
            "\u200b",  # zero width space
            "\u200c",  # zero width non-joiner
            "\u200d",  # zero width joiner
            "\u2060",  # word joiner
            "\u00a0",  # non-breaking space
        ]

        def has_textual_chars(s: str) -> bool:
            """Returns True if there is at least one non-digit character."""
            return any(not ch.isdigit() for ch in s)

        def perturb_string(s: str, strength: int = 20) -> str:
            """
            Applies either:
            - 1–2 swaps of adjacent non-digit characters, OR
            - insertion of exactly one Unicode noise character near non-digit chars.

            Digits remain in place and are never swapped.
            """
            if not s:
                # empty string defaults just to a unicode noise char as the whole cell
                return self.rng.choice(unicode_noise_chars)

            non_digit_positions = [i for i, ch in enumerate(s) if not ch.isdigit()]

            do_swap = self.rng.random() < 0.5 and len(non_digit_positions) > 1

            if do_swap:
                # nearby swaps on non-digit characters only
                chars = list(s)
                num_swaps = self.rng.randint(1, 2)
                for _ in range(num_swaps):
                    # find candidate adjacent non-digit pairs
                    pairs = [
                        (i, i + 1)
                        for i in range(len(chars) - 1)
                        if (i in non_digit_positions) and (i + 1 in non_digit_positions)
                    ]
                    if not pairs:
                        break
                    #i, j = random.choice(pairs)
                    chosen_pairs = self.rng.sample(pairs, int(len(pairs)*strength/100))
                    for i, j in chosen_pairs:
                        chars[i], chars[j] = chars[j], chars[i]
                return "".join(chars)
            else:
                # insert exactly one unicode noise char near non-digit chars
                #noise = random.choice(unicode_noise_chars)
                n = len(s)

                # candidate insertion positions where at all the neighbors are non-digits
                insertion_positions = []
                for pos in range(n + 1):
                    left_non_digit = pos > 0 and not s[pos - 1].isdigit()
                    right_non_digit = pos < n and not s[pos].isdigit()
                    if left_non_digit and right_non_digit:
                        insertion_positions.append(pos)

                if not insertion_positions:
                    # fallback: insert at the end
                    #pos = n
                    pos = [n]
                else:
                    #pos = random.choice(insertion_positions)
                    pos = self.rng.sample(insertion_positions, max(1, int(len(insertion_positions)*strength/100)))

                noises = self.rng.choices(unicode_noise_chars, k=len(pos))
                pos = sorted(pos)[::-1]
                for i in range(len(noises)):
                    s = s[:pos[i]] + noises[i] + s[pos[i]:]

                return s

        def perturb_label(label, strength: int = 20):
            """
            Perturb an index/column label while preserving MultiIndex structure:
            - If label is a tuple: perturb one textual element.
            - Otherwise: perturb the label if it has textual chars.
            """
            if isinstance(label, tuple):
                elems = list(label)
                # identify textual elements
                text_positions = [
                    k for k, e in enumerate(elems)
                    if isinstance(e, str) and has_textual_chars(e)
                ]
                # SIDE-EFFECT: all numeric cells (like years) will never have typos inserted (which is fine)
                # but neither unicode characters (which wouldn't cause any troubles if inserted at the start or end of the cell)
                if not text_positions:
                    return label
                k = self.rng.choice(text_positions)
                elem_str = str(elems[k])
                elems[k] = perturb_string(elem_str, strength)
                return tuple(elems)

            # Single-level label
            if isinstance(label, str) and has_textual_chars(label):
                return perturb_string(label, strength)
            return label

        # build candidate slots: data cells + row index + column labels
        candidates = []

        n_rows, n_cols = df.shape

        # data cells
        for i in range(n_rows):
            for j in range(n_cols):
                val = df.iat[i, j]
                if pd.isna(val):
                    continue
                s = str(val)
                if has_textual_chars(s):
                    candidates.append(("data", i, j))

        # row indices
        index_labels = list(df.index)
        for i, lbl in enumerate(index_labels):
            if isinstance(lbl, tuple):
                if any(isinstance(e, str) and has_textual_chars(e) for e in lbl):
                    candidates.append(("row_index", i))
            else:
                if isinstance(lbl, str) and has_textual_chars(lbl):
                    candidates.append(("row_index", i))

        # columns
        column_labels = list(df.columns)
        for j, lbl in enumerate(column_labels):
            if isinstance(lbl, tuple):
                if any(isinstance(e, str) and has_textual_chars(e) for e in lbl):
                    candidates.append(("col_index", j))
            else:
                if isinstance(lbl, str) and has_textual_chars(lbl):
                    candidates.append(("col_index", j))

        if not candidates:
            return df

        # choosing ~20% of all candidate slots
        num_to_modify = max(1, int(len(candidates) * (strength / 100)))  # 1 out of 5
        num_to_modify = min(num_to_modify, len(candidates))
        chosen_slots = self.rng.sample(candidates, k=num_to_modify)

        new_index_labels = list(df.index)
        new_column_labels = list(df.columns)

        # applying perturbations
        for slot in chosen_slots:
            kind = slot[0]

            if kind == "data":
                _, i, j = slot
                val = df.iat[i, j]
                s = str(val)
                if has_textual_chars(s):
                    df.iat[i, j] = perturb_string(s, strength)
            elif kind == "row_index":
                _, i = slot
                lbl = new_index_labels[i]
                new_index_labels[i] = perturb_label(lbl, strength)
            elif kind == "col_index":
                _, j = slot
                lbl = new_column_labels[j]
                new_column_labels[j] = perturb_label(lbl, strength)

        # reassigning indices / columns
        if isinstance(df.index, pd.MultiIndex):
            df.index = pd.MultiIndex.from_tuples(
                new_index_labels, names=df.index.names
            )
        else:
            df.index = pd.Index(new_index_labels, name=df.index.name)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples(
                new_column_labels, names=df.columns.names
            )
        else:
            df.columns = pd.Index(new_column_labels, name=df.columns.name)

        return df

class IntraAmbiguousPerturber(Perturber):
    def __init__(self):
        super(Perturber, self).__init__()
        self.perturbations = [self.entity_missing, self.intra_table_contradiction]

    def entity_missing(self, table: pd.DataFrame, full_mask: pd.Series):
        """
        removes some entities from the table, making it impossible to answer the questions without ambiguity
        """

        true_idx = full_mask[full_mask].index
        if len(true_idx) == 0:
            return table, full_mask # no true values, nothing to remove

        drop_idx = self.rng.choice(true_idx) #np.random.choice(true_idx)

        table = table.drop(index=drop_idx)
        full_mask = full_mask.drop(index=drop_idx)

        return table, full_mask

    def intra_table_contradiction(self, table: pd.DataFrame, value_col: str, random_state: int = 42):
        s = table[value_col]

        # random increments for each row
        rng = np.random.default_rng(random_state)
        inc = pd.Series(rng.randint(1, 11, size=len(s)), index=s.index)

        out = pd.Series(pd.NA, index=s.index, dtype="string")
        mask = s.notna()
        out.loc[mask] = s.loc[mask].astype(str) + " (" + (s.loc[mask] + inc.loc[mask]).astype(str) + ")"

        table[value_col] = out
        return table


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import random

    random.seed(1)
    np.random.seed(1)

    p = Perturber()

    # --- bigger sample relational table ---
    n = 40
    df = pd.DataFrame(
        {
            "id": [f"r{i:03d}" for i in range(1, n + 1)],
            "country": np.random.choice(["Italy", "France", "Germany", "Spain", "Portugal"], size=n),
            "product": np.random.choice(["Widget", "Gadget", "Doohickey", "Thingamajig"], size=n),
            "year": np.random.choice(["2021", "2022", "2023", "2024"], size=n),
            "channel": np.random.choice(["Online", "Retail", "Wholesale"], size=n),
            "segment": np.random.choice(["Consumer", "SMB", "Enterprise"], size=n),
            "sales": np.random.randint(50, 500, size=n),
        }
    )

    value_col = "sales"
    id_col = "id"

    # table_types must align with df.columns
    table_types = [
        "categorical",  # id
        "categorical",  # country
        "categorical",  # product
        "categorical",  # year (treated as categorical for pivoting)
        "categorical",  # channel
        "categorical",  # segment
        "numerical",    # sales
    ]

    # any boolean mask is fine; keep it simple
    full_mask = pd.Series(False, index=df.index)
    chosen = np.random.choice(df.index.to_numpy(), size=3, replace=False)
    full_mask.loc[chosen] = True

    print("\nORIGINAL\n", df)

    # --- 1) pre_hct_perturbations ---
    for fn in p.pre_hct_perturbations:
        if fn.__name__ == "null_perturbation":
            constraints = {"country": "Italy"}  # simple constraint
            df = fn(df, constraints=constraints, value_col=value_col)
        else:
            df = fn(df)
        print(f"\nAFTER pre_hct -> {fn.__name__}\n", df)

    # --- 2) multiheader_perturbation ---
    df = p.multiheader_perturbation(
        df,
        value_col=value_col,
        id_col=id_col,
        table_types=table_types.copy(),  # function mutates the list
        aggr="first",
        unit_in_cell=False,
        full_mask=full_mask,
    )
    print("\nAFTER multiheader_perturbation\n", df)

    # --- 3) post_hct_perturbations ---
    for fn in p.post_hct_perturbations:
        df = fn(df)
        print(f"\nAFTER post_hct -> {fn.__name__}\n", df)