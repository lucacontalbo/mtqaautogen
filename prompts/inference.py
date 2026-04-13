prompt = """Answer the following question given the provided HTML table.
First reason step-by-step, then write "Final answer:" followed exclusively by the correct answer. Do not write anything else after "Final answer:"
Every calculation must be done with a precision of exactly 6 decimal places.
Only the numerical value must be written in the final answer, except for comparative questions, where the final answer must be "yes" or "no".

Question: {question}
Table:
{table}

Let's think step-by-step. """