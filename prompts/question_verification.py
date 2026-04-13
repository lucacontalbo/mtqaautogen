prompt = """You will be given a question in natural language, an SQL question, a database table and the result of executing that SQL query on the database table.
Your task is to verify if the question in natural language is equivalent to the SQL question and produces the same result.
You must check if the natural language question does not specify a needed constraint that the SQL question specifies.
First, reason step-by-step and analyze the natural language question and the SQL question to determine if they are asking for the same information.
If they are equivalent (use the same information), in the end write exactly "Final answer: Yes".
If they are not equivalent, in the end write exactly "Final answer:" followed exclusively by the novel formulation of the natural language question that would make it equivalent to the SQL question. Do not write anything else after "Final answer:".

Natural Language Question: {nl_question}
SQL Question: {sql_question}
Database Table: {table}
SQL Query Result: {sql_result}

Let's think step-by-step."""



prompt_multi = """You will be given a question in natural language, some SQL extraction queries, some tables and the result of executing those SQL queries + an aggregation operation (e.g. sum, average, max/min extraction or similar) on those tables.
Your task is to verify if the question in natural language is equivalent to the aggregation of the SQL extractive queries and produces the same result.
You must check if the natural language question does not specify a needed constraint that the SQL queries specifies.
Also, you must check if the natural language question does not specify the final aggregation operation to apply.
First, reason step-by-step and analyze the natural language question and the SQL queries + aggregation operation to determine if they are asking for the same information.
If they are equivalent (use the same information), in the end write exactly "Final answer: Yes".
If they are not equivalent, in the end write exactly "Final answer:" followed exclusively by the novel formulation of the natural language question that would make it equivalent to the SQL queries + aggregation operation.
In this case, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Do not write anything else after "Final answer:".

Natural Language Question: {nl_question}
SQL Question: {sql_question}
Tables: {table}
Result: {sql_result}

Remember that, if you change the question, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).

Let's think step-by-step."""

prompt_multi_unit_variation = """You will be given a question in natural language, some SQL extraction queries, some tables and the result of executing those SQL queries + an aggregation operation (e.g. sum, average, max/min extraction or similar) on those tables. Also, you will be provided with the reference unit of measurement to use inside the question.
Your task is to verify if the question in natural language is equivalent to the aggregation of the SQL extractive queries and produces the same result, considering also possible unit of measurement normalizations.
You must check if the natural language question does not specify a needed constraint that the SQL queries specifies.
Also, you must check if the natural language question does not specify the final aggregation operation to apply or the final unit of measurement the answer expects
First, reason step-by-step and analyze the natural language question and the SQL queries + aggregation operation + unit of measurement to determine if they are asking for the same information.
If they are equivalent (use the same information), in the end write exactly "Final answer: Yes".
If they are not equivalent, in the end write exactly "Final answer:" followed exclusively by the novel formulation of the natural language question that would make it equivalent to the SQL queries + aggregation operation + unit of measurement.
In this case, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
The final natural language question must always specify the following unit of measurement: {unit}
Do not write anything else after "Final answer:".

Natural Language Question: {nl_question}
SQL Question: {sql_question}
Tables: {table}
Result: {sql_result}

Remember that, if you change the question, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Also, the final natural language question must always specify the following unit of measurement: {unit}

Let's think step-by-step."""

prompt_multi_fk = """You will be given a question in natural language, some SQL extraction queries that are applied sequentially to each table, some tables and the result of executing those SQL queries.
Your task is to verify if the question in natural language is equivalent to the sequential application of the SQL extractive queries and produces the same final result.
You must check if the natural language question does not specify a needed constraint that the SQL queries specifies.
At the same time, the question must not explicitly state intermediate results, i.e. results that have been extracted from previous queries: it's up to who answers the question to resolve each intermediate step and get the needed results.
First, reason step-by-step and analyze the natural language question and the SQL queries to determine if they are asking for the same information.
If they are equivalent (use the same information), in the end write exactly "Final answer: Yes".
If they are not equivalent, in the end write exactly "Final answer:" followed exclusively by the novel formulation of the natural language question that would make it equivalent to the SQL queries. The natural language question must be one, not a concatenation of multiple questions.
In this case, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Do not write anything else after "Final answer:".

Natural Language Question: {nl_question}
SQL Question: {sql_question}
Tables: {table}
Result: {sql_result}

Remember that, if you change the question, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).

Let's think step-by-step."""

prompt_multi_fk = """You will be given a list of instructions + questions in natural language, some SQL extraction queries that are applied sequentially to each table, some tables and the result of executing those SQL queries.
The natural language question is created so that the user first retrieves information related a first subquestion, then uses that information (which MUST NOT be explicitly visible inside the question) to solve the next subquestion, and so on.
You must check if the natural language question does not specify a needed constraint that, without it, makes it impossible to correctly answer the question. Intermediate results must not be visible: for example, if the second instructions depends on the previous instruction (where the dependence is introduced by "Considering..."), you must not indicate the result of the previous instruction. It is good that next instructions use "Considering..." related to the previous instruction, as they do not explicitly indicate intermediate results to the user.
If there's a missing constraint (except for dependent ones, as said before), your MUST explicitly add those constraints and fix the overall question.
First, reason step-by-step and analyze the natural language instructions+question and the SQL queries to determine if they are asking for the same information.
If they are equivalent (use the same information), in the end write exactly "Final answer: Yes".
If they are not equivalent, in the end write exactly "Final answer:" followed exclusively by the novel formulation of the natural language instructions+question that would make it equivalent to the SQL queries.
In this case, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
You must make sure that the natural language instructions + question contain EVERY information needed to answer the question, as the user cannot read the SQL queries, so don't take anything for granted (except for the dependent results).
Do not write anything else after "Final answer:".

Natural Language Question: {nl_question}
SQL Question: {sql_question}
Tables: {table}
Result: {sql_result}

Remember that, if you change the question, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Also, remember that intermediate results MUST NOT be explicitly visible inside the question: for example, if the n-th instructions depends on the (n-1)-th instruction, you must not indicate the result of the previous instruction as a constraint of the next instruction. It is good that next instructions use "Considering..." related to the previous instruction, as they do not explicitly indicate intermediate results to the user.
The final question must always clearly explicitate what are the values that are needed to apply the aggregation operation in the end. It must not say "what is the value across the specified indicators", but say something among the lines of "what is the value across: (1) ..., (n) ...". This is important, as the user cannot see the SQL query, and everything needs to be explicitly stated.

Let's think step-by-step."""

prompt_multi_fk = """You will be given a question in natural language, two SQL queries that are applied sequentially, some tables and the result of executing those SQL queries.
You must check if the natural language question does not specify a needed constraint that, without it, makes it impossible to correctly answer the question.
If there's a missing constraint (except for dependent ones, like the result of the first SQL query, or the hidden joins), your MUST explicitly add those constraints and fix the overall question.
First, reason step-by-step and analyze the natural language question and the SQL queries to determine if they are asking for the same information.
If they are equivalent (use the same information), in the end write exactly "Final answer: Yes".
If they are not equivalent, in the end write exactly "Final answer:" followed exclusively by the novel formulation of the natural language instructions+question that would make it equivalent to the SQL queries.
In this case, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
You must make sure that the natural language instructions + question contain EVERY information needed to answer the question, as the user cannot read the SQL queries, so don't take anything for granted (except for the dependent results).
Do not write anything else after "Final answer:".

Natural Language Question: {nl_question}
SQL Question: {sql_question}
Tables: {table}
Result: {sql_result}

Remember that, if you change the question, the natural language question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Also, remember that intermediate results MUST NOT be explicitly visible inside the question.
The final question must always clearly explicitate what are the values that are needed to apply the aggregation operation in the end. It must not say "what is the value across the specified indicators", but say something among the lines of "what is the value across: (1) ..., (n) ...". This is important, as the user cannot see the SQL query, and everything needs to be explicitly stated.

Let's think step-by-step."""