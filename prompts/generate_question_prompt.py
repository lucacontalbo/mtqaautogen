prompt = """Given an html table, an SQL query and the final result of the query, you must generate a natural language question.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
Make sure to include, in the natural language question, all the information and constraints appearing inside the SQL query.
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

Table:
{table}

SQL Query: {query}
Result: {result}

Let's think step-by-step."""

prompt_superlative = """Given an html table, an SQL query and the final result of the query, you must generate a natural language question.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
Make sure to include, in the natural language question, all the information and constraints appearing inside the SQL query.
For superlative questions, make sure to ask for the value, instead of asking for the entity.
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

Table:
{table}

SQL Query: {query}
Result: {result}

Let's think step-by-step."""

prompt_percentage_change = """Given an html table, an SQL query and the final result of the query, you must generate a natural language question.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
Make sure to include, in the natural language question, all the information and constraints appearing inside the SQL query.
For percentage change questions, frame the questions on the "percentage increase".
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

Table:
{table}

SQL Query: {query}
Result: {result}

Let's think step-by-step."""

prompt_comparison = """Given an html table, an SQL query and the final result of the query, you must generate a natural language question.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
Make sure to include, in the natural language question, all the information and constraints appearing inside the SQL query.
For comparison questions, frame the questions as "is the [...] greater/lower than [...]?", not as "between which values is [...]?", since the answer must be "yes" or "no".
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

Table:
{table}

SQL Query: {query}
Result: {result}

Let's think step-by-step."""


prompt_multi = """Given a list of SQL queries, a list of HTML tables, a final aggregation operator across the tables, and the final result, you must generate a natural language question.
In particular, to obtain the final result, each SQL query is executed on its corresponding table, and then the aggregation operator is applied to the intermediate results.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

{text}

The generated question must be of the following type: {method}
For superlative questions, make sure to ask for the value. Never ask for the entity.
The final aggregation operation, that is applied to the extracted values from each table, is the following: {aggregation}
This is the final result: {result}

Make sure that the generated question also explicits the final aggregation operation applied over the extracted values.
The question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Also, the natural language question must not miss any crucial information needed to answer the question.

Let's think step-by-step."""

prompt_multi_unit_variation = """Given a list of SQL queries, a list of HTML tables, a final aggregation operator across the tables, an unit of measurement and the final result, you must generate a natural language question.
In particular, to obtain the final result, each SQL query is executed on its corresponding table, the values must be normalized to the requested unit of measurement, and then the aggregation operator is applied to the intermediate results.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

{text}

The generated question must be of the following type: {method}
For superlative questions, make sure to ask for the value. Never ask for the entity.
The final aggregation operation, that is applied to the extracted values from each table, is the following: {aggregation}
This is the final result: {result}

Make sure that the generated question also explicits the final aggregation operation applied over the extracted values.
Make sure that the final natural language question always specifies the following unit of measurement: {unit}
The question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Also, the natural language question must not miss any crucial information needed to answer the question.

Let's think step-by-step."""

prompt_multi_fk = """Given a list of SQL queries, a list of HTML tables, and the final result, you must generate a single natural language question.
In particular, to obtain the final result, each SQL query is executed on its corresponding table. The response of intermediate queries is subsequently used to extract information on the next table, until the final table (and final result) is reached.
The natural language question must be answerable by the same result. However, you must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
The natural language question must not make clear what are the results of the intermediate steps. For example, if a value extracted from the first table and first sql query is needed to extract another value from the second table (with the second sql query) and so on, you must not indicate what is the intermediate value extracted from table 1.
Instead, you must write a question that asks the user to get the needed information without telling what are the intermediate results.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.
First reason step-by-step. Then, write exactly "Final question:" followed exclusively by the novel natural language question. Make sure to include all the information needed to answer correctly the question, so that the question is still answerable with the original SQL result.
Do not write anything else after "Final question:".

{text}

The generated question must be of the following type: {method}
For superlative questions, make sure to ask for the value. Never ask for the entity.
All the tables are needed to answer the question. In specific, the question must merge the different SQL queries into a unique natural language question that is answered by the final result.
This is the final result: {result}

The question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Make sure that the generated question comprehensively includes every step in the chain of queries.

Let's think step-by-step."""

prompt_multi_fk = """You will be given a question in natural language, some SQL extraction queries that are applied sequentially to each table, some tables and the result of executing those SQL queries.
First, translate each SQL query into a natural language question. Make sure not to lose any important details.
Second, for each question other than the first, identify the component that depends on the answer of the previous question. Usually it is the column used in the SELECT statement of the previous query.
Third, translate each question (except the last one) into a natural language instruction. All the dependent components must be substituted with a text indicating the previous instruction / question (e.g. "Considering the previous result, ..." or something similar). It is very important that the dependent components (parts of current instruction depending on previous instructions) are not explicitly mentioning the intermediate results, but they are just referring to the previous instruction in a generic way. This is to make sure that the final question does not make clear what are the results of the intermediate steps.
Finally, concatenate all the instructions with the last question (make sure that at the end we have a question, corresponding to the final SQL query, and can be answered by the same result). After all the previous "Considering...", introduce the final question with "Finally" or similar words.
Make sure not to lose any important details.

You must differ how the sentence is phrased and the words used, in order to avoid too much overlap between question and table contents.
You must not use the exact SQL table attributes, where the attribute name is in camel case, but you must use them in a more natural way, resembling user questions.

In the end, write "Final question:" followed exclusively by the final natural language instructions + question. Do not write anything else after "Final question:".

{text}

The generated question must be of the following type: {method}
For superlative questions, make sure to ask for the value. Never ask for the entity.

This is the final result: {result}

The question must be user-like: it must not contain SQL syntax, camel case, mathematical or boolean symbols, or explicit mentions of certain multi-word string values (appearing in the table) enclosed in "" (slight variations are fine).
Make sure that the generated question comprehensively includes every step in the chain of queries.
Keep in mind, however, that it is very important that the dependent components (parts of current instruction depending on previous instructions) are not explicitly mentioning the intermediate results, but they are just referring to the previous instruction in a generic way. This is to make sure that the final question does not make clear what are the results of the intermediate steps.

Let's think step-by-step."""

# Make sure to include, in the natural language question, all the information and constraints appearing inside the SQL query.