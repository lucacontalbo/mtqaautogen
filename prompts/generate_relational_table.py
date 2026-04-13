prompt = """Define the name/topic for a relational table. Do not choose among the following already chosen names/topics:
{past}

After that, define the table attributes, the types of the attributes and the range of values allowed.
In particular, only one column can contain float values. The other columns must contain categorical values.
The number of attributes (number of columns) must be exactly equal to {num_columns}.
The table must be pivoted later, so create a table that can tolerate pivot operations and still be meaningful. Make the table "real", using real entities while avoiding placeholders.

First reason step-by-step. Then write "Final answer: " followed exclusively by a Python dictionary exactly like the following:

{{
    "name": ..., # name of the table
    "attributes": ..., # list containing the names of the attributes. The names must not have whitespaces.
    "attribute_types": ..., # list containing the types of the attributes. the length of this list must be equal to the length of the "attributes" list. The type can either be "float", "int" or "categorical". Also numerical values (like years) can be categorical. The ID column must contain unique values and must be numerical
    "range": ..., # this list must have the same length as "attributes" and "attribute_types". for categorical attributes it is a list containing all the possible values. For possibly float and int values it is a list containing, at the first position, the start of the range and at the second position the end of the range.
    "value_col": ..., # string indicating the attribute name of the column to pivot later. This attribute must either be "int" or "float"
    "id_col": ..., # string indicating the attribute name of the id of the table
}}

Ensure the output is in the expected format. Do not write anything else after "Final answer: ".

Let's think step-by-step."""

prompt_domain = """You are the best table designer in the world for the {domain} topic. You always use lexicon highly specific to {domain}.
For the tables you create, you always make the tables "real", using real entities while avoiding placeholders. You always use lexicon that is specific to {domain}, and make textual entries in the tables be composed even by more words (unlike standard relational tables), like in the following examples (you must come up with different lexicon than the one used in the examples):

{examples}

You must use variations of the terminology used previously, but it's important you avoid using highly common terms. Use terms that only a real {domain} expert would know.
Then, create the following Python dictionary, where the number of attributes (number of columns) must be exactly equal to {num_columns}.

{{
    "name": ..., # name of the table, must be in pascal casing
    "attributes": ..., # list containing the names of the attributes. The names must be written in pascal casing. They must not have whitespaces
    "attributes_long": ..., # list containing the names of the attributes. This list must be exactly the same as the "attributes" list, but the attribute names must not be in camel casing, pascal casing or snake casing. They may contain consist of multiple words. Use {domain}-specific lexicon, as the one used in the "examples" below, but be creative and vary the lexicon. Not every word must have the initial letter in uppercase.
    "attribute_types": ..., # list containing the types of the attributes. the length of this list must be equal to the length of the "attributes" list. The type must be "categorical", or either "int" or "float" for the "value_col" column. Also numerical values (like years or IDs) can be categorical. Only the value_col column can be float.
    "range": ..., # this list must have the same length as "attributes" and "attribute_types". For categorical attributes it is a list containing {col_cardinality} different values, which can be lengthy like in standard web tables. Use {domain}-specific lexicon like the one used in the examples below, but be creative and vary the lexicon. For possible float and int values it is a list containing, at the first position, the start of the range and at the second position the end of the range (extremes included).
    "value_col": ..., # string indicating the attribute name of the column to pivot later. The name of the attribute must be one of the names in "attributes". This table attribute must contain values that are either "int" or "float". 
    "id_col": ..., # string indicating the attribute name of the id of the table. The name of the attribute must be one of the names in "attributes". The attribute must be only one, and not multiple.
    "unit_of_measurement": ..., # list of one string indicating the unit of measurement for each element in the "value_col" column. You can use symbols, abbreviations or full words to indicate the unit of measurement.
    "number_of_decimals": ... # dictionary that, for each unit of measurement in the "unit_of_measurement" list, indicates the number of decimals (integer greater or equal than 0) to use when representing values in the "value_col" column. If the unit of measurement is "None", do not include it in this dictionary. The unit of measurement strings must be indicated exactly like the ones inside "units_of_measurement" list
}}

The generated table must NOT use the following attributes and values:
Attributes of past tables: {past}
Corresponding values of past tables: {past_values}

First reason step-by-step. Then write exactly "Final answer: " (the text must be exactly the same) followed exclusively by the requested Python dictionary.
To avoid numerical inconsistencies, you must not specify any type of unit of measurement inside the table ("name", "attributes", "attributes_long", "attribute_types", "range", "value_col", "id_col"), but only inside the "unit_of_measurement" field. This is very important.
Ensure the output is in the expected format. Make sure that the proposed table is about {domain} and at the same time does uses completely different attributes and values as the tables used previously.
Make sure that the table values ("range") do not include any specification about units of measurement, as the unit of measurement must be specified only in the "unit_of_measurement" field. This is very important to avoid numerical inconsistencies.
Make sure to write exactly "Final answer: " at the end (the text, together with ":", must be exactly and completely the same), followed by the required dictionary. Do not write anything else after "Final answer: ".

Let's think step-by-step."""

prompt_domain_unit_specific = """You are the best table designer in the world for the {domain} topic. You always use lexicon highly specific to {domain}.
For the tables you create, you always make the tables "real", using real entities while avoiding placeholders. You always use lexicon that is specific to {domain}, and make textual entries in the tables be composed even by more words (unlike standard relational tables), like in the following examples (you must come up with different lexicon than the one used in the examples):

{examples}

You must use variations of the terminology used previously, but it's important you avoid using highly common terms. Use terms that only a real {domain} expert would know.
Then, create the following Python dictionary, where the number of attributes (number of columns) must be exactly equal to {num_columns}.

{{
    "name": ..., # name of the table, must be in pascal casing
    "attributes": ..., # list containing the names of the attributes. The names must be written in pascal casing. They must not have whitespaces
    "attributes_long": ..., # list containing the names of the attributes. This list must be exactly the same as the "attributes" list, but the attribute names must not be in camel casing, pascal casing or snake casing. They may contain consist of multiple words. Use {domain}-specific lexicon, as the one used in the "examples" below, but be creative and vary the lexicon. Not every word must have the initial letter in uppercase.
    "attribute_types": ..., # list containing the types of the attributes. the length of this list must be equal to the length of the "attributes" list. The type must be "categorical", or either "int" or "float" for the "value_col" column. Also numerical values (like years or IDs) can be categorical. Only the value_col column can be float.
    "range": ..., # this list must have the same length as "attributes" and "attribute_types". For categorical attributes it is a list containing {col_cardinality} different values, which can be lengthy like in standard web tables. Use {domain}-specific lexicon like the one used in the examples below, but be creative and vary the lexicon. For possible float and int values it is a list containing, at the first position, the start of the range and at the second position the end of the range (extremes included).
    "value_col": ..., # string indicating the attribute name of the column to pivot later. The name of the attribute must be one of the names in "attributes". This table attribute must contain values that are either "int" or "float". 
    "id_col": ..., # string indicating the attribute name of the id of the table. The name of the attribute must be one of the names in "attributes". The attribute must be only one, and not multiple.
    "unit_of_measurement": ..., # list of one string, where the unit must be one of the following units: {units}
    "number_of_decimals": ... # dictionary that, for each unit of measurement in the "unit_of_measurement" list, indicates the number of decimals (integer greater or equal than 0) to use when representing values in the "value_col" column. If the unit of measurement is "None", do not include it in this dictionary. The unit of measurement strings must be indicated exactly like the ones inside "units_of_measurement" list
}}

The generated table must NOT use the following attributes and values:
Attributes of past tables: {past}
Corresponding values of past tables: {past_values}

First reason step-by-step. Then write exactly "Final answer: " (the text must be exactly the same) followed exclusively by the requested Python dictionary.
To avoid numerical inconsistencies, you must not specify any type of unit of measurement inside the table ("name", "attributes", "attributes_long", "attribute_types", "range", "value_col", "id_col"), but only inside the "unit_of_measurement" field. This is very important.
Ensure the output is in the expected format. Make sure that the proposed table is about {domain} and at the same time does uses completely different attributes and values as the tables used previously.
Make sure that the table values ("range") do not include any specification about units of measurement, as the unit of measurement must be specified only in the "unit_of_measurement" field. This is very important to avoid numerical inconsistencies.
Make sure to write exactly "Final answer: " at the end (the text, together with ":", must be exactly and completely the same), followed by the required dictionary. Do not write anything else after "Final answer: ".

Let's think step-by-step."""