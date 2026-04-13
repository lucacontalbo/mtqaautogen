prompt = """You are the best analyst in the world for the {domain} topic. You always use lexicon highly specific to {domain}.
Given the schema of a table and other additional metadata, you must indicate if there are any pairs of values that break the semantic coherence of the table.
In particular, you will be provided with a table and metadata like the following:

{{
    "name": ..., # name of the table, must be in pascal casing
    "attributes": ..., # list containing the names of the attributes. The names must be written in pascal casing. They must not have whitespaces
    "attributes_long": ..., # list containing the names of the attributes. This list must be exactly the same as the "attributes" list, but the attribute names must not be in camel casing, pascal casing or snake casing. They may contain consist of multiple words. Use {domain}-specific lexicon, as the one used in the "examples" below, but be creative and vary the lexicon. Not every word must have the initial letter in uppercase.
    "attribute_types": ..., # list containing the types of the attributes. the length of this list must be equal to the length of the "attributes" list. The type must be "categorical", or either "int" or "float" for the "value_col" column. Also numerical values (like years or IDs) can be categorical.
    "range": ..., # this list must have the same length as "attributes" and "attribute_types". For categorical attributes it is a list containing all the possible values, which can be lengthy like in standard web tables. Use {domain}-specific lexicon like the one used in the examples below, but be creative and vary the lexicon. For possible float and int values it is a list containing, at the first position, the start of the range and at the second position the end of the range.
    "value_col": ..., # string indicating the attribute name of the column to pivot later. The name of the attribute must be one of the names in "attributes". This table attribute must contain values that are either "int" or "float". 
    "id_col": ..., # string indicating the attribute name of the id of the table. The name of the attribute must be one of the names in "attributes". The attribute must be only one, and not multiple.
    "unit_of_measurement": ..., # list of strings indicating the unit of measurement for each element in the "value_col" column. If the "value_col" is categorical, or if a single element in the "value_col" does not expect a unit of measurement, write "None" as unit of measurement. You can use symbols, abbreviations or full words to indicate the unit of measurement.
    "number_of_decimals": ... # dictionary that, for each unit of measurement in the "unit_of_measurement" list, indicates the number of decimals (integer greater or equal than 0) to use when representing values in the "value_col" column. If the unit of measurement is "None", do not include it in this dictionary. The unit of measurement strings must be indicated exactly like the ones inside "units_of_measurement" list
}}

Look at the range of values for non-float attributes and reason step-by-step about which pairs of values would break the semantic coherence of the table.
In the end, write exactly "Final answer: " (the text must be exactly the same) followed exclusively by a Python dictionary, where the keys are tuples containing the names of the attributes that contain the problematic values, and the values are lists of tuples with the problematic pairs of values for each attribute.
If there are no such pairs, answer "Final answer: {{}}".

Ensure the output is in the expected format.
Make sure to write exactly "Final answer: " at the end (the text, together with ":", must be exactly and completely the same), followed by the required dictionary. Do not write anything else after "Final answer: ".

{input}

Let's think step-by-step."""