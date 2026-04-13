constraint_prompt = """You are an operational research expert operating in the {domain} domain. Your task is to formulate optimization problems with specific constraints.
In particular, you will be given some table metadata in the following format:

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

Using the provided metadata, you must use your knowledge to determine the constraints between different attributes ("attributes" list) and/or different values ("range" list).
In particular:
1. for inter-row constraints, you must always identify the relation between different rows based on the "value_col" attribute, while matching the rows by using the attribute and respective values in the "range" list.
2. for intra-row constraints, you must always identify the relation between different attributes in the same row

To select specific rows, use the notation "[attribute].[value]" to indicate the attribute and the specific value from the "range" list.
If you want to select rows based on multiple attributes, use the notation "([attribute1].[value1] & [attribute2].[value2])".
For example, to select all rows where the "Company" attribute is equal to "Ferrari", use "Company.Ferrari". To select all rows where the "Company" attribute is equal to "Ferrari" and the "Profit" attribute is equal to "Gross", use "(Company.Ferrari & Profit.Gross)".

To write the rules, you can use all mathematical and boolean operators, as long as they are used in Python.
When you create the inter-row constraints, you are always comparing the "value_col" attribute of different rows, for example:
1. "(Company.Ferrari & Year.2020 & TypeOfMoney.Gross).Amount > (Company.Ferrari & Year.2020 & TypeOfMoney.Net).Amount", since gross profit is always higher than net profit for the same company in the same year.
2. "(Company.Ferrari & Year.2020 & TypeOfMoney.Gross).Amount = (Company.Ferrari & Year.2020 & TypeOfMoney.Total).Amount - (Company.Ferrari & Year.2020 & TypeOfMoney.COGS).Amount", since gross profit is obtained by subtracting cost of goods sold from total revenue for the same company and year.

When you create the intra-row constraints, you are always comparing different attributes in the same row, for example:
1. "if CompanyTier.Small then (Amount <= 5000000)", since small-tier companies cannot exceed 5M in the reported monetary measure.
2. "if TypeOfMoney.COGS then not Scenario.Forecast", since cost of goods sold is only reported for realized (historical) scenarios, not for forecasts.

First reason step-by-step. Then write exactly "Final answer:" followed by a Python dictionary in the following format:
{
    "inter_row_constraints": ..., # list of strings, each string is an inter-row constraint
    "intra_row_constraints": ...  # list of strings, each string is an intra-row constraint
}

Let's think step-by-step.
"""

constraint_prompt_gpt = """You are an Operations Research expert in the {domain} domain. Given table metadata, infer realistic constraints among attributes and values.

## Input
You will receive a JSON-like object:
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

## Task
Infer constraints using domain knowledge. Produce:
- inter-row constraints: relationships between *different rows*, always comparing the numeric value in `value_col`
- intra-row constraints: relationships between *attributes within the same row*
Do not create constraints that are already explicitly defined by the table structure (e.g., bounds in the range list, unit of measurement, number of decimals...).

## Row selection syntax
- Select a row by fixing categorical values:  Attribute == "Value" (the operator can be a python boolean operator)
- Combine selectors: (Attribute1 == "Value1" and Attribute2 == "Value2" and ...)
- Refer to the numeric value using the `value_col` name as a property:
  ( ...selectors... ).{{value_col}} 
  where {{value_col}} is the placeholder containing the value_col attribute name, not the string "value_col", and must not be used with the {{}} (which are part of the placeholder)

Example selectors:
- (Company == "Ferrari")
- (Year == 2020 and TypeOfMoney == "Gross")
String values expect to be enclosed within \"\", while numerical data must not.

## Constraint language
Write each rule as a Python-valid boolean/math expression (>, >=, ==, !=, +, -, *, /, and, or, not, parentheses).
- Inter-row constraints MUST compare `(...selector...).{{value_col}}` between two (or more) selected rows. The selector must be surrounded by brackets. Across each selected row, all non-specified attributes are considered to have the same value.
- Intra-row constraints MUST compare attributes in the same row and must use conditional form. The selector and expression must be surrounded by brackets:
  "if (...selector...) then (<expression>)"

Examples (inter-row):
- (TypeOfMoney == "Gross").{{value_col}} > (TypeOfMoney == "Net").{{value_col}}
- (Year == 2020 and TypeOfMoney == "BudgetYearBeginning").{{value_col}} == (Year == 2019 and TypeOfMoney == "BudgetYearEnd").{{value_col}}
    (this applies to the same Company in both rows, as Company is not specified in the selectors)

Examples (intra-row):
- if (CompanyTier == "Small") then ({{value_col}} <= 5000000)
- if (TypeOfMoney == "COGS" or Scenario == "Actual") then (Scenario != "Forecast")
The expression (e.g. Scenario != "Forecast") cannot contain "and", "or" or "not" operators (for the "and", multiple rules must be created, instead of using "and")
When you have lhs or rhs with mathematical operations (not boolean operations), enclose the lhs or rhs with "()" parentheses.

## Output format (strict)
First, reason step-by-step. Then output exactly "Final answer:" followed by a Python dictionary in this format:
{{
  "inter_row_constraints": [<string>, ...], # the list of strings can be empty if no inter-row constraints are found
  "intra_row_constraints": [<string>, ...] # the list of strings can be empty if no intra-row constraints are found
}}
Make sure that you use the exact attribute names and values as provided in the metadata, and that you exactly follow the constraint language and output format.
Remember that {{value_col}} is the placeholder containing the value_col attribute name, not the string "value_col", and must not be used with the {{}} (which are part of the placeholder)

Table info: {input}

Let's think step-by-step.
"""