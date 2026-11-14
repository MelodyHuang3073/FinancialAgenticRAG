"""Prompts for the financial-filing task of Google TableRAG.

Google's repo has prompts for tabfact, wtq, arcade and bird only. These are the wtq prompts (the closest task:
short factual answers from one table) with two additions, and nothing else changed:

1. a short description of the long table (columns, one row per line item and period, unit column, parentheses are
   negative), placed before the format rules;
2. one note asking for the answer in the unit, rounding and form the question asks for.

The reply format ("Final Answer: ...", one line) is kept, so answers stay short: questions that need an explanation
are answered with whatever the table can support. This is listed as a deviation in the baseline's README.
"""

TABLE_DESCRIPTION = (
    "The dataframe holds every financial-table cell of one company filing (10-K, 10-Q or 8-K), one row per line item "
    "and period. Columns: page (page number), line_item (row label as printed), period (a year or column label), "
    "value (the cell as printed), number (the cell parsed as a number; parentheses mean negative), unit (the unit "
    "of that page: millions, thousands or billions; empty if unknown). The same line item can appear on several "
    "pages, for example in a summary table and in the financial statements."
)

extract_column_prompt = '''
Given a large table, I want to answer a question: {query}
Since I cannot view the table directly, please suggest some column names that might contain the necessary data to answer this question.
Please answer with a list of column names in JSON format without any additional explanation.
Example:
["column1", "column2", "column3"]
'''

extract_cell_prompt = '''
Given a large table, I want to answer a question: {query}
Please extract some keywords which might appear in the table cells and help answer the question.
The keywords should be categorical values rather than numerical values.
The keywords should be contained in the question.
Please answer with a list of keywords in JSON format without any additional explanation.
Example:
["keyword1", "keyword2", "keyword3"]
'''

solve_table_prompt = '''
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}

''' + TABLE_DESCRIPTION + '''

Tool description:
- `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.

Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.

{schema_retrieval_result}

{cell_retrieval_result}

Strictly follow the given format to respond:
Thought: you should always think about what to do
Action: the single line Python command to execute
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations
Final Answer: the final answer to the original input question (Answer1, Answer2, ...)

Notes:
- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: Answer1, Answer2, ..." form, no other form.
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.
- Give the final answer in the unit, rounding and form the question asks for.

Now, please use ``python_repl_ast` with the column names and cell values above to answer the question: {query}

Begin!
'''


def register(prompts_module):
    """Add task 'fin' to Google's prompt lookup (its get_prompt reads this function at call time)."""
    original = prompts_module.get_prompt_templates

    def get_prompt_templates(task, agent_type):
        if task == "fin" and "TableRAG" in agent_type:
            return {
                "extract_column_prompt": extract_column_prompt,
                "extract_cell_prompt": extract_cell_prompt,
                "solve_table_prompt": solve_table_prompt,
            }
        return original(task, agent_type)

    prompts_module.get_prompt_templates = get_prompt_templates
