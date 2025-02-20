
import io
import contextlib
import re
import func_timeout
import sys

def refine_code(response: str):
    """Refine the code.

    Repair
    ```
    Step x: from xxxx import xxx
    ```
    to
    ```
    Step x:
    from xxxx import xxx
    ```

    Args:
        response (str): _description_

    Returns:
        _type_: _description_
    """
    pattern = r"(Step \d+: )from[\w\s,]*import"
    replacement = r"\1\nfrom"
    result = re.sub(pattern, replacement, response)
    return result


def parse_code(response:str, with_tag:bool=False):
    """Parse python code from the response of Coding/Math_PoT task.

    Args:
        response (str): The entire response string.
        with_tag (bool): Whether the returned string contains the format of code block ("```python" and "```")
    Returns:
        str: the python code.
    """
    response = refine_code(response)
    response = r"{}".format(response)
    pattern = r"```python([\s\S]+?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        answer_str = match.group(0) if with_tag else match.group(1)
    else:
        answer_str = "Not Found"
    return answer_str


def execute(code:str, std_input:str=""):
    """Execute the input code string in python.
    * If there is a std output, capture the output.
        (```print("xxxx")```)
    * If there is no std output, get the variable in the last line.
        (```result = a + b``` or ```result```, mostly in jupter notebook mode.)

    Args:
        code (str): The code to be executed

    Returns:
        str: str(executed_result)
    """
    output = io.StringIO()
    input_stream = io.StringIO(std_input)

    original_stdin = sys.stdin
    sys.stdin = input_stream

    try:
        with contextlib.redirect_stdout(output):
            exec_namespace = {}
            local_variables = {}
            exec(code, exec_namespace, local_variables)
    finally:
        sys.stdin = original_stdin
    captured_output = output.getvalue()
    if captured_output == "":
        target = code.strip().split('\n')[-1]
        if target == "```" or target == "":
            target = code.strip().split('\n')[-2]
        if '=' in target:
            target = target.split('=')[0].strip()
        captured_output = local_variables.get(target)
    return str(captured_output).strip()


def safe_execute(code_string:str, std_input_string):
    code_string = parse_code(code_string)
    try:
        ans = func_timeout.func_timeout(60, execute, args=(code_string,std_input_string))
    except func_timeout.FunctionTimedOut:
        ans = "Error: Timeout"
    except Exception as e:
        ans = "Error: " + str(e)
    return ans

def parse_and_execute(response_content: str, target: str=""):
    """Get the execution result of the code.
    All codes in UltraInteract Dataset uses `print()` to output the result.
    So we can capture the output by redirecting the stdout.
    Args:
        response_content (str): The total response_content.
    """

    code_string = parse_code(response_content)
    if "<code>" in target:
        target = target.replace("<code>", "").replace("</code>", "")
    ans = safe_execute(code_string)

    return ans