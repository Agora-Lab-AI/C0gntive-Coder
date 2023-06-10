import subprocess
import os
import subprocess
import sys

from typing import List, Dict



# Read the system prompt
with open(os.path.join(os.path.dirname(__file__), "..", "prompt.txt"), "r") as f:
    SYSTEM_PROMPT = f.read()
    
def run_script(script_name: str, script_args: List) -> str:
    """
    If script_name.endswith(".py") then run with python
    else run with node
    """
    script_args = [str(arg) for arg in script_args]
    subprocess_args = (
        [sys.executable, script_name, *script_args]
        if script_name.endswith(".py")
        else ["node", script_name, *script_args]
    )

    try:
        result = subprocess.check_output(subprocess_args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        return error.output.decode("utf-8"), error.returncode
    return result.decode("utf-8"), 0
