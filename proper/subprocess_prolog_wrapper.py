import subprocess
import ast

def get_prolog_result(complete_filename, subprocess_filename = 'subprocess_prolog.py'):
    result = subprocess.run(['python', subprocess_filename, complete_filename], capture_output=True, text=True)
    output = result.stdout
    output = ast.literal_eval(output)
    return output
