import os
import json
import functools
import re
from utils import *
from dataset import extract_answer, INVALID_ANS
from subprocess_prolog_wrapper import get_prolog_result
from threading import Thread
from tqdm import tqdm

output_file_path = "data/mistral-7b-r32/gsm_prolog_fixed_result"
output_file_name = "gsm_prolog_fixed_output_hard.jsonl"
test_file_name = "gsm_prolog_fixed_test_hard.jsonl"
accuracy_file_name_gsm = "gsm_fixed_accuracy_hard.json"
accuracy_file_name_prolog = "gsm_prolog_fixed_accuracy_hard.json"
# base_model = "2-7B_hf-r32"
# base_model = "code-llama-r32"
base_model = "mistral-7b-r32"
type = "gsm_prolog"

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(10)
def get_prolog_result_with_timeout(complete_filename):
    return get_prolog_result(complete_filename)

if __name__ == "__main__":
    with open(os.path.join(output_file_path, output_file_name), 'r') as f:
        data_points = f.readlines()

    with open(os.path.join(output_file_path, test_file_name), 'r') as f:
        answer_points = f.readlines()

    assert len(data_points) == len(answer_points)

    print("Start checking answers.")

    if type == "gsm_prolog":
        gsm = False
        prolog = True
        output_template = r"(.*)"
        output_map = {1:"code"}
    elif type == "gsm_prolog_aug":
        gsm = True
        prolog = True
        output_template = r"Answer:\n(.*)\n\nProlog code:\n(.*)"
        output_map = {1:"answer", 2:"code"}
    elif type == "gsm_prolog_aug2":
        gsm = True
        prolog = True
        output_template = r"Prolog code:\n(.*)\n\nAnswer:\n(.*)"
        output_map = {1:"code", 2:"answer"}
    elif type == "gsm":
        gsm = True
        prolog = False
        output_template = r"(.*)"
        output_map = {1:"answer"}
    else:
        raise Exception("Wrong format type!")
    RE = re.compile(output_template, re.DOTALL)

    total_sample = len(data_points)
    error_idx_prolog = dict()
    accuracy_prolog = 0
    error_prolog = 0
    error_idx_gsm = dict()
    accuracy_gsm = 0
    error_gsm = 0

    for idx in tqdm(range(len(data_points))):
        data_point, answer_point = json.loads(data_points[idx]), json.loads(answer_points[idx])
        instruction, input, output, answer = data_point['instruction'], data_point['input'], data_point['output'], answer_point['output']
        if isinstance(answer, int):
            output_match, answer_match = RE.search(output), answer
        else:
            output_match, answer_match = RE.search(output), RE.search(answer)
        assert answer_match is not None
        if output_match is None:
            error_idx_prolog[idx] = {"error_type": "SYNTAX"}
            error_prolog += 1
            error_idx_gsm[idx] = {'error_type': "SYNTAX"}
            error_gsm += 1
        else:
            output_dict, answer_dict = dict(), dict()
            for key, value in output_map.items():
                output_dict[value] = output_match.group(key).strip()
                if not isinstance(answer, int):
                    answer_dict[value] = answer_match.group(key).strip()
            if prolog:
                with open("temp_output.pl", 'w', encoding="utf-8") as f:
                    f.write(output_dict['code'])
                if not isinstance(answer, int):
                    with open("temp_answer.pl", 'w', encoding="utf-8") as f:
                        f.write(answer_dict['code'])
                try:
                    output_result = get_prolog_result_with_timeout("temp_output.pl")
                    if isinstance(answer, int):
                        answer_result = {'consult_ans': answer}
                    else:
                        answer_result = get_prolog_result_with_timeout("temp_answer.pl")
                except:
                    error_idx_prolog[idx] = {"error_type": "SYNTAX"}
                    error_prolog += 1
                else:
                    try:
                        if output_result['message'] == SYNTAXERROR_CODE:
                            error_idx_prolog[idx] = {'error_type': "SYNTAX"}
                            error_prolog += 1
                        elif output_result['consult_ans'] != answer_result['consult_ans']:
                            error_idx_prolog[idx] = {'error_type': "SEMANTIC", "prediction": output_result['consult_ans'], "target": answer_result['consult_ans']}
                            error_prolog += 1
                        else:
                            accuracy_prolog += 1
                    except:
                        error_idx_prolog[idx] = {'error_type': "SYNTAX"}
                        error_prolog += 1
            if gsm:
                output_result = extract_answer(output_dict['answer'])
                if isinstance(answer, int):
                    answer_result = str(answer)
                else:
                    answer_result = extract_answer(answer_dict['answer'])
                assert answer_result != INVALID_ANS
                if output_result == INVALID_ANS:
                    error_idx_gsm[idx] = {'error_type': "SYNTAX"}
                    error_gsm += 1
                elif output_result != answer_result:
                    error_idx_gsm[idx] = {'error_type': "SEMANTIC", "prediction": output_result, "target": answer_result}
                    error_gsm += 1
                else:
                    accuracy_gsm += 1

    accuracy_prolog /= total_sample
    error_prolog /= total_sample
    accuracy_gsm /= total_sample
    error_gsm /= total_sample

    print("Start saving accuracy records.")

    if gsm:
        with open(os.path.join(output_file_path, accuracy_file_name_gsm), 'w') as f:
            accuracy_record = {
                "error_idx": error_idx_gsm,
                "total_sample": total_sample,
                "accuracy": accuracy_gsm,
                "error": error_gsm,
                "base_model": base_model,
                }
            json.dump(accuracy_record, f)
    if prolog:
        with open(os.path.join(output_file_path, accuracy_file_name_prolog), 'w') as f:
            accuracy_record = {
                "error_idx": error_idx_prolog,
                "total_sample": total_sample,
                "accuracy": accuracy_prolog,
                "error": error_prolog,
                "base_model": base_model,
                }
            json.dump(accuracy_record, f)

    print("Job done.")
