import os
import openai
import backoff
import re
import tiktoken
import functools
import random
from threading import Thread
from collections import Counter
from tqdm import tqdm
from utils import *
from dataset import get_examples, extract_answer
from subprocess_prolog_wrapper import get_prolog_result

examples = get_examples('train')
targeted_path = 'prolog_files'

openai.api_base = ""
openai.api_key = ""

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_messages(questions, target, wrong_code=None):
    if wrong_code is not None:
        messages = [
            {"role": "system", "content": "You are a helpful, pattern-following assistant that translates math problems into Prolog codes and corrects wrong codes."},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful, pattern-following assistant that translates math problems into Prolog codes."},
        ]

    for idx in questions:
        question = examples[idx]['question'].strip()
        answer = examples[idx]['answer'].strip()
        with open(f'prolog_files/{idx}.pl', 'r') as prolog_file:
            prolog_code = prolog_file.read().strip()
        if wrong_code is not None:
            if os.path.exists(f'prolog_files/{idx}#.pl'):
                with open(f'prolog_files/{idx}#.pl', 'r') as prolog_file:
                    prolog_wrong_code = prolog_file.read().strip()
            else:
                with open(f'prolog_files/{idx}##.pl', 'r') as prolog_file:
                    prolog_wrong_code = prolog_file.read().strip()
            messages.append(
                {"role": "system", "name": "example_user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}\n\nWrong Code:\n{prolog_wrong_code}"},
            )
        else:
            messages.append(
                {"role": "system", "name": "example_user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"},
            )
        messages.append(
            {"role": "system", "name": "example_assistant", "content": f"{prolog_code}"},
        )

    target_question = examples[target]['question'].strip()
    target_answer = examples[target]['answer'].strip()
    target_ans = extract_answer(target_answer)
    if wrong_code is not None:
        messages.append(
            {"role": "user", "content": f"Question:\n{target_question}\n\nAnswer:\n{target_answer}\n\nnWrong Code:\n{wrong_code}"},
        )
    else:
        messages.append(
            {"role": "user", "content": f"Question:\n{target_question}\n\nAnswer:\n{target_answer}"},
        )

    return messages, target_ans

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value, allowed_special={"<|endoftext|>"}))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

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

@timeout(600)
def get_prolog_result_with_timeout(complete_filename):
    return get_prolog_result(complete_filename)

if __name__ == "__main__":

    for epoch in range(1):

        file_lengths = Counter()

        for (root, dirs, files) in os.walk(targeted_path):
            for filename in files:
                if "#" not in filename:
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r') as prolog_file:
                        file_lengths[int(filename[:-3])] = len(prolog_file.read().strip())

        prompt_candidate_tuples = file_lengths.most_common(64)
        prompt_candidate_2 = [item[0] for item in prompt_candidate_tuples][:20]

        prompt_candidate_1 = [40,11,30,25,38] + random.sample([item[0] for item in prompt_candidate_tuples],5)

        prompt_candidate_0 = [13,22,64,67,72,83,1910,2115,223,231,2340,2439,2825,3081,3387,3442,3492,3740,3799,4108,4216,441,4426,4559,4590,4790,4795,4878,49,4936,4945,4994,5094,5289,529,5400,5404,5539,5587,5605,5979,6129,6234,6299,6465,6564,6711,6770,6790,6894,7043,7055,7077,7109,7304,7320,7366,7418,767,769,783,886,916,941,947]

        # messages, target_ans = generate_messages(prompt_candidate[:20], prompt_candidate[0])
        # print(num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"))

        prompt_candidate = prompt_candidate_1

        num = 0
        fixed_num = 0
        for (root, dirs, files) in os.walk(targeted_path):
            for filename in tqdm(files, desc=f"Epoch {epoch}"):
                if "#" in filename:

                    regex = re.compile(r"(\-?[0-9\.\,]+)#")
                    match = regex.search(filename)
                    target = int(match.group(1).strip())

                    if target in prompt_candidate_0:
                        continue

                    with open(os.path.join(root, filename), 'r') as prolog_file:
                        wrong_code = prolog_file.read().strip()

                    messages, target_ans = generate_messages(prompt_candidate, target)

                    completion = completions_with_backoff(
                        model = "gpt-3.5-turbo-16k",
                        messages = messages,
                    )

                    prolog_code = completion.choices[0].message.content

                    messages.append(
                        {"role": "assistant", "content": f"{prolog_code}"},
                    )

                    with open(f"prolog_files/{target}.pl", 'w') as prolog_file:
                        try:
                            regex = re.compile(r":- use_module\(library\(clpq\)\)\..*?solve.*?\.", re.DOTALL)
                            prolog_code = regex.search(prolog_code).group()
                        except:
                            pass
                        finally:
                            prolog_file.write(prolog_code)

                    try:
                        assert prolog_code != ''
                        complete_filename = f"prolog_files/{target}.pl"
                        prolog_result = get_prolog_result_with_timeout(complete_filename)
                        assert prolog_result['message'] == 0
                        consult_ans = prolog_result['consult_ans']
                        if int(consult_ans) != int(target_ans):
                            os.remove(os.path.join(root, filename))
                            os.rename(f"prolog_files/{target}.pl", f"prolog_files/{target}{SEMANTICERROR_FLAG}.pl")
                        else:
                            fixed_num += 1
                            num += 1
                            os.remove(os.path.join(root, filename))
                            tqdm.write(f"Fixed sample: {target}")
                    except Exception as e:
                        # tqdm.write(e.__class__.__name__)
                        os.remove(os.path.join(root, filename))
                        os.rename(f"prolog_files/{target}.pl", f"prolog_files/{target}{SYNTAXERROR_FLAG}.pl")

                else:
                    num += 1
        print(f"Valid samples: {num}, Fixed samples: {fixed_num}")