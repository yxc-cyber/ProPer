import os
import openai
import backoff
from tqdm import tqdm
from utils import *
from dataset import get_examples, extract_answer
from subprocess_prolog_wrapper import get_prolog_result

examples = get_examples('train')

openai.api_base = ""
openai.api_key = ""

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_messages(questions, target):
    messages = [
        {"role": "system", "content": "You are a helpful, pattern-following assistant that translates math problems into Proog codes."},
    ]

    for idx in questions:
        question = examples[idx]['question'].strip()
        answer = examples[idx]['answer'].strip()
        with open(f'prolog_files/{idx}.pl', 'r') as prolog_file:
            prolog_code = prolog_file.read().strip()
        messages.append(
            {"role": "system", "name": "example_user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"},
        )
        messages.append(
            {"role": "system", "name": "example_assistant", "content": f"{prolog_code}"},
        )

    target_question = examples[target]['question'].strip()
    target_answer = examples[target]['answer'].strip()
    target_ans = extract_answer(target_answer)
    messages.append(
        {"role": "user", "content": f"Question:\n{target_question}\n\nAnswer:\n{target_answer}"},
    )

    return messages, target_ans

if __name__ == "__main__":

    for target in tqdm(range(1319)):

        messages, target_ans = generate_messages([40,11,30,25,38,37,8,9], target)

        completion = completions_with_backoff(
            model = "gpt-3.5-turbo-16k",
            messages = messages,
        )

        prolog_code = completion.choices[0].message.content

        messages.append(
            {"role": "assistant", "content": f"{prolog_code}"},
        )

        with open(f"prolog_files/{target}.pl", 'w') as prolog_file:
            prolog_file.write(prolog_code)

        try:
            complete_filename = f"prolog_files/{target}.pl"
            prolog_result = get_prolog_result(complete_filename)
            assert prolog_result['message'] == 0
            consult_ans = prolog_result['consult_ans']
            if int(consult_ans) != int(target_ans):
                os.rename(f"prolog_files/{target}.pl", f"prolog_files/{target}{SEMANTICERROR_FLAG}.pl")
        except:
            os.rename(f"prolog_files/{target}.pl", f"prolog_files/{target}{SYNTAXERROR_FLAG}.pl")