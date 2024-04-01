"""
This file is adapted from Chinese-Vicuna repository.
https://github.com/Facico/Chinese-Vicuna
"""

import sys
import torch
import json
from tqdm import tqdm
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
from datasets import load_dataset
import argparse
import warnings
import os
import re
from utils import SteamGenerationMixin
"""
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
"""
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/model/13B_hf")
parser.add_argument("--lora_path", type=str, default="/home/tianjie/Documents/DialogueGeneration/trl/checkpoint/checkpoint-3000")
parser.add_argument("--data_path", type=str, default="/scratch/xy2128/Chinese-Vicuna/gsm_prolog_data/test.json")
parser.add_argument("--output_path", type=str, default="/scratch/xy2128/Chinese-Vicuna/gsm_prolog_data")
parser.add_argument("--use_local", type=int, default=1)
parser.add_argument("--part", type=int, default=1)
parser.add_argument("--output_name", type=str, default=None)
args = parser.parse_args()
print(args)
#tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# Xiaocheng: paddings config
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

CUTOFF_LEN = 274
K = 1
batch_size = 2

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path
DATA_PATH = args.data_path
OUTPUT_PATH = args.output_path
PART = args.part

if args.lora_path != "None":
    # fix the path for local checkpoint
    lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
    print(lora_bin_path)
    if not os.path.exists(lora_bin_path) and args.use_local:
        pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
        print(pytorch_bin_path)
        if os.path.exists(pytorch_bin_path):
            os.rename(pytorch_bin_path, lora_bin_path)
            warnings.warn(
                "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
            )
        else:
            assert ('Checkpoint is not Found!')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
#    model = LlamaForCausalLM.from_pretrained(
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    if args.lora_path != "None":
        model = SteamGenerationMixin.from_pretrained(
            model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map={"": 0}
        )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = SteamGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    if args.lora_path != "None":
        model = SteamGenerationMixin.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# Xiaocheng: mapping function
def generate_and_tokenize_prompt(data_points):
    user_prompt = generate_prompt(
        instruction = data_points["instruction"][0],
        input=data_points["input"][0],
    )
    input_ids = tokenizer(
        user_prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )["input_ids"]
    return {"input_ids": [input_ids]*K}

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.


model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    input_ids,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    max_new_tokens=768,
    min_new_tokens=1,
    repetition_penalty=1.3,
    **kwargs,
):
    input_ids = input_ids.to(device)
    generation_config = GenerationConfig(
        do_sample=False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=repetition_penalty,
        )
        return generation_output

data = load_dataset("json", data_files=DATA_PATH, split="train")
data = data.map(generate_and_tokenize_prompt, batched=True, batch_size=1, remove_columns=["instruction", "input", "output"], num_proc=4)
if PART == 1:
    data = data.select(list(range(0, 1*(len(data)//3))))
elif PART == 2:
    data = data.select(list(range(1*(len(data)//3), 2*(len(data)//3))))
elif PART == 3:
    data = data.select(list(range(2*(len(data)//3), len(data))))
else:
    data = data
data.set_format(type="torch", columns=["input_ids"])
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size)

if args.output_name:
    output_data = os.path.join(OUTPUT_PATH, args.output_name)
else:
    output_data = os.path.join(OUTPUT_PATH, f"output_part{PART}.json")

output_template = r"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
(.*)

### Input:
(.*)

### Response:(.*)"""
RE = re.compile(output_template, re.DOTALL)

already_ready = []
if os.path.isfile(output_data):
    with open(output_data, "r") as f:
        lines = f.readlines()
    for line in lines:
        if len(line) > 0:
            already_ready.append(json.loads(line))

with open(output_data, "w") as output_file:
    for ib, batch in enumerate(tqdm(dataloader)):
        if (ib + 1) * batch_size <= len(already_ready):
            for i in range(ib * batch_size, (ib + 1) * batch_size):
                data_point_to_write = dict()
                data_point_to_write["instruction"] = already_ready[i]["instruction"]
                data_point_to_write["input"] = already_ready[i]["input"]
                data_point_to_write["output"] = already_ready[i]["output"]
                json.dump(data_point_to_write, output_file)
                output_file.write("\n")
        else:
            generation_outputs = evaluate(batch["input_ids"]).sequences
            outputs = tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)
            for output in outputs:
                match = RE.search(output)
                if match is None:
                    print(output)
                data_point_to_write = dict()
                data_point_to_write["instruction"] = match.group(1).strip()
                data_point_to_write["input"] = match.group(2).strip()
                data_point_to_write["output"] = match.group(3).strip()
                json.dump(data_point_to_write, output_file)
                output_file.write("\n")

