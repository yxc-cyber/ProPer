"""
This file is adapted from Chinese-Vicuna repository.
https://github.com/Facico/Chinese-Vicuna
"""

import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets, load_from_disk
import transformers
import argparse
import warnings
"""
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
"""
# Wilson
# Make it auto
#from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
# END
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

"""
# Xiaocheng: Enable the overwriting of peft model.
from peft.utils import (
    WEIGHTS_NAME,
    PromptLearningConfig,
)
"""

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--val_data_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--micro_batch", type=int, default=4)
parser.add_argument("--total_batch", type=int, default=128)
parser.add_argument("--log_steps", type=int, default=20)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--test_size", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"
# optimized for RTX 4090. for larger GPUs, increase some of these?
#MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
ref_columns = ["instruction", "input", "output"]
MICRO_BATCH_SIZE = args.micro_batch  # this could actually be 5 but i like powers of 2
#BATCH_SIZE = 128
BATCH_SIZE = args.total_batch
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 512
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size

# Wilson: Need to look at the variable names to be LORA-adapted
if "bloom" in args.model_path:
    TARGET_MODULES = [
        "query_key_value",
    ]
    # really a hack here. we should resolve the torch/cudnn issue
    torch.backends.cudnn.enabled = False
else:
    # Good for LLama, OPT
    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
# END

DATA_PATH = args.data_path #"/home/cciip/private/fanchenghao/dataset/instruction/merge.json"
VAL_DATA_PATH = args.val_data_path
OUTPUT_DIR = args.output_path #"lora-Vicuna"
if VAL_DATA_PATH:
    VAL_SET_SIZE = 0

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(args.model_path)
# Wilson
#model = LlamaForCausalLM.from_pretrained(
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
)
# Wilson
#tokenizer = LlamaTokenizer.from_pretrained(
#    args.model_path, add_eos_token=True
#tokenizer = AutoTokenizer.from_pretrained(
#    args.model_path
#)
# Xiaocheng: The tokenizer above does not have eos. Check tokenizer_config and you should find eos token is turned off.
tokenizer =  AutoTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
#tokenizer.padding_side = "left"  # Allow batched inference
#Xiaocheng: By default, the padding side is left. We need to change it to make sure input ids end up with eos.
tokenizer.padding_side = "right"

if "," in DATA_PATH:
    # multiple dataset paths/names
    data_names = DATA_PATH.split(",")
    dataset_objs = []
    for name in data_names:
        if name.endswith(".json"):
            x = load_dataset("json", data_files=name, split="train")
        else:
            if name.endswith("-local"):
                x = load_from_disk(name)
            else:
                x = load_dataset(name, split="train")
            if name == "MBZUAI/LaMini-instruction" or name.endswith("MBZUAI/LaMini-instruction-local"):
                x = x.add_column("input", [""] * len(x))
                x = x.rename_columns({"response": "output"})
            elif name.endswith("unnatural_instruction_gpt4_data.json"):
                x = x.remove_columns(["output"])
                x = x.rename_columns({"label": "output"})
        rm_cols = [z for z in x.features if z not in ref_columns]
        if len(rm_cols) > 0:
           x = x.remove_columns(rm_cols)
        dataset_objs.append(x)
    data = concatenate_datasets(dataset_objs)
else:
    if DATA_PATH.endswith(".json"):
        data = load_dataset("json", data_files=DATA_PATH, split="train")
    else:
        # assumed an HF dataset (one dataset is supported here)
        if DATA_PATH.endswith("-local"):
            data = load_from_disk(DATA_PATH)
        else:
            data = load_dataset(DATA_PATH, split="train")
        if DATA_PATH == "MBZUAI/LaMini-instruction" or DATA_PATH.endswith("MBZUAI/LaMini-instruction-local"):
            data = data.rename_columns({"response": "output"})
        elif DATA_PATH.endswith("unnatural_instruction_gpt4_data.json"):
            data = data.remove_columns(["output"])
            data = data.rename_columns({"label": "output"})

now_max_steps = max((len(data) - VAL_SET_SIZE) // BATCH_SIZE * EPOCHS, EPOCHS)

if args.resume_from_checkpoint:
# Check the available weights and load them
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
)  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None  # So the trainer won't try loading its state
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        #model = set_peft_model_state_dict(model, adapters_weights)
        #Xiaocheng: The line above is problematic.
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
    
    train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
    
    if os.path.exists(train_args_path):
        import json
        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps


model.print_trainable_parameters()

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        )
        if "input" in data_point and data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
        - 1
    )  # no eos token
    result = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    full_tokens = result["input_ids"][:-1]
    attention_mask = result["attention_mask"][:-1]
    labels = [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:]
    # Xiaocheng: exclude the paddings
    labels = masked_fill_for_list(labels, attention_mask, -100)
    return {
        "input_ids": full_tokens,
        "labels": labels,
        "attention_mask": attention_mask,
    }

# Xiaocheng: List objects does not have masked_fill_. Thus, it's mannully implemented.
def masked_fill_for_list(lissy, mask, value):
    assert len(lissy) == len(mask)
    for i in range(len(mask)):
        if mask[i] == 0:
            lissy[i] = value
    return lissy

if VAL_SET_SIZE > 0:
    train_val = data.train_test_split(
        # Xiaocheng:
        # Don't shuffle and swtich the position for val and training sets so that each time the val set will always pick the first several samples.
        # test_size=VAL_SET_SIZE, shuffle=True, seed=42
        test_size=len(data)-VAL_SET_SIZE, shuffle=False, seed=42,
    )
    # Xiaocheng: Note that here val and training sets have been switched.
    train_data = train_val["test"].map(generate_and_tokenize_prompt, num_proc=4)
    val_data = train_val["train"].map(generate_and_tokenize_prompt)
else:
    train_data = data.shuffle(seed=42).map(generate_and_tokenize_prompt, num_proc=4)
    if VAL_DATA_PATH:
       data = load_dataset("json", data_files=VAL_DATA_PATH, split="train")
       val_data = data.shuffle(seed=42).map(generate_and_tokenize_prompt)
    else:
        val_data = None
VAL_SET_SIZE = args.test_size

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=args.warmup_steps,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_steps=args.log_steps,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=60,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
# Wilson
# Always set to False as 1GPU also needs this one
#        ddp_find_unused_parameters=False if ddp else None,
        ddp_find_unused_parameters=False if ddp else False,
# END
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
        deepspeed="./deepspeed_config_s3.json" if args.deepspeed else None,
        local_rank = args.local_rank,
    ),
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

"""
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))
"""

'''
# Xiaocheng: Overwrite the save_pretrained() method to avoid calling get_model_state_dict() twice.
def save_pretrained(self, save_directory, **kwargs):
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)

    for adapter_name, peft_config in self.peft_config.items():
        # save only the trainable weights
        #output_state_dict = get_peft_model_state_dict(
        #    self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
        #)
        # Xiaocheng: Don't peft twice.
        output_state_dict = kwargs.get("state_dict", None)
        if output_state_dict is None:
            output_state_dict = self.state_dict()
        output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
        os.makedirs(output_dir, exist_ok=True)
        torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if isinstance(peft_config, PromptLearningConfig)
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True
        peft_config.save_pretrained(output_dir)
        peft_config.inference_mode = inference_mode

model.save_pretrained = save_pretrained.__get__(model, type(model))
'''

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("\n If there's a warning about missing keys above, please disregard :)")

# This is just to save the adapter_config.json conveniently
model.save_pretrained(OUTPUT_DIR)
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
#trainer.train(resume_from_checkpoint=True)

#model.save_pretrained(OUTPUT_DIR)
# Xiaocheng: The line above seems not able to correctly save the best model. It just save an empty one instead.
final_dir = os.path.join(OUTPUT_DIR, "checkpoint-final")
os.makedirs(final_dir, exist_ok = True)
trainer.save_model(final_dir)

