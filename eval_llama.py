import os
import gc
import torch
import shutil
import argparse
from tqdm import tqdm
from templates import double_eval_template, single_eval_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import defaultdict
from copy import deepcopy
import json
import jsonlines
import re
from pathlib import Path

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    else: Path(file_path).touch()
    return data

def get_model_generation(inputs, model, tokenizer, max_new_tokens=128):
    encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(**encoded_inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0).detach().cpu()
        sanity_generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if "[ANSWER]assistant\n\n" in sanity_generation[0]: sanity_generation = [generation.split("[ANSWER]assistant\n\n")[1].strip() for generation in sanity_generation]
    else: sanity_generation = [generation.split("[ANSWER] ")[1].strip() for generation in sanity_generation]

    del encoded_inputs
    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return sanity_generation
    
def get_result_dir(args):
    if "ME+GD" in args.method:
        return f'results/tofu/llama2-7b/{args.forget}/{args.method}/seed_1001/epoch{args.epochs}_{args.lr}_FixRefFalse_maskFalse_1.0_1.0/1/unlearn_times_1'
    else:
        return f'results/tofu/llama2-7b/{args.forget}/{args.method}/seed_1001/epoch{args.epochs}_{args.lr}_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1'

def get_args():
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--method', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--use_LoRA', action='store_true', help="If given, use LoRA.")
    parser.add_argument('--forget', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)

    return parser.parse_args()

def generate(args):
    if args.method == "Original": model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif args.method == "Retrain" or args.use_LoRA: model_name = "locuslab/tofu_ft_llama2-7b"
    else: model_name = f"{get_result_dir(args)}/checkpoint-last"

    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.method not in ["Original", "Retrain"] and args.use_LoRA:
        model = PeftModel.from_pretrained(model, f"{get_result_dir(args)}/checkpoint-last")
        model = model.merge_and_unload()

    inputs = []
    model.eval()

    data = load_jsonl(f'generated_data/TRUE/{args.forget}_TRUE.jsonl')
    progress_data = load_jsonl(f'{get_result_dir(args)}/eval_results-last/generated.jsonl')
    progress = len(progress_data)

    for i in tqdm(range(progress, len(data), args.batch_size)):
        inputs = []
        raw_inputs = data[i:i+args.batch_size]
        for raw_idx, raw_input in enumerate(raw_inputs):
            inputs += [f"<<SYS>> You are a helpful assistant. <</SYS>> [INST] Q: {raw_input['question']}. [/INST] [ANSWER] "]
            inputs += [f"<<SYS>> You are a helpful assistant. <</SYS>> [INST] Q: {raw_input['retain_question']}. [/INST] [ANSWER] "]
            if raw_idx > 0:
                inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] Q1: {raw_inputs[raw_idx-1]['question']}. Q2: {raw_input['question']} [/INST] [ANSWER] "]
                inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] Q1: {raw_inputs[raw_idx-1]['retain_question']}. Q2: {raw_input['retain_question']} [/INST] [ANSWER] "]
            else:
                inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] Q1: {raw_inputs[len(raw_inputs)-1]['question']}. Q2: {raw_input['question']} [/INST] [ANSWER] "]
                inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] Q1: {raw_inputs[len(raw_inputs)-1]['retain_question']}. Q2: {raw_input['retain_question']} [/INST] [ANSWER] "]

            inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] Q1: {raw_input['question']}. Q2: {raw_input['retain_question']} [/INST] [ANSWER] "]
            inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] Q1: {raw_input['retain_question']}. Q2: {raw_input['question']} [/INST] [ANSWER] "]
        generations = get_model_generation(inputs, model, tokenizer, max_new_tokens=128)
        with open(f'{get_result_dir(args)}/eval_results-last/generated.jsonl', 'a') as writer:
            for idx, raw_input in enumerate(raw_inputs):
                new_line = deepcopy(raw_input)
                new_line['forget'] = generations[6*idx]
                new_line['retain'] = generations[6*idx+1]
                new_line['forget_forget'] = generations[6*idx+2]
                new_line['retain_retain'] = generations[6*idx+3]
                new_line['forget_retain'] = generations[6*idx+4]
                new_line['retain_forget'] = generations[6*idx+5]
                writer.write(json.dumps(new_line) + '\n')

        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
        
def scoring(args):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()

    data = load_jsonl(f'{get_result_dir(args)}/eval_results-last/generated.jsonl')
    progress_data = load_jsonl(f'{get_result_dir(args)}/eval_results-last/evaluated.jsonl')
    progress = len(progress_data)

    for i in tqdm(range(progress, len(data), args.batch_size)):
        inputs = []
        raw_inputs = data[i:i+args.batch_size]
        for raw_idx, raw_input in enumerate(raw_inputs):
            inputs += [single_eval_template(raw_input['question'], raw_input['answer'], raw_input['forget'], tokenizer)]
            inputs += [single_eval_template(raw_input['retain_question'], raw_input['retain_answer'], raw_input['retain'], tokenizer)]
            if raw_idx > 0:
                inputs += [double_eval_template(raw_inputs[raw_idx-1]['question'], raw_input['question'], raw_inputs[raw_idx-1]['answer'], raw_input['answer'], raw_input['forget_forget'], tokenizer)]
                inputs += [double_eval_template(raw_inputs[raw_idx-1]['retain_question'], raw_input['retain_question'], raw_inputs[raw_idx-1]['retain_answer'], raw_input['retain_answer'], raw_input['retain_retain'], tokenizer)]
            else:
                inputs += [double_eval_template(raw_inputs[len(raw_inputs)-1]['question'], raw_input['question'], raw_inputs[len(raw_inputs)-1]['answer'], raw_input['answer'], raw_input['forget_forget'], tokenizer)]
                inputs += [double_eval_template(raw_inputs[len(raw_inputs)-1]['retain_question'], raw_input['retain_question'], raw_inputs[len(raw_inputs)-1]['retain_answer'], raw_input['retain_answer'], raw_input['retain_retain'], tokenizer)]

            inputs += [double_eval_template(raw_input['question'], raw_input['retain_question'], raw_input['answer'], raw_input['retain_answer'], raw_input['forget_retain'], tokenizer)]
            inputs += [double_eval_template(raw_input['retain_question'], raw_input['question'], raw_input['retain_answer'], raw_input['answer'], raw_input['retain_forget'], tokenizer)]

        generations = get_model_generation(inputs, model, tokenizer, max_new_tokens=10)

        with open(f'{get_result_dir(args)}/eval_results-last/evaluated.jsonl', 'a') as writer:
            for idx, raw_input in enumerate(raw_inputs):
                new_line = deepcopy(raw_input)
                new_line['forget_score'] = generations[6*idx]
                new_line['retain_score'] = generations[6*idx+1]
                new_line['forget_forget_score'] = generations[6*idx+2] # [forget, forget]
                new_line['retain_retain_score'] = generations[6*idx+3] # [retain, retain]
                new_line['forget_retain_score'] = generations[6*idx+4] # [forget, retain]
                new_line['retain_forget_score'] = generations[6*idx+5] # [retain, forget]
                writer.write(json.dumps(new_line) + '\n')

def organize(args):
    lines = load_jsonl(f'{get_result_dir(args)}/eval_results-last/evaluated.jsonl')
    results = defaultdict(list)

    try: 
        for line in lines:
            print(line)

            r_score = int(line['retain_score'])
            f_score = int(line['forget_score'])
            

            rr_score = re.findall(r'\d+', line['retain_retain_score'])
            rr_score = [int(score) for score in rr_score]
            ff_score = re.findall(r'\d+', line['forget_forget_score'])
            ff_score = [int(score) for score in ff_score]

            fr_score = re.findall(r'\d+', line['forget_retain_score'])
            fr_score = [int(score) for score in fr_score]
            rf_score = re.findall(r'\d+', line['retain_forget_score'])
            rf_score = [int(score) for score in rf_score]

            results['r'] += [r_score]
            results['f'] += [f_score]

            results['rr_1r'] += [rr_score[0]]
            results['rr_2r'] += [rr_score[1]]
            results['ff_1f'] += [ff_score[0]]
            results['ff_2f'] += [ff_score[1]]

            results['rf_r'] += [rf_score[0]]
            results['rf_f'] += [rf_score[1]]
            results['fr_f'] += [fr_score[0]]
            results['fr_r'] += [fr_score[1]]
    except:
        pass
    
    r_single = sum(results['r']) / len(results['r'])
    f_single = sum(results['f']) / len(results['f'])

    rr_1r = sum(results['rr_1r']) / len(results['rr_1r'])
    rr_2r = sum(results['rr_2r']) / len(results['rr_2r'])
    ff_1f = sum(results['ff_1f']) / len(results['ff_1f'])
    ff_2f = sum(results['ff_2f']) / len(results['ff_2f'])

    rf_r = sum(results['rf_r']) / len(results['rf_r'])
    rf_f = sum(results['rf_f']) / len(results['rf_f'])
    fr_r = sum(results['fr_r']) / len(results['fr_r'])
    fr_f = sum(results['fr_f']) / len(results['fr_f'])


    f = (rf_f + fr_f + ff_1f + ff_2f) / 4
    r = (fr_r + rf_r + rr_1r + rr_2r) / 4


    with open(f'{get_result_dir(args)}/eval_results-last/mixed_results.txt', 'w') as txtfile:
        txtfile.write(f"[Single] Retain Score: {r_single}\n")
        txtfile.write(f"[Single] Forget Score: {f_single}\n")

        txtfile.write(f"[Retain-Retain] 1st Retain Score: {rr_1r}\n")
        txtfile.write(f"[Retain-Retain] 2nd Retain Score: {rr_2r}\n")
        txtfile.write(f"[Forget-Forget] 1st Forget Score: {ff_1f}\n")
        txtfile.write(f"[Forget-Forget] 2nd Forget Score: {ff_2f}\n")

        txtfile.write(f"[Retain-Forget] Retain Score: {rf_r}\n")
        txtfile.write(f"[Retain-Forget] Forget Score: {rf_f}\n")
        txtfile.write(f"[Forget-Retain] Retain Score: {fr_r}\n")
        txtfile.write(f"[Forget-Retain] Forget Score: {fr_f}\n")

        txtfile.write(f"Mean Retain Score: {r}\n")
        txtfile.write(f"Mean Forget Score: {f}\n")

def main(args):
    if not os.path.exists(f'{get_result_dir(args)}/eval_results-last/evaluated.jsonl'): 
        print("Start Generation")
        generate(args)

    if not os.path.exists(f'{get_result_dir(args)}/eval_results-last/mixed_results.txt'):
        print("Start Scoring")
        scoring(args)
    
    organize(args)
    if not args.use_LoRA: shutil.rmtree(f'{get_result_dir(args)}/checkpoint-last')

if __name__ == '__main__':
    args = get_args()

    print(f"[ Testing on {args.forget} ] Generate {args.method} batch with LR {args.lr}. Use LoRA: {args.use_LoRA}")

    main(args)