import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from peft import PeftModel
import argparse

def main(method_name, forget, lr, batch_size):
    if method_name == "Original": model_name = "meta-llama/Llama-2-7b-chat-hf"
    else: model_name = "locuslab/tofu_ft_llama2-7b"

    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if method_name not in ["Original", "Retrain"]:
        model = PeftModel.from_pretrained(model, f"results/tofu/llama2-7b/{forget}/{method_name}/seed_1001/epoch5_{lr}_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last")
        model = model.merge_and_unload()
        print("Model loaded")

    model = model.eval()

    method_data = load_jsonl(f"generated_data/{forget}/{method_name}_{lr}.jsonl")
    already_written_length = len(method_data)

    print("Start from index:", already_written_length)

    data = load_jsonl(f"generated_data/{forget}_TRUE.jsonl")

    for i in tqdm(range(already_written_length, len(data), batch_size)):
        inputs = data[i:i+batch_size]
        outputs = get_model_generation(inputs, model, tokenizer)
        print(outputs)
        print(len(outputs))

        for idx, output in enumerate(range(batch_size)):
            new_line = deepcopy(data[i + idx])
            new_line["forget_response"] = outputs[idx]
            for top_idx, topk in enumerate([1, 5, 1800, 3600]):
                new_line[f"retain_response_top{topk}"] = outputs[idx + (top_idx + 1) * batch_size]
            for top_idx, topk in enumerate([1, 5, 1800, 3600]):
                new_line[f"mixed_response_top{topk}"] = outputs[idx + (top_idx + 5) * batch_size]
            for top_idx, topk in enumerate([1, 5, 1800, 3600]):
                new_line[f"mixed_response_reversed_top{topk}"] = outputs[idx + (top_idx + 9) * batch_size]
            new_line["retain_retain_response"] = outputs[idx + 13 * batch_size]
            with open(f"generated_data/{forget}/{method_name}_{lr}.jsonl", 'a') as writer:
                writer.write(json.dumps(new_line) + '\n')

    model.eval()

def get_model_generation(jsonl_inputs, model, tokenizer):
    length = len(jsonl_inputs)
    inputs = [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] {inp['question']} [/INST] [ANSWER] " for inp in jsonl_inputs]
    for mixed in [False, True]:
        for topk in [1, 5, 1800, 3600]:
            for i in range(length):
                if mixed: inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] {jsonl_inputs[i]['question']}, and also {jsonl_inputs[i]['retain_question_top'+str(topk)]} [/INST] [ANSWER] "]
                else: inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] {jsonl_inputs[i]['retain_question_top'+str(topk)]} [/INST] [ANSWER] "]

    for topk in [1, 5, 1800, 3600]:
        for i in range(length):
            inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] {jsonl_inputs[i]['retain_question_top'+str(topk)]}, and also {jsonl_inputs[i]['question']} [/INST] [ANSWER] "]

    inputs += [f" <<SYS>> You are a helpful assistant. <</SYS>> [INST] {jsonl_inputs[i]['retain_question_top3600']}, and also {jsonl_inputs[i]['retain_question_top1800']} [/INST] [ANSWER] " for inp in jsonl_inputs]

    print("INPUT LENGTH", len(inputs))
    
    encoded_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**encoded_inputs.to(model.device), max_new_tokens=128, do_sample=False, temperature=0.0).detach().cpu()
        sanity_generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(sanity_generation)
    sanity_generation = [generation.split("[ANSWER] ")[1].strip() for generation in sanity_generation]

    return sanity_generation

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    return data

def save_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--method', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--forget', type=str)
    args = parser.parse_args()

    print(f"Generate {args.method} batch with LR {args.lr}")
    main(args.method, args.forget, args.lr, 2)
