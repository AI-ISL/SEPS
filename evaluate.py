import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from peft import PeftModel
import argparse

def single_eval_template(q, gt, o, tokenizer):
    out = f"""Evaluation Criteria
Evaluate whether the [OUTPUT] contains full information about the [Ground Truth] on a scale of 0 to 10.

If the [OUTPUT] contains all the information from the [Ground Truth], assign a score of 10.
If it contains none or is gibberish, assign a score of 0.
For partially correct responses, assign scores between 1 and 9 based on the amount of relevant information included:
    * 1-3: Very little information present.
    * 4-6: Some relevant information but incomplete.
    * 7-9: Most information included but with minor omissions or inaccuracies.

[Question] {q}
[Ground Truth] {gt}

Now evaluate the [OUTPUT] based on the QA pairs.

[OUTPUT]"{o}"

PLEASE ONLY TYPE a single number for [ANSWER] nothing else.

[ANSWER] """

    out = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": out}
    ]

    return tokenizer.apply_chat_template(out, tokenize=False)

def double_eval_template(qf, qr, gtf, gtr, o, tokenizer):
    out = f"""Evaluation Criteria
Evaluate whether the [OUTPUT] contains full information about [GT 1] and [GT 2] on a scale of 0 to 10.

If the [OUTPUT] contains all the information from [GT 1], assign a score of 10.
If it contains none or is gibberish, assign a score of 0.
For partially correct responses, assign scores between 1 and 9 based on the amount of relevant information included:
    * 1-3: Very little information present.
    * 4-6: Some relevant information but incomplete.
    * 7-9: Most information included but with minor omissions or inaccuracies.
Repeat the evaluation for [GT 2].

Provide the scores as a list in the format ['A','B'], where:

'A' is the score for [GT 1]. (0-10)
'B' is the score for [GT 2]. (0-10)

[QUESTION 1] {qf} [GT 1] {gtf}
[QUESTION 2] {qr} [GT 2] {gtr}

Now evaluate the [OUTPUT] based on the QA pairs.

[OUTPUT]"{o}"

PLEASE ONLY TYPE ['A','B'] for [ANSWER] nothing else.

[ANSWER] """

    out = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": out}
    ]

    return tokenizer.apply_chat_template(out, tokenize=False)

def main(method_name, forget, lr, batch_size):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()

    score_data = load_jsonl(f"generated_data/{forget}/{method_name}_{lr}_score.jsonl")
    already_written_length = len(score_data)

    print("Start from index:", already_written_length)

    method_data = load_jsonl(f"generated_data/{forget}/{method_name}_{lr}.jsonl")

    for i in tqdm(range(already_written_length, len(method_data), batch_size)):
        inputs = method_data[i:i+batch_size]
        outputs = get_model_generation(inputs, model, tokenizer)

        print(outputs)
        print(len(outputs))
        for idx, output in enumerate(range(batch_size)):
            new_line = deepcopy(method_data[i + idx])
            new_line["forget_score"] = outputs[idx]
            for top_idx, topk in enumerate([1, 5, 1800, 3600]):
                new_line[f"retain_score_top{topk}"] = outputs[idx + (top_idx + 1) * batch_size]
            for top_idx, topk in enumerate([1, 5, 1800, 3600]):
                new_line[f"mixed_score_top{topk}"] = outputs[idx + (top_idx + 5) * batch_size]
            for top_idx, topk in enumerate([1, 5, 1800, 3600]):
                new_line[f"mixed_score_reversed_top{topk}"] = outputs[idx + (top_idx + 9) * batch_size]
            new_line["retain_retain_score"] = outputs[idx + 13 * batch_size]
            with open(f"generated_data/{forget}/{method_name}_{lr}_score.jsonl", 'a') as writer:
                writer.write(json.dumps(new_line) + '\n')

    model.eval()

def get_model_generation(jsonl_inputs, model, tokenizer):
    length = len(jsonl_inputs)
    inputs = [single_eval_template(inp["question"], inp["answer"], inp["forget_response"], tokenizer) for inp in jsonl_inputs]
    for mixed in [False, True]:
        for topk in [1, 5, 1800, 3600]:
            for i in range(length):
                if mixed: inputs += [double_eval_template(jsonl_inputs[i]["question"], jsonl_inputs[i][f"retain_question_top{topk}"], jsonl_inputs[i]["answer"], jsonl_inputs[i][f"retain_answer_top{topk}"], jsonl_inputs[i][f"mixed_response_top{topk}"], tokenizer)]
                else: inputs += [single_eval_template(jsonl_inputs[i][f"retain_question_top{topk}"], jsonl_inputs[i][f"retain_answer_top{topk}"], jsonl_inputs[i][f"retain_response_top{topk}"], tokenizer)]

    for topk in [1, 5, 1800, 3600]:
        for i in range(length):
            inputs += [double_eval_template(jsonl_inputs[i]["question"], jsonl_inputs[i][f"retain_question_top{topk}"], jsonl_inputs[i]['answer'], jsonl_inputs[i][f"retain_answer_top{topk}"], jsonl_inputs[i][f"mixed_response_reversed_top{topk}"], tokenizer)]
    
    inputs += [double_eval_template(inp['retain_question_top3600'], inp['retain_question_top1800'], inp['retain_answer_top3600'], inp['retain_answer_top1800'], inp['retain_retain_response'], tokenizer) for inp in jsonl_inputs]

    outputs = []
    print(type(inputs))
    print(len(inputs))

    with torch.no_grad():
        for inp in tqdm(inputs):
            encoded_input = tokenizer(inp, return_tensors="pt", padding=True, truncation=True)
            outputs += model.generate(**encoded_input.to(model.device), max_new_tokens=10, do_sample=False, temperature=0.0).detach().cpu()

        sanity_generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(sanity_generation)
    sanity_generation = [generation.split("assistant\n\n")[1].strip() for generation in sanity_generation]

    return sanity_generation

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--method', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--forget', type=str)
    args = parser.parse_args()

    print(f"Generate {args.method} batch with LR {args.lr}")
    main(args.method, args.forget, args.lr, 2)
