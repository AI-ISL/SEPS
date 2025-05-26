import json
import os

import datasets
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import get_model_identifiers_from_yaml


def dataset_to_json(dataset, filename, ):
    data_nums = len(dataset)
    with open(filename, "w") as f:
        for i in range(data_nums):
            row_data = dataset[i]
            json_data = json.dumps(row_data)
            f.write(json_data)
            f.write('\n')


# adopt from TOFU: https://github.com/locuslab/tofu/blob/80159d8ea39edf147fb09cd82aefa08e506e6718/data_module.py#L8
def convert_raw_forget_data_to_model_format(tokenizer, max_length, question, answer, model_configs, mask=True, prefilling=None):
    question_start_token, question_end_token, answer_token = model_configs[
        'question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    if prefilling is not None:
        new_question = question_start_token + question + question_end_token + prefilling
    else:
        new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if mask:
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        # change label to -100 for question tokens
        for i in range(num_question_tokens): label[i] = -100
    else:
        label = pad_input_ids
    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs[
        'question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    # print(tokenizer.special_tokens_map)
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)




class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None, question_key='question',
                 answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = datasets.load_dataset(
            'json', data_files=os.path.join(data_path, split + '.json'))['train']

        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(), \
            torch.stack(label_list).squeeze(), \
            torch.stack(pad_attention_mask_list).squeeze()


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(
        attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_forget(samples):
    rets = []

    # Extracting samples for each data type
    data_types = ["forget", "retain", "forget_idk", "retain_idk", "forget+retain_nomask","retain+forget_nomask","forget+retain_nomask_untargeted","retain+forget_nomask_untargeted"]

    samples_dict = {data_type: [sample[i] for sample in samples] for i, data_type in enumerate(data_types)}

    for data_type in data_types:
        data = samples_dict[data_type]

        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]

        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

    return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss



class TextForgetDatasetQA(Dataset):
    def __init__(self, tokenizer, model_family, forget_data, retain_data, max_length=512, mask=False):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.mask = mask

        self.model_configs = get_model_identifiers_from_yaml(model_family)

        self.idontknowfile = "data/idontknow.jsonl"
        with open(self.idontknowfile, "r") as f:
            self.idk = f.readlines()

        self.data_types = ["forget", "retain", "forget_idk", "retain_idk", "forget+retain_nomask","retain+forget_nomask","forget+retain_nomask_untargeted","retain+forget_nomask_untargeted"]

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        torch.manual_seed(idx)
        retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        rand_pos = torch.randint(0, len(self.idk), (1,)).item()

        for data_type in self.data_types:
            if "+" in data_type:
                if "forget+retain" in data_type:      ##"forget+retain"
                    question = "1." + self.forget_data[idx]['question'] + "\n" + "2. " + self.retain_data[retain_idx]['question']
                    if "nomask_untargeted" in data_type:
                        answer_first = None
                        answer_last = "1." + self.forget_data[idx]['answer'] + "\n" + "2. " + self.retain_data[retain_idx]['answer']
                    elif "nomask" in data_type:
                        answer_first = None
                        answer_last = "1." + self.idk[rand_pos].strip() + "\n" + "2. " + self.retain_data[retain_idx]['answer']
                elif "retain+forget" in data_type:    ##"retain+forget"
                    question = "1. " + self.retain_data[retain_idx]['question'] + "\n" + "2." + self.forget_data[idx]['question']
                    if "nomask_untargeted" in data_type:
                        answer_first = None
                        answer_last = "1. " + self.retain_data[retain_idx]['answer'] + "\n" + "2." + self.forget_data[idx]['answer']
                    elif "nomask" in data_type:
                        answer_first = None
                        answer_last = "1. " + self.retain_data[retain_idx]['answer'] + "\n" + "2." + self.idk[rand_pos].strip()

                        
            elif "retain" in data_type:
                data = self.retain_data
                question = data[retain_idx]['question']
                answer = data[retain_idx]['answer']
            else:
                data = self.forget_data
                question = data[idx]['question']
                answer = data[idx]['answer']
                retain_question = self.retain_data[idx]['question'] 


            if "idk" in data_type:
                answer = self.idk[rand_pos].strip()



            
            if data_type == 'forget':
                converted_data = convert_raw_forget_data_to_model_format(self.tokenizer, self.max_length, question,
                                                                         answer, self.model_configs, mask=self.mask)
            elif "nomask_untargeted" in data_type:
                converted_data = convert_raw_forget_data_to_model_format(self.tokenizer, self.max_length, question,
                                                                      answer_last, self.model_configs, mask=False)                                         
            elif "pre" in data_type:
                converted_data = convert_raw_forget_data_to_model_format(self.tokenizer, self.max_length, question,
                                                                         answer, self.model_configs, prefilling=preflling_sentence)
            elif "+" in data_type:
                converted_data = convert_raw_forget_data_to_model_format(self.tokenizer, self.max_length, question,
                                                                      answer_last, self.model_configs,prefilling=answer_first)
            else:
                converted_data = convert_raw_forget_data_to_model_format(self.tokenizer, self.max_length, question,
                                                                         answer, self.model_configs)
            rets.append(converted_data)
        return rets
