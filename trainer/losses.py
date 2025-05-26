import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import AutoTokenizer

def get_loss(model, ref_model, inputs, loss_type, beta=0.1):
    # forget_loss
    if 'GA' in loss_type:
        forget_loss = ga_loss(model, inputs)
    elif 'NPO' in loss_type:
        forget_loss = npo_loss(model, ref_model, inputs, beta=beta)
    elif 'DPO' in loss_type:
        forget_loss = dpo_loss(model, ref_model, inputs, beta=beta)
    elif 'IDK' in loss_type:
        forget_loss = idk_loss(model, inputs)
    elif 'ME' in loss_type:
        forget_loss = me_loss(model, inputs)
    else: forget_loss = 0


    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    elif 'KL' in loss_type:
        regularization_loss = kl_loss(model, ref_model, inputs)
    elif 'AP' in loss_type:
        regularization_loss = ap_loss(model, inputs, beta=beta)
    else:
        regularization_loss = 0

    # mixed_loss
    if 'MPUT' in loss_type:
        regularization_loss += (mixed_fr_untargeted_kl_loss(model, ref_model, inputs) + mixed_rf_untargeted_kl_loss(model, ref_model, inputs))
    elif "MPT" in loss_type:
        regularization_loss += (gd_fr_nm_loss(model, inputs) + gd_rf_nm_loss(model, inputs))

    return forget_loss, regularization_loss

    


def ga_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss


def npo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss



def idk_loss(model, inputs):
    forget_idk_inputs = inputs[2]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def dpo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[0], inputs[2]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() * 2 / beta
    return loss


# Regularization Loss: AP
def ap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[1], inputs[3]
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = retain_idk_inputs

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)

    loss = get_batch_loss(outputs.logits, retain_labels)
    loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss



# Regularization Loss: KL
def kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, outputs.logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss


# Regularization Loss: GD
def gd_loss(model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def gd_fr_nm_loss(model, inputs):
    retain_inputs = inputs[4]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss



def gd_rf_nm_loss(model, inputs):
    retain_inputs = inputs[5]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def mixed_fr_untargeted_kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[6] 
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    loss = get_mixed_untargeted_loss1(
        outputs.logits, labels, input_ids, ref_logits=outputs_ref.logits
    )

    return loss

def mixed_rf_untargeted_kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[7]  
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)

    loss = get_mixed_untargeted_loss2(
        outputs.logits, labels, input_ids, attention_mask, ref_logits=outputs_ref.logits
    )

    return loss

def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def me_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=None, attention_mask=attention_mask)
    loss = get_me_loss(outputs.logits, labels)

    return loss


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction='none').sum(-1)  # (bs*(seq_len - 1))

    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss




def find_all_token_indices_batch1(input_ids):

    target_strings = ["[INST] "," [/INST]", "\n", "1."] 
    batch_indices = [] 
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    for batch_idx, batch in enumerate(input_ids):  
        batch_result = {}
        
        for j, target in enumerate(target_strings):
            target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"][1:]
            target_indices = []
            for i in range(len(batch) - len(target_token_ids) + 1):
                if batch[i:i+len(target_token_ids)].tolist() == target_token_ids:
                    target_indices.append(i)  
            batch_result[target] = target_indices if target_indices else -1  

        batch_indices.append(batch_result)

    return batch_indices
def get_mixed_untargeted_loss1(logits, labels, input_ids, ref_logits):

    indices = find_all_token_indices_batch1(input_ids)
    num_labels = logits.shape[-1]

    labels = labels[:, 1:].clone()  
    logits = logits[:, :-1, :]  
    ref_logits = ref_logits[:, :-1, :]  

    batch_size, seq_len_minus_one = labels.shape
    flat_logits = logits.reshape(-1, num_labels)  
    flat_ref_logits = ref_logits.reshape(-1, num_labels)  

    uniform_dist = torch.full_like(flat_ref_logits, 1.0 / num_labels)

    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]
        if " [/INST]" in batch_indices and "\n" in batch_indices:
            one_positions = batch_indices[" [/INST]"]
            newline_positions = batch_indices["\n"]
            start_positions = batch_indices["[INST] "]
            for i, pos in enumerate(start_positions):
                if i==0:
                    ranges.append((pos, newline_positions[0]))
            for i, pos in enumerate(one_positions):
                if i == 0:  
                    ranges.append((pos, newline_positions[1]))


        for i, (start, end) in enumerate(ranges):  
            if i==0:
                for idx in range(start+5, end): 
                    flat_idx = batch_idx * seq_len_minus_one + idx
                    if flat_idx < flat_ref_logits.size(0): 
                        flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
                    else:
                        print(f"Warning: flat_idx {flat_idx} is out of range")
            elif i==1:
                for idx in range(start+6, end):  
                    flat_idx = batch_idx * seq_len_minus_one + idx
                    if flat_idx < flat_ref_logits.size(0):  
                        flat_ref_logits[flat_idx] = uniform_dist[flat_idx]
                    else:
                        print(f"Warning: flat_idx {flat_idx} is out of range")

    soft_outputs = F.softmax(flat_logits, dim=-1) 
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)  

    soft_outputs = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div *loss_mask

    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss




def get_mixed_untargeted_loss2(logits, labels, input_ids, attention_mask, ref_logits):

    indices = find_all_token_indices_batch2(input_ids, attention_mask)
    num_labels = logits.shape[-1]

    labels = labels[:, 1:].clone() 
    logits = logits[:, :-1, :] 
    ref_logits = ref_logits[:, :-1, :] 

    batch_size, seq_len_minus_one = labels.shape
    flat_logits = logits.reshape(-1, num_labels)  
    flat_ref_logits = ref_logits.reshape(-1, num_labels)

    uniform_dist = torch.full_like(flat_ref_logits, 1.0 / num_labels)

    for batch_idx in range(batch_size):
        ranges = []
        batch_indices = indices[batch_idx]
        one_positions = batch_indices["\n"]
        newline_positions1 = batch_indices[" [/INST]"]
        newline_positions2 = batch_indices["valid_end"]
        for i, pos in enumerate(one_positions):
            if i ==0:
                ranges.append((pos,newline_positions1[0]))
            elif i==1:
                ranges.append((pos,newline_positions2))
        

        for start, end in ranges:
            for idx in range(start+3, end):
                flat_idx = batch_idx * seq_len_minus_one + idx
                flat_ref_logits[flat_idx] = uniform_dist[flat_idx]

    soft_outputs = F.softmax(flat_logits, dim=-1) 
    soft_ref_outputs = F.softmax(flat_ref_logits, dim=-1)  

    soft_outputs = torch.clamp(soft_outputs, min=1e-12, max=1.0)
    soft_ref_outputs = torch.clamp(soft_ref_outputs, min=1e-12, max=1.0)

    kl_div = (soft_outputs * (torch.log(soft_outputs) - torch.log(soft_ref_outputs))).sum(-1)

    loss_mask = (labels != -100).view(-1)
    masked_kl_div = kl_div *loss_mask

    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss


def find_all_token_indices_batch2(input_ids, attention_mask):

    target_strings = ["\n", " [/INST]"] 
    batch_indices = []  
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    pad_token_id = tokenizer.eos_token_id
    for batch_idx, batch in enumerate(input_ids):
        batch_result = {}

        if attention_mask is not None:
            valid_end = attention_mask[batch_idx].nonzero(as_tuple=True)[0][-1].item() + 1
        else:
            valid_end = (batch != pad_token_id).nonzero(as_tuple=True)[0][-1].item() + 1
        batch_result["valid_end"] = valid_end 
        for i, target in enumerate(target_strings):
            target_token_ids = tokenizer(target, add_special_tokens=False)["input_ids"][1:]

            target_indices = []
            for i in range(len(batch) - len(target_token_ids) + 1):
                if batch[i:i+len(target_token_ids)].tolist() == target_token_ids:
                    target_indices.append(i)  
            batch_result[target] = target_indices if target_indices else -1  


        batch_indices.append(batch_result) 


    return batch_indices
