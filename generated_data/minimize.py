import jsonlines

for forget in ['forget01', 'forget05', 'forget10']:
    with jsonlines.open(f"{forget}_all_TRUE.jsonl", mode='r') as reader:
        lines = list(reader)
    
    with jsonlines.open(f'{forget}_TRUE.jsonl', mode='w') as writer:
        for line in lines:
            new_line = {}
            new_line['question'] = line['question']
            new_line['answer'] = line['answer']
            new_line['retain_question'] = line['retain_question_top1800']
            new_line['retain_answer'] = line['retain_answer_top1800']

            writer.write(new_line)