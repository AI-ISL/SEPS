import json

def check_duplicates(file_path):
    questions = set()
    answers = set()
    duplicates = []

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            question = data['question']
            answer = data['answer']
            if question in questions and answer in answers:
                duplicates.append((question, answer))
            else:
                questions.add(question)
                answers.add(answer)

    return duplicates

file_path = 'all.jsonl'
duplicated_questions = check_duplicates(file_path)
if duplicated_questions:
    print("Duplicated questions found:")
    for question in duplicated_questions:
        print(question)
else:
    print("No duplicated questions found.")
