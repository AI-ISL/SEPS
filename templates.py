def single_eval_template(q, gt, o, tokenizer):
    out = f"""Evaluation Criteria
Evaluate whether the [OUTPUT] contains full information about the [Ground Truth] on a scale of 0 to 10.

If the [OUTPUT] contains all the information from the [Ground Truth], assign a score of 10.
If it contains none or is gibberish, assign a score of 0.
For partially correct responses, assign scores between 1 and 9 based on the amount of relevant information included.
Provide the score as a single number (0-10)

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

def single_eval_gpt_template(q, gt, o):
    out = f"""Evaluation Criteria
Evaluate whether the [OUTPUT] contains full information about the [Ground Truth] on a scale of 0 to 10.

If the [OUTPUT] contains all the information from the [Ground Truth], assign a score of 10.
If it contains none or is gibberish, assign a score of 0.
For partially correct responses, assign scores between 1 and 9 based on the amount of relevant information included.
Provide the score as a single number (0-10)

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

    return out

def double_eval_gpt_template(qf, qr, gtf, gtr, o):
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

    return out