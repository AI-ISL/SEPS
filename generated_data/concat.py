import faiss
import jsonlines
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
input_file_path = "all.jsonl"
output1 = 'forget_01_TRUE.jsonl'
output5 = 'forget_05_TRUE.jsonl'
output10 = 'forget_10_TRUE.jsonl'


lid = 0
with jsonlines.open(input_file_path, mode='r') as reader:
    lines = list(reader)


first400 = lines[:400].copy()
last3600 = lines[400:].copy()

sentences400 = [line['question'] for line in first400]
sentences3600 = [line['question'] for line in last3600]

embeddings400 = model.encode(sentences400, convert_to_tensor=False)  
embeddings3600 = model.encode(sentences3600, convert_to_tensor=False)  

print(f"Query Embeddings Shape: {embeddings400.shape}")
print(f"Database Embeddings Shape: {embeddings3600.shape}")


index = faiss.IndexFlatIP(768)
index.add(embeddings3600)
D, I = index.search(embeddings400, 3600)

for idx, sentence in enumerate(sentences400):
    for top_idx in list([1,5,1800,3600]):
        first400[idx][f'retain_question_top{top_idx}'] = last3600[I[idx][top_idx-1]]['question']
        first400[idx][f'retain_answer_top{top_idx}'] = last3600[I[idx][top_idx-1]]['answer']

with jsonlines.open(output1, mode='w') as writer:
    for line in first400[:40]:
        writer.write(line)

with jsonlines.open(output5, mode='w') as writer:
    for line in first400[:200]:
        writer.write(line)

with jsonlines.open(output10, mode='w') as writer:
    for line in first400:
        writer.write(line)

    