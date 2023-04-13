from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def vectorize(input: str):
    inputList = [input]
    embedding = model.encode(inputList)
    return json.dumps(embedding.tolist())