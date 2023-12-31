from sentence_transformers import util
from constants import MIN_COS_SIM_VALUE

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# def vectorize(input: str):
#     inputList = [input]
#     inputList = [sentence.lower()\
#              .replace('br','')\
#              .replace('<',"")\
#              .replace(">","")\
#              .replace('\\',"")\
#              .replace('\/',"")\
#              for sentence in inputList]
#     embedding = model.encode(inputList)
#     return json.dumps(embedding.tolist())

# def vectorizeObj(input: str):
#     inputList = [input]
#     inputList = [sentence.lower()\
#              .replace('br','')\
#              .replace('<',"")\
#              .replace(">","")\
#              .replace('\\',"")\
#              .replace('\/',"")\
#              for sentence in inputList]
#     embedding = model.encode(inputList)
#     return embedding

def compare(comparerVector, targetVectors):
    cos_sims = []
    for targetVector in targetVectors:
        cos_sim = util.cos_sim(comparerVector, targetVector)
        cos_sims.append(cos_sim)
    
    winners = []

    index = 0
    for ar in cos_sims:
        for arr in ar:
            for i, each_val in enumerate(arr):
                value = each_val.item()
                if (value >= MIN_COS_SIM_VALUE()):
                    winners.append({ 'index': index, 'value': value})
            index += 1


    winners.sort(key=lambda x: x["value"], reverse=True)
    if (len(winners)>3):
        winners = winners[:3]

    return winners