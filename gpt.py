
from chat import ChatBot


queriesPrompt = """### Memories ###
{0}
### Instructions ###
Input: Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?
Output: 
"""

def getMemoryQueries(memories):
    memories = [x for x in memories]
    memories = "\n".join([str(i+1)+". "+x for i,x in enumerate(memories)])

    bot = ChatBot("")

    prompt = queriesPrompt.format(memories)
    #print('ChatGPT prompt:', prompt)

    queries = bot(prompt)
    #print('ChatGPT queries:', queries)

    queries = queries.split("\n")
    queries = [x[3:] for x in queries]
    return queries