
from chat import ChatBot


queriesPrompt = """### Memories ###
{0}
### Instructions ###
Input: Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?
Output: 
"""

def getMemoryQueries(memories):
    memories = [x["memory"] for x in memories]
    memories = "\n".join([str(i+1)+". "+x for i,x in enumerate(memories)])

    bot = ChatBot("")

    prompt = queriesPrompt.format(memories)
    queries = bot(prompt)
    queries = queries.split("\n")
    queries = [x[3:] for x in queries]
    return queries