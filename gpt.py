
from chat import ChatBot


queriesPrompt = """### Memories ###
{0}
### Instructions ###
Input: Given only the information above, what are {1} most salient high-level questions we can answer about the subjects in the statements?
Output: 
"""

questionPrompt = """### Questions ###
{0}
### Relevant context ###
{1}

Input: Given only and all the relevant context above, answer all the {2} questions given above elaborately, separate each answer by a new line, do not index the answers.
Output: 
"""

def getMemoryQueries(memories, top_ns=3):
    memories = [x['memory'] for x in memories]
    memories = "\n".join([str(i+1)+". "+x for i,x in enumerate(memories)])

    bot = ChatBot("")

    prompt = queriesPrompt.format(memories, top_ns)
    #print('ChatGPT prompt:', prompt)

    queries = bot(prompt)
    #print('ChatGPT queries:', queries)

    queries = queries.split("\n")
    queries = [x[3:] for x in queries]
    return queries

def getMemoryAnswers(queries, context, top_ns=3):
    queries_pmpt = "\n".join([str(i+1)+" "+x for i,x in enumerate(queries)])
    context_pmpt = "\n".join([str(i+1)+". "+x for i,x in enumerate(context)])

    bot = ChatBot("")

    prompt = questionPrompt.format(queries_pmpt, context_pmpt, top_ns)
    #print('ChatGPT prompt:', prompt)

    queries = bot(prompt)
    #print('ChatGPT queries:', queries)

    answers = queries.split("\n")
    # answers = [x[3:] for x in queries]
    return answers