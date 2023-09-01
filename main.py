import traceback
from envReader import read, getValue
read()

import asyncio
import os
from dataclasses import dataclass
from fastapi import FastAPI, Request, BackgroundTasks
import logging
import os
from time import time
from fastapi.middleware.cors import CORSMiddleware
from database import addPlanMemory, deleteplan_memories, embedding_model, get_lastk_base_memories, get_plan_memories, get_relationship_memories, getRelevantPlanMemories, addBaseMemory, getRelevantBaseMemoriesFrom, addRelationshipMemory, getRelevantRelationshipMemoriesFrom
from gpt import getMemoryQueries, getMemoryAnswers
from pydantic import BaseModel

logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO)

class MemoryData():
    npcId: str
    memory: str
    timestamp: float
    importance: str
    addOnlyIfUnique: bool = False
    
class MassMemoryData(BaseModel):
    memories: list

class AddPlanMemoryData(BaseModel):
    npcId: str
    recalled_summary: str
    superset_command_of_god_id: str
    planID: str
    intendedPeople: list
    intendedPeopleIDs: list
    routine_entries: list
    timestamp: float
    importance: int
    plannedDate: float

class AddInMemoryData(BaseModel):
    npcId: str
    memory: str
    timestamp: float
    importance: str
    addOnlyIfUnique: bool = False
    
#logging.basicConfig(level=logging.INFO, format="%(levelname)-9s %(asctime)s - %(name)s - %(message)s")
#LOGGER = logging.getLogger(__name__)

EXPERIMENTS_BASE_DIR = "/experiments/"
QUERY_BUFFER = {}

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loop = asyncio.get_event_loop()

@dataclass
class Query():
    query_name: str
    query_sequence: str
    query_type: int
    input: str
    npcId: str
    memory: str
    timestamp: float
    vector: str
    importance: str
    checker: bool
    memories: str
    memory_query: str
    top_k: int = 1
    result: str = ""
    experiment_id: str = None
    status: str = "pending"

    def __post_init__(self):
        self.experiment_id = str(time())
        self.experiment_dir = os.path.join(EXPERIMENTS_BASE_DIR, self.experiment_id)

@app.get("/reflection")
async def relevant_memories(request: Request, npcId: str, last_k:int = 100 , top_ns:int = 3, top_k: int = 5):
    #last_k: number of memories to consider for reflection
    #top_ns: number of salient questions generated for the last_k memories
    #top_k: number of memories retreived for each salient question
    memories = []
    _memories = get_lastk_base_memories(npcId, k = last_k)
    for mem in _memories:
        memories.append(mem)
    # print(memories, type(memories[0]))

    queries = getMemoryQueries(memories, top_ns)
    # print('=========', queries[0])
    relevantMemories = getRelevantBaseMemoriesFrom(queries, npcId, top_k = top_k)

    relevantMemoriesString = []
    for memory in relevantMemories:
        relevantMemoriesString.append(memory["memory"])

    answers = getMemoryAnswers(queries ,relevantMemoriesString, top_ns)

    #To complete the reflection, first run this GET route, then add all the entries in `answers` to the memory with `/add_in_memory` route

    return answers
    # return list(dict.fromkeys(relevantMemoriesString)) 

@app.post("/reflection")
async def post_reflection(request: Request, npcId: str, timestamp: float):
    body = await request.json()
    memories = []
    for reflection in body:
        # print(reflection)
        memory = reflection["text"] + "(Because of "
        for reason in reflection["references"]:
            memory += reason + ", "
        memory = memory[:-2]
        memory += ")"
        # print(memory)
        #vector = embedding_model.embed_query(memory)
        # print("vector done")
        returnedMemory = addBaseMemory(npcId, memory, timestamp, timestamp, -1)
        if "_id" in returnedMemory:
            del returnedMemory["_id"]
            
        memories.append(returnedMemory)

    # print(memories)
    return memories

@app.get("/query")
async def query(request: Request,background_tasks: BackgroundTasks, npcId: str, input: str, top_k: int = 1):
    res = getRelevantBaseMemoriesFrom([input], npcId, top_k = top_k)
    return res

@app.get("/memories")
async def memories(request: Request, npcId: str):
    memories = get_lastk_base_memories(npcId, k = 100)
    return memories

@app.post("/add_in_memory")
async def add_in_memory(request: Request,background_tasks: BackgroundTasks, data: AddInMemoryData):
    query = Query(query_name="add_in_memory", query_sequence=1, input="", vector = None, query_type=2, npcId=data.npcId, memory=data.memory, timestamp=data.timestamp, importance=data.importance, checker=data.addOnlyIfUnique, memories=[], memory_query=None)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    return {"id": query.experiment_id}

@app.post("/mass_add_in_memory")
async def mass_add_in_memory(request: Request,background_tasks: BackgroundTasks, data: MassMemoryData):
    for memory in data.memories:
        memory['vector'] = embedding_model.embed_query(memory['memory'])
    query = Query(query_name="test", query_sequence=5, input=input, query_type=1, npcId="", memory="", timestamp=0, importance="", checker=False, memories=data.memories, vector=None, memory_query=None)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    return {"id": query.experiment_id}

@app.get("/vectorize")
async def root(request: Request, background_tasks: BackgroundTasks, input: str):
    query = Query(query_name="test", query_sequence=5, input=input, query_type=0, npcId="", memory="", timestamp=0, importance="", checker=False, vector=None, memory_query=None)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    return {"id": query.experiment_id}

@app.get("/result")
async def result(request: Request, query_id: str):
    try:
        if query_id in QUERY_BUFFER:
            if QUERY_BUFFER[query_id] == None:
                return {"status": "finished"}
            if QUERY_BUFFER[query_id].status == "done":
                if (QUERY_BUFFER[query_id].query_type == 0):
                    return { 'vector': QUERY_BUFFER[query_id].result }
                elif (QUERY_BUFFER[query_id].query_type == 1):
                    res = QUERY_BUFFER[query_id].result
                    for memory in res:
                        if "_id" in memory:
                            del memory["_id"]
                    del QUERY_BUFFER[query_id]
                    return res
                elif (QUERY_BUFFER[query_id].query_type == 3):
                    res = QUERY_BUFFER[query_id].result
                    result = []
                    print(res)
                    for memory in res:
                        if "_id" in memory:
                            del memory["_id"]
                        result.append(memory["memory"])
                    del QUERY_BUFFER[query_id]
                    return result
                else:
                    res = QUERY_BUFFER[query_id].result
                    if "_id" in res:
                        del res["_id"]
                    del QUERY_BUFFER[query_id]
                    return res
                del QUERY_BUFFER[query_id]
                return {}
            return {"status": QUERY_BUFFER[query_id].status}
        else:
            return {"status": "finished"}
    except Exception as e:
        print(e)
        return {"status": "finished"}

@app.get("/relationship_query")
async def relationship_query(request: Request, npcId: str, input: str, max_memories = -1, top_k: int = 1):
    res = getRelevantRelationshipMemoriesFrom([input], npcId, max_memories=max_memories, top_k=top_k)
    return res

@app.post("/relationship_add_in_memory")
async def relationship_add_in_memory(request: Request,background_tasks: BackgroundTasks, data: MassMemoryData):
    res = []
    for memory in data.memories:
        addOnlyIfUnique = False
        if 'addOnlyIfUnique' in memory:
            addOnlyIfUnique = memory['addOnlyIfUnique']
        mem = addRelationshipMemory(memory['npcId'], memory['memory'], memory['timestamp'], memory['importance'])
        if "_id" in mem:
            del mem["_id"]
        res.append(mem)
    return res

@app.get("/relationship_memories")
async def relationship_memories(request: Request, npcId: str):
    memories = get_relationship_memories(npcId)
    return memories

@app.get("/plan_memories")
async def plan_memories(request: Request, npcId: str):
    memories = get_plan_memories(npcId)
    return memories

@app.post("/add_in_plan_memory")
async def add_plan_memory(request: Request,background_tasks: BackgroundTasks, data: AddPlanMemoryData):
    memory = addPlanMemory(data.npcId, data.recalled_summary, data.timestamp, data.superset_command_of_god_id, data.planID, data.intendedPeople, data.intendedPeopleIDs, data.routine_entries, data.importance, data.plannedDate)
    if "_id" in memory:
        del memory["_id"]
    return memory

@app.get("/delete_plan_memories")
async def delete_plan_memories(request: Request,  planID: str):
    deleteplan_memories(planID)
    return True

@app.get("/plan_query")
async def plan_query(request: Request, npcId: str, input: str, max_memories = -1, threshold: float = 0.8):
    res = getRelevantPlanMemories([input], npcId, max_memories=max_memories, threshold=threshold)
    return res

def process(query):
    try:
        res = None
        if (query.query_type == 0):
            res = embedding_model.embed_query(query.input)
        elif (query.query_type == 1):
            res = []
            for memory in query.memories:
                vector = memory['vector']
                addOnlyIfUnique = False
                if 'addOnlyIfUnique' in memory:
                    addOnlyIfUnique = memory['addOnlyIfUnique']
                newMemory = addBaseMemory(memory['npcId'], memory['memory'], memory['timestamp'], vector, memory['importance'], addOnlyIfUnique)
                if not newMemory is None:
                    res.append(newMemory)
        elif (query.query_type == 2):
            query.vector = embedding_model.embed_query(query.memory)
            res = addBaseMemory(query.npcId, query.memory, query.timestamp, query.importance, query.checker)
        elif (query.query_type == 3):
            memories = getRelevantBaseMemoriesFrom([query.memory_query], query.npcId, top_k = query.top_k)
            res = memories
        else:
            print("No Query Type was present")
            
        QUERY_BUFFER[query.experiment_id].result = res
        QUERY_BUFFER[query.experiment_id].status = "done"
    except Exception as e:
            print("===============")
            print(e)
            print(traceback.format_exc())
            print("===============")

@app.get("/backlog/")
def return_backlog():
    return {f"return_backlog - Currently {len(QUERY_BUFFER)} jobs in the backlog."}

@app.get('/healthz', status_code=200, response_model=dict)
async def healthz():
    """"
    Health check endpoint required for Kubernetes LB periodic pings
    """
    return {'status': 'OK'}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8080, workers=2)
