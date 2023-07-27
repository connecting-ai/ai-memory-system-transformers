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
from database import embedding_model, getRelationshipMemoriesRetrieved, npcID_to_base_retriever, addBaseMemory, getRelevantBaseMemoriesFrom, addRelationshipMemory, getRelevantRelationshipMemoriesFrom
from gpt import getMemoryQueries, getMemoryAnswers
from pydantic import BaseModel
import atexit
# from vectorizer import vectorize

def exit_handler():
    print("saving memories")
    saveMemories()

atexit.register(exit_handler)

class MemoryData():
    npcId: str
    memory: str
    timestamp: float
    lastAccess: float
    importance: str
    addOnlyIfUnique: bool = False
    
class MassMemoryData(BaseModel):
    memories: list
    
class AddInMemoryData(BaseModel):
    npcId: str
    memory: str
    timestamp: float
    lastAccess: float
    importance: str
    addOnlyIfUnique: bool = False
    
logging.basicConfig(level=logging.INFO, format="%(levelname)-9s %(asctime)s - %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

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
    lastAccess: float
    vector: str
    importance: str
    checker: bool
    memories: str
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
    for x in npcID_to_base_retriever[npcId].memory_stream[-last_k:]:
        memories.append(x.page_content)
    # print(memories, type(memories[0]))

    queries = getMemoryQueries(memories, top_ns)
    # print('=========', queries[0])
    relevantMemories = getRelevantBaseMemoriesFrom(queries, npcId, top_k)

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
        vector = embedding_model.embed_query(memory)
        # print("vector done")
        returnedMemory = addBaseMemory(npcId, memory, timestamp, timestamp, vector, -1)
        if "_id" in returnedMemory:
            del returnedMemory["_id"]
            
        memories.append(returnedMemory)

    # print(memories)
    return memories

@app.get("/query")
async def query(request: Request, npcId: str, input: str, top_k: int = 1):
    memories = getRelevantBaseMemoriesFrom([input], npcId, top_k)
    res = []
    for memory in memories:
        res.append(memory["memory"])

    return res

@app.get("/memories")
async def memories(request: Request, npcId: str):
    if npcId not in npcID_to_base_retriever.keys():
        return []
    memories = []
    for x in npcID_to_base_retriever[npcId].memory_stream:
        timestamp = 0
        lastAccess = 0
        importance = 0

        for key, value in x.metadata.items():
            if "timestamp" in key.lower():
                timestamp = value
            elif "lastaccess" in key.lower():
                lastAccess = value
            elif "importance" in key.lower():
                importance = value

        memories.append({
                "npcId": npcId,
                "timestamp":timestamp,
                "lastAccess":lastAccess,
                "importance":importance,
                "memory":x.page_content
            })
        
    return memories

@app.post("/add_in_memory")
async def add_in_memory(request: Request,background_tasks: BackgroundTasks, data: AddInMemoryData):
    query = Query(query_name="add_in_memory", query_sequence=1, input="", vector = None, query_type=2, npcId=data.npcId, memory=data.memory, timestamp=data.timestamp, lastAccess=data.lastAccess, importance=data.importance, checker=data.addOnlyIfUnique, memories=[])
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    return {"id": query.experiment_id}

@app.post("/mass_add_in_memory")
async def mass_add_in_memory(request: Request,background_tasks: BackgroundTasks, data: MassMemoryData):
    query = Query(query_name="test", query_sequence=5, input=input, query_type=1, npcId="", memory="", timestamp=0, lastAccess=0, importance="", checker=False, memories=data.memories, vector=None)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    return {"id": query.experiment_id}

@app.get("/vectorize")
async def root(request: Request, background_tasks: BackgroundTasks, input: str):
    query = Query(query_name="test", query_sequence=5, input=input, query_type=0, npcId="", memory="", timestamp=0, lastAccess=0, importance="", checker=False, vector=None)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    return {"id": query.experiment_id}

@app.get("/result")
async def result(request: Request, query_id: str):
    if query_id in QUERY_BUFFER:
        if QUERY_BUFFER[query_id].status == "done":
            resp = {}
            if (QUERY_BUFFER[query_id].query_type == 0):
                resp = { 'vector': QUERY_BUFFER[query_id].result }
            elif (QUERY_BUFFER[query_id].query_type == 1):
                res = QUERY_BUFFER[query_id].result
                for memory in res:
                    if "_id" in memory:
                        del memory["_id"]
                del QUERY_BUFFER[query_id]
                return res
            else:
                res = QUERY_BUFFER[query_id].result
                if "_id" in res:
                    del res["_id"]
                return res
            del QUERY_BUFFER[query_id]
            return resp
        return {"status": QUERY_BUFFER[query_id].status}
    else:
        return {"status": "finished"}

@app.get("/relationship_query")
async def relationship_query(request: Request, npcId: str, input: str, max_memories = -1, top_k: int = 1):
    res = getRelevantRelationshipMemoriesFrom([input], npcId, max_memories=max_memories, top_k=top_k)
    return res

@app.post("/relationship_add_in_memory")
async def relationship_add_in_memory(request: Request,background_tasks: BackgroundTasks, data: MassMemoryData):
    res = []
    for memory in data.memories:
        vector = embedding_model.embed_query(memory['memory'])
        addOnlyIfUnique = False
        if 'addOnlyIfUnique' in memory:
            addOnlyIfUnique = memory['addOnlyIfUnique']
        mem = addRelationshipMemory(memory['npcId'], memory['memory'], memory['timestamp'], memory['lastAccess'], vector, memory['importance'], addOnlyIfUnique)
        if "id" in mem:
            del mem["id"]
        res.append(mem)
    return res

@app.get("/relationship_memories")
async def relationship_memories(request: Request, npcId: str):
    retriever = getRelationshipMemoriesRetrieved()[npcId]
    memories = []
    for x in retriever.memory_stream:
        timestamp = 0
        lastAccess = 0
        importance = 0

        for key, value in x.metadata.items():
            if "timestamp" in key.lower():
                timestamp = value
            elif "lastaccess" in key.lower():
                lastAccess = value
            elif "importance" in key.lower():
                importance = value

        memories.append({
                "npcId": npcId,
                "timestamp":timestamp,
                "lastAccess":lastAccess,
                "importance":importance,
                "memory":x.page_content
            })
    
    return memories

def process(query):
    res = None
    if (query.query_type == 0):
        res = embedding_model.embed_query(query.input)
    elif (query.query_type == 1):
        res = []
        for memory in query.memories:
            print("calculating vector for:", memory['memory'])
            vector = None
            try:
                vector = embedding_model.embed_query(memory['memory'])
            except:
                print("could not calculate vector for:", memory['memory'])
                continue
            print("vector calculated")
            print("got vector: ", vector, "for:", memory['memory'])
            addOnlyIfUnique = False
            if 'addOnlyIfUnique' in memory:
                addOnlyIfUnique = memory['addOnlyIfUnique']
            newMemory = addBaseMemory(memory['npcId'], memory['memory'], memory['timestamp'], memory['lastAccess'], vector, memory['importance'], addOnlyIfUnique)
            res.append(newMemory)
    elif (query.query_type == 2):
        query.vector = embedding_model.embed_query(query.memory)
        res = addBaseMemory(query.npcId, query.memory, query.timestamp, query.lastAccess, query.vector, query.importance, query.checker)
    else:
        print("No Query Type was present")
        
    QUERY_BUFFER[query.experiment_id].result = res
    QUERY_BUFFER[query.experiment_id].status = "done"

    #if (query.query_type == 1 or query.query_type == 2):
    #    saveMemories()

def saveMemories():
    import _pickle as cPickle
    import bz2
    #Save the retriever to disk after every memory addition
    try:
        with bz2.BZ2File("/db/retreiver.pbz2", "w") as f: 
            cPickle.dump(npcID_to_base_retriever, f)
    except Exception:
        pass

@app.get("/backlog/")
def return_backlog():
    return {f"return_backlog - Currently {len(QUERY_BUFFER)} jobs in the backlog."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(getValue("PORT")))
