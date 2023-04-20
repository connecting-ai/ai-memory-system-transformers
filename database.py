import datetime
import json
import numpy as np
import pymongo 
from constants import DB_NAME, MONGO_URL, COL_NAME
from vectorizer import compare, vectorizeObj

client = pymongo.MongoClient(MONGO_URL())


db = client[DB_NAME()]

col = db[COL_NAME()]

def addMemory(npcId, memory, timestamp, lastAccess, vector, importance, checker=False):
    memoryObject = {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }
    
    if checker == True:
        query = {
            "npcId": npcId,
            "memory": memory
        }

        if col.find_one(query) != None:
            print("memory already exists, updating old timestamp")
            memoryObject = col.find_one(query)
            memoryObject["timestamp"] = timestamp
            memoryObject["lastAccess"] = lastAccess
            memoryObject["vector"] = vector
            memoryObject["importance"] = importance
            col.update_one(query, {"$set": memoryObject})

            unixTimeNow = datetime.datetime.now().timestamp()
            memoryTime = memoryObject["lastAccess"]
            recency = unixTimeNow - memoryTime
            recency = recency * 0.99
            memoryObject["recency"] = recency
            return memoryObject
    
    print("adding new memory")
    col.insert_one(memoryObject)

    unixTimeNow = datetime.datetime.now().timestamp()
    memoryTime = memoryObject["lastAccess"]
    recency = unixTimeNow - memoryTime
    recency = recency * 0.99
    memoryObject["recency"] = recency

    return memoryObject

def getMemory(npcId, memory):
    query = {
        "npcId": npcId,
        "memory": memory
    }
    
    memory = col.find_one(query)
    
    unixTimeNow = datetime.datetime.now().timestamp()
    memoryTime = memory["lastAccess"]
    recency = unixTimeNow - memoryTime
    recency = recency * 0.99
    memory["recency"] = recency

    return memory

def getMemories(npcId):
    query = {
        "npcId": npcId
    }
    cursor = col.find(query)
    memories = [x for x in cursor]

    unixTimeNow = datetime.datetime.now().timestamp()
    for memory in memories:
        memoryTime = memory["lastAccess"]
        recency = unixTimeNow - memoryTime
        recency = recency * 0.99
        memory["recency"] = recency

    return memories

def getMemoriesShortedByLastAccess(npcId, max=100):
    query = {
        "npcId": npcId
    }
    cursor = col.find(query).sort("lastAccess", -1)
    arr = [x for x in cursor]
    print('dada', len(arr))
    for x in arr:
        x["lastAccess"] = datetime.datetime.fromtimestamp(x["lastAccess"])
    arr.sort(key=lambda x: x["lastAccess"], reverse=True)
    if (max>0 and len(arr)>max):
        arr = arr[:max]
    
    for x in arr:
        x["lastAccess"] = x["lastAccess"].timestamp()

    unixTimeNow = datetime.datetime.now().timestamp()
    for memory in arr:
        memoryTime = memory["lastAccess"]
        recency = unixTimeNow - memoryTime
        recency = recency * 0.99
        memory["recency"] = recency

    return arr

def getRelevantMemoriesFrom(queries, npcId):
    memories = getMemoriesShortedByLastAccess(npcId, max=100)
    print(len(memories))

    _vectors = []
    for memory in memories:
        _vectors.append(vectorizeObj(memory["memory"]))

    relevant = []
    for query in queries:
        vector = vectorizeObj(query)
        memory = getRelevantMemories(memories, _vectors, vector)
        for mem in memory:
            if mem not in relevant:
                relevant.append(mem)
    
    return relevant

def getRelevantMemories(memories, targetVectors, query):
    winner = compare(query, targetVectors)

    winMemory = []
    for win in winner:
        winMemory.append(memories[win["index"]])
    return winMemory    