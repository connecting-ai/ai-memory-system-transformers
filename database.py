import datetime

from langchain.embeddings import HuggingFaceEmbeddings 

from comparer import cosine_similarity


import lancedb
db_init = lancedb.connect("./lancedb")

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

#uri = "mongodb+srv://zeref94:V7a2zauORu79GH4u@npc-memory.wyxvdjw.mongodb.net/?retryWrites=true&w=majority"
uri = "mongodb+srv://alex:o0uV7BkNFsRcmv5q@npc-memory.wyxvdjw.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
#check if db exists
embedding_size = 768

db = client['npc_memory_db']

# Initialize Mongo collections if they don't exist
if 'base_memory' not in db.list_collection_names():
    db.create_collection('base_memory')

if 'relationship_memory' not in db.list_collection_names():
    db.create_collection('relationship_memory')

if 'plan_memory' not in db.list_collection_names():
    db.create_collection('plan_memory')

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
# Initialize the vectorstore as empty

#### Base memory functions ####
def addBaseMemory(npcId, memory, timestamp, lastAccess, vector, importance, checker=False):
    collection = db['base_memory']
    doc = {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }
    collection.insert_one(doc)
    return doc


def getRelevantBaseMemoriesFrom(queries, npcId, max_memories=-1, top_k=1):
    collection = db['base_memory']
    relevant = []

    for query in queries:
        vector_query = embedding_model.embed_query(query)
        cursor = collection.find({"npcId": npcId, "vector": vector_query}).limit(top_k)

        for doc in cursor:
            memory = {
                "npcId": doc["npcId"],
                "memory": doc["memory"],
                "timestamp": doc["timestamp"],
                "lastAccess": doc["lastAccess"],
                "vector": doc["vector"],
                "importance": doc["importance"],
                "recency": datetime.datetime.now().timestamp() - doc["lastAccess"]
            }
            if memory not in relevant:
                relevant.append(memory)

    relevant.sort(key=lambda x: x["recency"], reverse=True)
    if max_memories > 0:
        relevant = relevant[:max_memories]

    return relevant


#### Relationship memory methods ####
def addRelationshipMemory(npcId, memory, timestamp, lastAccess, vector, importance):
    collection = db['relationship_memory']
    doc = {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }
    collection.insert_one(doc)
    return doc



def getRelevantRelationshipMemoriesFrom(queries, npcId, max_memories = -1, top_k=1):
    collection = db['relationship_memory']
    relevant = []

    for query in queries:

        vector_query = embedding_model.embed_query(query)
        cursor = collection.find({"npcId": npcId, "vector": vector_query}).limit(top_k)

        for doc in cursor:
            memory = {
                "npcId": doc["npcId"],
                "memory": doc["memory"],
                "timestamp": doc["timestamp"],
                "lastAccess": doc["lastAccess"],
                "vector": doc["vector"],
                "importance": doc["importance"],
                "recency": datetime.datetime.now().timestamp() - doc["lastAccess"]
            }
            if memory not in relevant:
                relevant.append(memory)

    relevant.sort(key=lambda x: x["recency"], reverse=True)
    if max_memories > 0:
        relevant = relevant[:max_memories]

    return relevant

def getRelevantPlanMemories(queries, npcId, max_memories = -1, threshold=0.8):
    collection = db['plan_memory']
    relevant = []

    for query in queries:
        vector_query = embedding_model.embed_query(query)

        # Search for memories associated with the given npcId
        cursor = collection.find({"npcId": npcId})

        for doc in cursor:
            vector_memory = doc["recalled_summary_vector"]
            similarity = cosine_similarity(vector_query, vector_memory)
                        
            # Filter out memories below the threshold
            if similarity >= threshold:
                memory = {
                    "npcId": doc["npcId"],
                    "recalled_summary": doc["recalled_summary"],
                    "planID": doc["planID"],
                    "superset_command_of_god_id": doc["superset_command_of_god_id"],
                    "intendedPeople": doc["intendedPeople"],
                    "intendedPeopleIDs": doc["intendedPeopleIDs"],
                    "routine_entries": doc["routine_entries"],
                    "plannedDate": doc["plannedDate"],
                    "timestamp": doc["timestamp"],
                    "lastAccess": doc["lastAccess"],
                    "recalled_summary_vector": doc["recalled_summary_vector"],
                    "importance": doc["importance"],
                    "recency": datetime.datetime.now().timestamp() - doc["lastAccess"]
                }
                
                if memory not in relevant:
                    relevant.append(memory)

    # Sort based on recency
    relevant.sort(key=lambda x: x["recency"], reverse=True)

    if max_memories > 0:
        relevant = relevant[:max_memories]

    return relevant

def delete_plan_memories(planId):
    collection = db['plan_memory']
    # Deleting all memories related to the given planID
    collection.delete_many({"planID": planId})

def get_document_from_plan_memory(npcId, pageContent):
    collection = db['plan_memory']
    
    # Finding the document using npcId and pageContent (recalled_summary in this case)
    doc = collection.find_one({"npcId": npcId, "recalled_summary": pageContent})

    return doc if doc else None

def addPlanMemory(npcId, recalled_summary, timestamp, lastAccess, superset_command_of_god_id, planID, intendedPeople, intendedPeopleIDs, routine_entries, importance, plannedDate, vector):
    collection = db['plan_memory']
    
    memory = {
        "npcId": npcId,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "recalled_summary": recalled_summary,
        "superset_command_of_god_id": superset_command_of_god_id,
        "planID": planID,
        "intendedPeople": intendedPeople,
        "intendedPeopleIDs": intendedPeopleIDs,
        "routine_entries": routine_entries,
        "importance": importance,
        "plannedDate": plannedDate,
        "recalled_summary_vector": vector
    }

    # Inserting the memory into the database
    inserted_id = collection.insert_one(memory).inserted_id

    return memory if inserted_id else None

def get_base_memories(npcId):
    collection = db['base_memory']  # Assuming the collection name is 'base_memory'. Adjust if necessary.
    
    # If you want to keep the original functionality of checking for the npcId existence, you'd do:
    npc_exists = collection.find_one({"npcId": npcId})
    if not npc_exists:
        return []
    
    cursor = collection.find({"npcId": npcId})
    memories = []

    for doc in cursor:
        timestamp = doc.get("timestamp", 0)
        lastAccess = doc.get("lastAccess", 0)
        importance = doc.get("importance", 0)
        memory_content = doc.get("memory", "")

        memories.append({
            "npcId": npcId,
            "timestamp": timestamp,
            "lastAccess": lastAccess,
            "importance": importance,
            "memory": memory_content
        })

    return memories

def get_relationship_memories(npcId):
    collection = db['relationship_memory']  # Assuming the collection name is 'relationship_memory'. Adjust if necessary.

    # If you want to keep the original functionality of checking for the npcId existence, you'd do:
    npc_exists = collection.find_one({"npcId": npcId})
    if not npc_exists:
        return []
    
    cursor = collection.find({"npcId": npcId})
    memories = []

    for doc in cursor:
        timestamp = doc.get("timestamp", 0)
        lastAccess = doc.get("lastAccess", 0)
        importance = doc.get("importance", 0)
        memory_content = doc.get("memory", "")

        memories.append({
            "npcId": npcId,
            "timestamp": timestamp,
            "lastAccess": lastAccess,
            "importance": importance,
            "memory": memory_content
        })

    return memories

def get_plan_memories(npcId):
    collection = db['plan_memory']  # Assuming the collection name is 'plan_memory'. Adjust if necessary.

    # If you want to keep the original functionality of checking for the npcId existence, you'd do:
    npc_exists = collection.find_one({"npcId": npcId})
    if not npc_exists:
        return []
    
    cursor = collection.find({"npcId": npcId})
    memories = []

    for doc in cursor:
        timestamp = doc.get("timestamp", 0)
        lastAccess = doc.get("lastAccess", 0)
        importance = doc.get("importance", 0)
        recalled_summary = doc.get("recalled_summary", "")
        superset_command_of_god_id = doc.get("superset_command_of_god_id", "")
        planID = doc.get("planID", "")
        intendedPeople = doc.get("intendedPeople", "")
        intendedPeopleIDs = doc.get("intendedPeopleIDs", "")
        routine_entries = doc.get("routine_entries", "")
        plannedDate = doc.get("plannedDate", "")
        recalled_summary_vector = doc.get("recalled_summary_vector", "")

        memories.append({
            "timestamp": timestamp,
            "lastAccess": lastAccess,
            "importance": importance,
            "recalled_summary": recalled_summary,
            "superset_command_of_god_id": superset_command_of_god_id,
            "planID": planID,
            "intendedPeople": intendedPeople,
            "intendedPeopleIDs": intendedPeopleIDs,
            "routine_entries": routine_entries,
            "plannedDate": plannedDate,
            "recalled_summary_vector": recalled_summary_vector
        })

    return memories