import datetime

from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from typing import Dict, List, Optional, Tuple
from comparer import cosine_similarity
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

#uri = "mongodb+srv://zeref94:V7a2zauORu79GH4u@npc-memory.wyxvdjw.mongodb.net/?retryWrites=true&w=majority"
uri = "mongodb+srv://alex:o0uV7BkNFsRcmv5q@npc-memory.wyxvdjw.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
#check if db exists
embedding_size = 384

def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
    """Get the hours passed between two datetime objects."""
    #convert ref_time from unix timestamp to datetime
    ref_time = datetime.datetime.fromtimestamp(int(ref_time))
    return (time - ref_time).total_seconds() / 3600

class TimeWeightedVectorStoreRetriever_custom(TimeWeightedVectorStoreRetriever):
    def _get_combined_score(
            self,
            document: Document,
            vector_relevance: Optional[float],
            current_time: datetime,
        ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["timestamp"],
        )
        #Note: We can change the above 'last_accessed_at' above to 'created_at' to rank memory based on when it was created (rather than when it was last accessed in the Langchain default implementation)
        #https://github.com/hwchase17/langchain/blob/85dae78548ed0c11db06e9154c7eb4236a1ee246/langchain/retrievers/time_weighted_retriever.py#L119

        score = (1.0 - self.decay_rate) ** hours_passed
        # print(f'score contributed by time: {score}')
        for key in self.other_score_keys:
            if key in document.metadata:
                if key != 'importance':
                  score += document.metadata[key]
                else:
                  score += int(document.metadata[key])/10.
                #   print(f'score contributed by importance: {int(document.metadata[key])/10.}')

        if vector_relevance is not None:
            score += vector_relevance
            # print(f'score contributed by vector relevance: {vector_relevance}')

        # print(f'total score: {score}')
        # print('------------')
        return score                         
    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
        """Return documents that are salient to the query."""
        docs_and_scores: List[Tuple[Document, float]]

        #Note: Changed to `vectorstore.similarity_search` for usage with Chroma and Lance--->
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k = self.k
        )
        print(docs_and_scores)
        results = {}
        counter = 0
        for doc in docs_and_scores:
            fetched_doc = doc[0]
            relevance = doc[1]
            results[counter] = (fetched_doc, relevance)
            counter += 1
        return results
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.datetime.now()
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            for doc in self.memory_stream[-self.k :]
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = [doc[0] for doc in rescored_docs]
        return result
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            #if "lastAccess" not in doc.metadata:
            #    doc.metadata["lastAccess"] = current_time
            if "timestamp" not in doc.metadata:
                doc.metadata["timestamp"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time")
        if current_time is None:
            current_time = datetime.datetime.now()
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            #if "lastAccess" not in doc.metadata:
            #    doc.metadata["lastAccess"] = current_time
            if "timestamp" not in doc.metadata:
                doc.metadata["timestamp"] = current_time
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return await self.vectorstore.aadd_documents(dup_docs, **kwargs)

    def remove_document(self, document):
        self.memory_stream.remove(document)


db = client['npc_memory_db']

# Initialize Mongo collections if they don't exist
if 'base_memory' not in db.list_collection_names():
    db.create_collection('base_memory')

if 'relationship_memory' not in db.list_collection_names():
    db.create_collection('relationship_memory')

if 'plan_memory' not in db.list_collection_names():
    db.create_collection('plan_memory')

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
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

def deleteplan_memories(planId):
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