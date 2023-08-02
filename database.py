import datetime

from langchain.embeddings import HuggingFaceEmbeddings 

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document

from typing import Dict, List, Optional, Tuple

import os
import getpass
from pymongo import MongoClient

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://zeref94:V7a2zauORu79GH4u@npc-memory.wyxvdjw.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['vector-memories']

def _get_hours_passed(time: datetime, ref_time: datetime) -> float:
    """Get the hours passed between two datetime objects."""
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
            document.metadata["last_accessed_at"],
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
        docs_and_scores = self.vectorstore.similarity_search(
            query, **self.search_kwargs
        )
        results = {}
        for doc in docs_and_scores:
            fetched_doc = doc
            relevance = doc.metadata['score']
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = fetched_doc.metadata["buffer_idx"]
                doc = self.memory_stream[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

model_name = "intfloat/e5-base-v2"
model_kwargs = {'device': 'cpu'}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs)
# Initialize the vectorstore as empty
embedding_size = 768

npcID_to_base_retriever = {}
npcID_to_relationship_retriever = {}
npcID_to_plan_retriever = {}

def getRelationshipMemoriesRetrieved():
    return npcID_to_relationship_retriever

def getPlanMemoriesRetrieved():
    return npcID_to_plan_retriever

#### Base memory functions ####
def addBaseMemory(npcId, memory, timestamp, lastAccess, vector, importance, checker=False):
    try:
        if (vector == None):
            print("empty vector")
            return None
        
        #check if vector is a numpy array
        if (not type(vector) == list):
            vector = vector.tolist()

        if not db.list_collection_names(filter={'name': npcId}):
            # If the npcId collection does not exist, create it
            db.create_collection(npcId)

            # And create an initial document
            db[npcId].insert_one({
                "vector": embedding_model.embed_query("<NULL ENTRY>"),  # Convert to list for JSON serialization
                "text": "<NULL ENTRY>",
                "timestamp": timestamp,
                "lastAccess": lastAccess,
                "importance": importance
            })

        # Create the memory object and insert it into the collection
        memoryObject = {
            "timestamp": timestamp,
            "lastAccess": lastAccess,
            "vector": vector,  # Convert to list for JSON serialization
            "importance": importance
        }

        db[npcId].insert_one({
            "page_content": memory, 
            "metadata": memoryObject
        })

        return {
            "npcId": npcId,
            "memory": memory,
            "timestamp": timestamp,
            "lastAccess": lastAccess,
            "vector": vector,  # Convert to list for JSON serialization
            "importance": importance,
            "recency": datetime.datetime.now().timestamp() - memoryObject["lastAccess"]
        }
    except Exception as e:
        print(e)


def getRelevantBaseMemoriesFrom(queries, npcId, max_memories = -1, top_k=1):
    if npcId not in npcID_to_base_retriever.keys():
        return []

    retriever = npcID_to_base_retriever[npcId]
    relevant = []

    for query in queries:
        # vector = embedding_model.embed_query(query)
        retriever.k = top_k
        retrieved_docs = retriever.get_relevant_documents(query)
        retriever.k = 1

        for doc in retrieved_docs:
            timestamp = 0
            lastAccess = 0
            importance = 0
            vector = ""

            for key, value in doc.metadata.items():
                if "timestamp" in key.lower():
                    timestamp = value
                elif "lastaccess" in key.lower():
                    lastAccess = value
                elif "importance" in key.lower():
                    importance = value
                elif "vector" in key.lower():
                    vector = value
                    
            memory = {
                "npcId": npcId,
                "memory": doc.page_content,
                "timestamp": timestamp,
                "lastAccess": lastAccess,
                "vector": vector,
                "importance": importance,
                "recency": datetime.datetime.now().timestamp() - lastAccess
            }
            if memory not in relevant:
                relevant.append(memory)
                
            #sort them based on recency
            relevant.sort(key=lambda x: x["recency"], reverse=True)
            if max_memories>0:
                relevant = relevant[:max_memories]

    return relevant

def addRelationshipMemory(npcId, memory, timestamp, lastAccess, vector, importance, checker=False):
    if not db.list_collection_names(filter={'name': npcId + "_relationship"}):
        # If the npcId collection does not exist, create it
        db.create_collection(npcId + "_relationship")

        # And create an initial document
        db[npcId].insert_one({
            "vector": embedding_model.embed_query("<NULL ENTRY>"),  # Convert to list for JSON serialization
            "text": "<NULL ENTRY>",
            "timestamp": timestamp,
            "lastAccess": lastAccess,
            "importance": importance
        })

    # Create the memory object and insert it into the collection
    memoryObject = {
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,  # Convert to list for JSON serialization
        "importance": importance
    }

    db[npcId + "_relationship"].insert_one({
        "page_content": memory, 
        "metadata": memoryObject
    })

    return {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,  # Convert to list for JSON serialization
        "importance": importance,
        "recency": datetime.datetime.now().timestamp() - memoryObject["lastAccess"]
    }



def getRelevantRelationshipMemoriesFrom(queries, npcId, max_memories = -1, top_k=1):
    if npcId not in npcID_to_relationship_retriever.keys():
        return []

    retriever = npcID_to_relationship_retriever[npcId + "_relationship"]
    relevant = []

    for query in queries:
        # vector = embedding_model.embed_query(query)
        retriever.k = top_k
        retrieved_docs = retriever.get_relevant_documents(query)
        retriever.k = 1

        for doc in retrieved_docs:
            timestamp = 0
            lastAccess = 0
            importance = 0
            vector = ""

            for key, value in doc.metadata.items():
                if "timestamp" in key.lower():
                    timestamp = value
                elif "lastaccess" in key.lower():
                    lastAccess = value
                elif "importance" in key.lower():
                    importance = value
                elif "vector" in key.lower():
                    vector = value
                    
            memory = {
                "npcId": npcId,
                "memory": doc.page_content,
                "timestamp": timestamp,
                "lastAccess": lastAccess,
                "vector": vector,
                "importance": importance,
                "recency": datetime.datetime.now().timestamp() - lastAccess
            }
            if memory not in relevant:
                relevant.append(memory)
                
            #sort them based on recency
            relevant.sort(key=lambda x: x["recency"], reverse=True)
            if max_memories>0:
                relevant = relevant[:max_memories]

    return relevant

def getRelevantRelationshipMemoriesFrom(queries, npcId, max_memories = -1, top_k=1):
    if npcId not in npcID_to_plan_retriever.keys():
        return []

    retriever = npcID_to_plan_retriever[npcId + "_plan"]
    relevant = []

    for query in queries:
        # vector = embedding_model.embed_query(query)
        retriever.k = top_k
        retrieved_docs = retriever.get_relevant_documents(query)
        retriever.k = 1

        for doc in retrieved_docs:
            timestamp = 0
            importance = 0
            vector = ""
            recalled_summary = ""
            superset_command_of_god_id = ""
            planID = ""
            intendedPeople = []
            routine_entries = []

            for key, value in doc.metadata.items():
                if "timestamp" in key.lower():
                    timestamp = value
                elif "importance" in key.lower():
                    importance = value
                elif "vector" in key.lower():
                    vector = value
                elif "recalled_summary" in key.lower():
                    recalled_summary = value
                elif "superset_command_of_god_id" in key.lower():
                    superset_command_of_god_id = value
                elif "planID" in key.lower():
                    planID = value
                elif "intendedPeople" in key.lower():
                    intendedPeople = value
                elif "routine_entries" in key.lower():
                    routine_entries = value
                    
            memory = {
                "npcId": npcId,
                "timestamp": timestamp,
                "recalled_summary": recalled_summary,
                "superset_command_of_god_id": superset_command_of_god_id,
                "planID": planID,
                "intendedPeople": intendedPeople,
                "routine_entries": routine_entries,
                "vector": vector,
                "importance": importance,
                "recency": datetime.datetime.now().timestamp() - timestamp
            }
            if memory not in relevant:
                relevant.append(memory)
                
            #sort them based on recency
            relevant.sort(key=lambda x: x["recency"], reverse=True)
            if max_memories>0:
                relevant = relevant[:max_memories]

    return relevant

def addPlanMemory(npcId, recalled_summary, timestamp, superset_command_of_god_id, planID, intendedPeople, routine_entries, importance, vector):
    print("lala", npcId)
    print(db)
    print("exists:", db.list_collection_names(filter={'name': npcId + "_plan"}))
    print("lala2")
    if not db.list_collection_names(filter={'name': npcId + "_plan"}):
        # If the npcId collection does not exist, create it
        db.create_collection(npcId + "_plan")

        # And create an initial document
        db[npcId].insert_one({
            "vector": embedding_model.embed_query("<NULL ENTRY>"),  # Convert to list for JSON serialization
            "text": "<NULL ENTRY>",
            "timestamp": timestamp,
            "importance": importance,
            "recalled_summary": recalled_summary,
            "superset_command_of_god_id": superset_command_of_god_id,
            "planID": planID,
            "intendedPeople": intendedPeople,
            "routine_entries": routine_entries
        })

    # Create the memory object and insert it into the collection
    memoryObject = {
        "timestamp": timestamp,
        "vector": vector,
        "recalled_summary": recalled_summary,
        "superset_command_of_god_id": superset_command_of_god_id,
        "planID": planID,
        "intendedPeople": intendedPeople,
        "routine_entries": routine_entries,
        "importance": importance
    }

    db[npcId + "_plan"].insert_one({
        "page_content": recalled_summary, 
        "metadata": memoryObject
    })
    
    return {
        "npcId": npcId,
        "vector": vector,
        "recalled_summary": recalled_summary,
        "timestamp": timestamp,
        "superset_command_of_god_id": superset_command_of_god_id,
        "planID": planID,
        "intendedPeople": intendedPeople,
        "routine_entries": routine_entries,
        "importance": importance,
        "recency": datetime.datetime.now().timestamp() - memoryObject["timestamp"]
    }