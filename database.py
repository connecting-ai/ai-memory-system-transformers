import datetime
import json
import numpy as np
import pymongo 

import _pickle as cPickle
import os
import bz2

from constants import DB_NAME, MONGO_URL, COL_NAME
from vectorizer import compare

from constants import OPENAI_KEY

from langchain.vectorstores import LanceDB
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document


from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import lancedb
db_init = lancedb.connect("./lancedb")

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

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
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

    #If the npcId has not been seen before, create a memory database and retriever for it
    if npcId not in npcID_to_base_retriever.keys():
        table = db_init.create_table(
                    npcId,
                    data=[
                        {
                            "vector": embedding_model.embed_query("<NULL ENTRY>"),
                            "text": "<NULL ENTRY>",
                        }
                    ],
                    mode="overwrite",
                    on_bad_vectors = 'drop'
                )
        vectordb = LanceDB(embedding = embedding_model , connection=table)
        retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectordb, other_score_keys = ['importance'] , decay_rate=.01, k=1) 
        npcID_to_base_retriever[npcId] = retriever

    memoryObject = {
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }

    npcID_to_base_retriever[npcId].add_documents([Document(page_content=memory, metadata=memoryObject)])

    return {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance,
        "recency": datetime.datetime.now().timestamp() - memoryObject["lastAccess"]
    }


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


#### Relationship memory methods ####
def addRelationshipMemory(npcId, memory, timestamp, lastAccess, vector, importance, checker=False):

    #If the npcId has not been seen before, create a memory database and retriever for it
    if npcId not in npcID_to_relationship_retriever.keys():
        table = db_init.create_table(
                    npcId,
                    data=[
                        {
                            "vector": embedding_model.embed_query("<NULL ENTRY>"),
                            "text": "<NULL ENTRY>",
                        }
                    ],
                    mode="overwrite",
                )
        vectordb = LanceDB(embedding = embedding_model , connection=table)
        retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectordb, other_score_keys = ['importance'] , decay_rate=.01, k=1) 
        npcID_to_relationship_retriever[npcId] = retriever

    memoryObject = {
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }

    npcID_to_relationship_retriever[npcId].add_documents([Document(page_content=memory, metadata=memoryObject)])

    return {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance,
        "recency": datetime.datetime.now().timestamp() - memoryObject["lastAccess"]
    }


def getRelevantRelationshipMemoriesFrom(queries, npcId, max_memories = -1, top_k=1):
    if npcId not in npcID_to_relationship_retriever.keys():
        return []

    retriever = npcID_to_relationship_retriever[npcId]
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

def getRelevantPlanMemories(queries, npcId, max_memories = -1, top_k=1):
    tempNpcId = npcId
    npcId = npcId + "_plan"
    if npcId not in npcID_to_relationship_retriever.keys():
        return []

    retriever = npcID_to_relationship_retriever[npcId]
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
                "npcId": tempNpcId,
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
    tempNpcId = npcId
    npcId = npcId + "_plan"
    #If the npcId has not been seen before, create a memory database and retriever for it
    if npcId not in npcID_to_plan_retriever.keys():
        table = db_init.create_table(
                    npcId,
                    data=[
                        {
                            "vector": embedding_model.embed_query("<NULL ENTRY>"),
                            "text": "<NULL ENTRY>",
                        }
                    ],
                    mode="overwrite",
                )
        vectordb = LanceDB(embedding = embedding_model , connection=table)
        retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectordb, other_score_keys = ['importance'] , decay_rate=.01, k=1) 
        npcID_to_plan_retriever[npcId] = retriever

    memoryObject = {
        "timestamp": timestamp,
        "recalled_summary": recalled_summary,
        "superset_command_of_god_id": superset_command_of_god_id,
        "planID": planID,
        "intendedPeople": intendedPeople,
        "routine_entries": routine_entries,
        "importance": importance
    }

    npcID_to_plan_retriever[npcId].add_documents([Document(page_content=recalled_summary, metadata=memoryObject)])

    return {
        "npcId": tempNpcId,
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