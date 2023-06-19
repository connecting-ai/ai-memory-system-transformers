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

import faiss 
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings 

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain.vectorstores import FAISS


from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy


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


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Initialize the vectorstore as empty
embedding_size = 384

npcID_to_retriever = {}
print("starting, will try to load - exists", os.path.exists("retriever.pkl"))
if os.path.exists("retreiver.pbz2"):
    print("load file found")
    data = bz2.BZ2File("retreiver.pbz2", "rb")
    data = cPickle.load(data)
    npcID_to_retriever = data
    print("loaded:", npcID_to_retriever)


index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embedding_model.embed_query, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1) 



#Edit this method to store in-game time input argument from the metadata
def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
    """Add documents to vectorstore."""
    current_time = kwargs.get("current_time", datetime.now())
    # Avoid mutating input documents
    dup_docs = [deepcopy(d) for d in documents]
    for i, doc in enumerate(dup_docs):
        if "last_accessed_at" not in doc.metadata:
            doc.metadata["last_accessed_at"] = current_time
        if "created_at" not in doc.metadata:
            doc.metadata["created_at"] = current_time
        doc.metadata["buffer_idx"] = len(self.memory_stream) + i
    self.memory_stream.extend(dup_docs)
    return self.vectorstore.add_documents(dup_docs, **kwargs)                     





def addMemory(npcId, memory, timestamp, lastAccess, vector, importance, checker=False):

    #If the npcId has not been seen before, create a memory database and retriever for it
    if npcId not in npcID_to_retriever.keys():
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embedding_model.embed_query, index, InMemoryDocstore({}), {})
        retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1) 
        npcID_to_retriever[npcId] = retriever

    memoryObject = {
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance
    }

    npcID_to_retriever[npcId].add_documents([Document(page_content=memory, metadata=memoryObject)])

    

    return {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "lastAccess": lastAccess,
        "vector": vector,
        "importance": importance,
        "recency": datetime.datetime.now().timestamp() - memoryObject["lastAccess"]
    }



# def getMemoriesShortedByLastAccess(npcId, max=100):
#     query = {
#         "npcId": npcId
#     }
#     cursor = col.find(query).sort("lastAccess", -1)
#     arr = [x for x in cursor]
#     print('dada', len(arr))
#     for x in arr:
#         x["lastAccess"] = datetime.datetime.fromtimestamp(x["lastAccess"])
#     arr.sort(key=lambda x: x["lastAccess"], reverse=True)
#     if (max>0 and len(arr)>max):
#         arr = arr[:max]
    
#     for x in arr:
#         x["lastAccess"] = x["lastAccess"].timestamp()

#     unixTimeNow = datetime.datetime.now().timestamp()
#     for memory in arr:
#         memoryTime = memory["lastAccess"]
#         recency = unixTimeNow - memoryTime
#         recency = recency * 0.99
#         memory["recency"] = recency

#     return arr

def getRelevantMemoriesFrom(queries, npcId):
    if npcId not in npcID_to_retriever.keys():
        return []

    retriever = npcID_to_retriever[npcId]
    relevant = []

    for query in queries:
        # vector = embedding_model.embed_query(query)
        retrieved_docs = retriever.get_relevant_documents(query)

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

    return relevant


def getRelevantMemories(memories, targetVectors, query):
    winner = compare(query, targetVectors)

    winMemory = []
    for win in winner:
        winMemory.append(memories[win["index"]])
    return winMemory    
