import datetime

from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.callbacks import manager
from langchain.schema import Document
from typing import Dict, List, Optional, Tuple
from comparer import cosine_similarity
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://zeref:HHoKxFQYk9seO5pj@cluster1.4ocb6.mongodb.net/?retryWrites=true&w=majority"
#uri = "mongodb+srv://alex:o0uV7BkNFsRcmv5q@npc-memory.wyxvdjw.mongodb.net/?retryWrites=true&w=majority"
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
            query, **self.search_kwargs
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
        self, query: str, *, run_manager
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
def addBaseMemory(npcId, memory, timestamp, importance, checker=False):
    collection = db['base_memory']
    index_name = 'default'
    vectorstore = MongoDBAtlasVectorSearch(
        collection, embedding_model, index_name=index_name, embedding_key = "vector", text_key = "memory"
    )
    retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1)
    
    memory_object = {
        "npcId": npcId,
        "timestamp": timestamp,
        "importance": importance
    }
    returned_id = retriever.add_documents(([Document(page_content=memory, metadata=memory_object)]))
    
    return {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "importance": importance
    }

def getRelevantBaseMemoriesFrom(queries, npcId, max_memories=-1, top_k=1):
    collection = db['base_memory']
    index_name = 'default'
    vectorstore = MongoDBAtlasVectorSearch(
        collection, embedding_model, index_name=index_name, embedding_key = "vector", text_key = "memory"
    )

    #Documentation for `pre_filter`
    #https://www.mongodb.com/docs/atlas/atlas-search/knn-beta/ 
    #https://www.mongodb.com/community/forums/t/autocomplete-with-filter-compound-query/177582
    pre_filter = {
        "text": {
            'query': npcId,
            "path": "npcId"
        }
    }
    retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1, search_kwargs = {'pre_filter': pre_filter})
    retriever.k = top_k

    relevant = []

    for query in queries:
        #vector_query = embedding_model.embed_query(query)
        retrieved_docs = retriever.get_relevant_documents(query)

        for doc in retrieved_docs:
            memory = {
                "npcId": doc.metadata["npcId"],
                "memory": doc.page_content,
                "timestamp": doc.metadata["timestamp"],
                "vector": doc.metadata["vector"],
                "importance": doc.metadata["importance"],
                "recency": datetime.datetime.now().timestamp() - doc.metadata["timestamp"]
            }
            if memory not in relevant:
                relevant.append(memory)

    relevant.sort(key=lambda x: x["recency"], reverse=True)
    if max_memories > 0:
        relevant = relevant[:max_memories]

    return relevant


#### Relationship memory methods ####
def addRelationshipMemory(npcId, memory, timestamp, importance):
    collection = db['relationship_memory']
    
    index_name = 'default'
    vectorstore = MongoDBAtlasVectorSearch(
        collection, embedding_model, index_name=index_name, embedding_key = "vector", text_key = "memory"
    )
    retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1)
    
    memory_object = {
        "npcId": npcId,
        "timestamp": timestamp,
        "importance": importance
    }
    returned_id = retriever.add_documents(([Document(page_content=memory[0], metadata=memory_object)]))
    
    return {
        "npcId": npcId,
        "memory": memory,
        "timestamp": timestamp,
        "importance": importance
    }



def getRelevantRelationshipMemoriesFrom(queries, npcId, max_memories = -1, top_k=1):
    collection = db['relationship_memory']
    index_name = 'default'
    vectorstore = MongoDBAtlasVectorSearch(
        collection, embedding_model, index_name=index_name, embedding_key = "vector", text_key = "memory"
    )

    #Documentation for `pre_filter`
    #https://www.mongodb.com/docs/atlas/atlas-search/knn-beta/ 
    #https://www.mongodb.com/community/forums/t/autocomplete-with-filter-compound-query/177582
    pre_filter = {
        "text": {
            'query': npcId,
            "path": "npcId"
        }
    }
    retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1, search_kwargs = {'pre_filter': pre_filter})
    retriever.k = top_k

    relevant = []

    for query in queries:
        #vector_query = embedding_model.embed_query(query)
        retrieved_docs = retriever.get_relevant_documents(query)

        for doc in retrieved_docs:
            memory = {
                "npcId": doc.metadata["npcId"],
                "memory": doc.page_content,
                "timestamp": doc.metadata["timestamp"],
                "vector": doc.metadata["vector"],
                "importance": doc.metadata["importance"],
                "recency": datetime.datetime.now().timestamp() - doc.metadata["timestamp"]
            }
            if memory not in relevant:
                relevant.append(memory)

    relevant.sort(key=lambda x: x["recency"], reverse=True)
    if max_memories > 0:
        relevant = relevant[:max_memories]

    return relevant

def getRelevantPlanMemories(queries, npcId, max_memories = -1, threshold=0.8):
    collection = db['plan_memory']
    index_name = 'default'
    vectorstore = MongoDBAtlasVectorSearch(
        collection, embedding_model, index_name=index_name, embedding_key = "recalled_summary_vector", text_key = "recalled_summary"
    )

    #Documentation for `pre_filter`
    #https://www.mongodb.com/docs/atlas/atlas-search/knn-beta/ 
    #https://www.mongodb.com/community/forums/t/autocomplete-with-filter-compound-query/177582
    pre_filter = {
        "text": {
            'query': npcId,
            "path": "npcId"
        }
    }
    post_filter_pipeline = [
        {
            "$match": {
                "score": {"$gt": threshold}
            }
        }
    ]
    retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1, search_kwargs = {'pre_filter': pre_filter,'post_filter_pipeline': post_filter_pipeline})
    relevant = []
    
    for query in queries:
        retrieved_docs = retriever.get_relevant_documents(query)

        for doc in retrieved_docs:
            memory = {
                "npcId": doc.metadata["npcId"],
                "recalled_summary": doc.page_content,
                "planID": doc.metadata["planID"],
                "superset_command_of_god_id": doc.metadata["superset_command_of_god_id"],
                "timestamp": doc.metadata["timestamp"],
                "plannedDate": doc.metadata["plannedDate"],
                "recalled_summary_vector": doc.metadata["recalled_summary_vector"],
                "importance": doc.metadata["importance"],
                "recency": datetime.datetime.now().timestamp() - doc.metadata["timestamp"],
                "intendedPeople": doc.metadata["intendedPeople"],
                "intendedPeopleIDs": doc.metadata["intendedPeopleIDs"],
                "routine_entries": doc.metadata["routine_entries"],
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

#def get_document_from_plan_memory(npcId, pageContent):
#    collection = db['plan_memory']
    
    # Finding the document using npcId and pageContent (recalled_summary in this case)
#    doc = collection.find_one({"npcId": npcId, "recalled_summary": pageContent})
#    return doc if doc else None

def addPlanMemory(npcId, recalled_summary, timestamp, superset_command_of_god_id, planID, intendedPeople, intendedPeopleIDs, routine_entries, importance, plannedDate):
    collection = db['plan_memory']
    index_name = 'default'
    vectorstore = MongoDBAtlasVectorSearch(
        collection, embedding_model, index_name=index_name, embedding_key = "recalled_summary_vector", text_key = "recalled_summary"
    )

    retriever = TimeWeightedVectorStoreRetriever_custom(vectorstore=vectorstore, other_score_keys = ['importance'] , decay_rate=.01, k=1)
    memory_object = {
        "npcId": npcId,
        "timestamp": timestamp,
        "superset_command_of_god_id": superset_command_of_god_id,
        "planID": planID,
        "intendedPeople": intendedPeople,
        "intendedPeopleIDs": intendedPeopleIDs,
        "routine_entries": routine_entries,
        "importance": importance,
        "plannedDate": plannedDate,
    }
    returned_id = retriever.add_documents(([Document(page_content=recalled_summary, metadata=memory_object)]))
    
    return {
        "npcId": npcId,
        "timestamp": timestamp,
        "superset_command_of_god_id": superset_command_of_god_id,
        "planID": planID,
        "intendedPeople": intendedPeople,
        "intendedPeopleIDs": intendedPeopleIDs,
        "routine_entries": routine_entries,
        "importance": importance,
        "plannedDate": plannedDate,
        "recalled_summary" : recalled_summary
    }

def get_lastk_base_memories(npcId, k = 100):
    collection = db['base_memory']  # Assuming the collection name is 'base_memory'. Adjust if necessary.
    
    # If you want to keep the original functionality of checking for the npcId existence, you'd do:
    npc_exists = collection.find_one({"npcId": npcId})
    if not npc_exists:
        return []
    
    cursor = collection.find({"npcId": npcId})
    memories = []

    for doc in cursor:
        timestamp = doc.get("timestamp", 0)
        importance = doc.get("importance", 0)
        memory_content = doc.get("memory", "")

        memories.append({
            "npcId": npcId,
            "timestamp": timestamp,
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
        importance = doc.get("importance", 0)
        memory_content = doc.get("memory", "")

        memories.append({
            "npcId": npcId,
            "timestamp": timestamp,
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
