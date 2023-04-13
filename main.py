import asyncio
import os
from dataclasses import dataclass
from fastapi import FastAPI, Request, BackgroundTasks
import logging
import os
from time import time
from fastapi.middleware.cors import CORSMiddleware
from vectorizer import vectorize

logging.basicConfig(level=logging.INFO, format="%(levelname)-9s %(asctime)s - %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

EXPERIMENTS_BASE_DIR = "/experiments/"
QUERY_BUFFER = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
loop = asyncio.get_event_loop()

@dataclass
class Query():
    query_name: str
    query_sequence: str
    input: str
    result: str = ""
    extention: str = ".glb"
    experiment_id: str = None
    status: str = "pending"

    def __post_init__(self):
        self.experiment_id = str(time())
        self.experiment_dir = os.path.join(EXPERIMENTS_BASE_DIR, self.experiment_id)

@app.get("/generate")
async def root(request: Request, background_tasks: BackgroundTasks, input: str):
    query = Query(query_name="test", query_sequence=5, input=input)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    LOGGER.info(f'root - added task')
    return {"id": query.experiment_id}

@app.get("/generate_result")
async def result(request: Request, query_id: str):
    if query_id in QUERY_BUFFER:
        if QUERY_BUFFER[query_id].status == "done":
            resp = { 'vector': QUERY_BUFFER[query_id].result }
            del QUERY_BUFFER[query_id]
            return resp
        return {"status": QUERY_BUFFER[query_id].status}
    else:
        return {"status": "finished"}

def process(query):
    LOGGER.info(f"process - {query.experiment_id} - Submitted query job. Now run non-IO work for 10 seconds...")
    start_time = time()
    res = vectorize(query.input)
    QUERY_BUFFER[query.experiment_id].result = res
    print("--- %s seconds ---" % (time() - start_time))
    QUERY_BUFFER[query.experiment_id].status = "done"
    LOGGER.info(f'process - {query.experiment_id} - done!')

@app.get("/backlog/")
def return_backlog():
    return {f"return_backlog - Currently {len(QUERY_BUFFER)} jobs in the backlog."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=65529)