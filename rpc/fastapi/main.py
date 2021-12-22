from fastapi import FastAPI, Request, Body
from typing import Optional
from pydantic import BaseModel

import importlib
import time
from newsched import gr

class FlowgraphProperties(BaseModel):
    name: str

class BlockProperties(BaseModel):
    name: str
    module: str
    id: str
    props: dict


class Session:
    def __init__(self):
        self.fgs = {}
        self.blocks = {}
        self.edges = {}

    # {"name": "Foo"}
    def create_flowgraph(self, fg):
        self.fgs[fg.name] = gr.flowgraph(fg.name)
        return "{status: 0}"

    def start_flowgraph(self, fg):
        self.fgs[fg.name].start()
        return "{status: 0}"

    def wait_flowgraph(self, fg):
        self.fgs[fg.name].wait()
        print('flowgraph done')
        time.sleep(3)
        return "{status: 0}"

    def stop_flowgraph(self, fg):
        self.fgs[fg.name].stop()
        return "{status: 0}"

    # {"name": "src", "module": "blocks", "id": "vector_source_c", "properties": {"data": [1,2,3,4,5], "repeat": false }}
    # {"name": "copy_0", "module": "blocks", "id": "copy", "properties": {"itemsize": 8}}
    # {"name": "copy_1", "module": "blocks", "id": "copy", "properties": {"itemsize": 8}}
    # {"name": "snk", "module": "blocks", "id": "vector_sink_c", "properties": {}}

    def create_block(self, payload):
        print(payload)
        self.blocks[payload['name']] = importlib.import_module(
            'newsched.' + payload['module']).__getattribute__(payload['id'])(**payload['properties'])

        print(self.blocks[payload['name']])

        return "{status: 0}"

    # {"flowgraph": "Foo", "src": ["copy_0",0], "snk": ["copy_1",1] }
    # {"flowgraph": "Foo", "src": ["copy_0",0], "snk": ["copy_1",1] }
    def connect_blocks(self, payload):
        print(self.blocks)
        src  = (self.blocks[payload['src'][0]],payload['src'][1])
        dest = (self.blocks[payload['dest'][0]],payload['dest'][1])
        print(src)
        print(dest)
        edge = self.fgs[payload['flowgraph']].connect(src, dest)

        # if the edge is named in the payload, the store the 
        if 'edge_name' in payload:
            edge_name = payload['edge_name']
        else:
            edge_name = edge.identifier()

        return f"{{\"status\": 0, \"edge\": {edge_name}}}"


def create_app():

    app = FastAPI()
    session = Session()

    # @app.on_event("startup")
    # async def startup():
    #     await db.create_pool()

    # @app.on_event("shutdown")
    # async def shutdown():
    #     # cleanup
    #     pass

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    @app.post("/flowgraph/create")
    async def create_flowgraph(fg: FlowgraphProperties):
        return session.create_flowgraph(fg)

    @app.post("/flowgraph/start")
    async def start_flowgraph(fg: FlowgraphProperties):
        return session.start_flowgraph(fg)

    @app.post("/flowgraph/wait")
    async def wait_flowgraph(fg: FlowgraphProperties):
        return session.wait_flowgraph(fg)

    @app.post("/flowgraph/stop")
    async def stop_flowgraph(fg: FlowgraphProperties):
        return session.stop_flowgraph(fg)

    @app.post("/block/create")
    async def create_block(payload: dict = Body(...)):
        return session.create_block(payload)

    @app.post("/flowgraph/connect")
    async def connect_blocks(payload: dict = Body(...)):
        return session.connect_blocks(payload)

    return app


app = create_app

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/items/")
# async def create_item(item: Item):
#     return item
