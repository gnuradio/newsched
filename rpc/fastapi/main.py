from fastapi import FastAPI, Request, Body
from typing import Optional
from pydantic import BaseModel

import importlib
import time
from newsched import gr
import json
import numpy as np

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
    def create_flowgraph(self, fg_name):
        self.fgs[fg_name] = gr.flowgraph(fg_name)
        return "{status: 0}"

    def start_flowgraph(self, fg_name):
        self.fgs[fg_name].start()
        return "{status: 0}"

    def wait_flowgraph(self, fg_name):
        self.fgs[fg_name].wait()
        return "{status: 0}"

    def stop_flowgraph(self, fg_name):
        self.fgs[fg_name].stop()
        return "{status: 0}"

    # {"name": "src", "module": "blocks", "id": "vector_source_c", "properties": {"data": [1,2,3,4,5], "repeat": false }}
    # {"name": "copy_0", "module": "blocks", "id": "copy", "properties": {"itemsize": 8}}
    # {"name": "copy_1", "module": "blocks", "id": "copy", "properties": {"itemsize": 8}}
    # {"name": "snk", "module": "blocks", "id": "vector_sink_c", "properties": {}}

    def create_block(self, block_name, payload):
        if 'format' in payload and payload['format'] == 'b64':
            self.blocks[block_name] = importlib.import_module(
                'newsched.' + payload['module']).__getattribute__(payload['id']).make_from_params(json.dumps(payload['parameters']))
        else:
            self.blocks[block_name] = importlib.import_module(
                'newsched.' + payload['module']).__getattribute__(payload['id'])(**payload['parameters'])

        return "{status: 0}"

    def block_method(self, block_name, method, payload):

        ret = {}
        ret['result'] = self.blocks[block_name].__getattribute__(method)(**payload)
        print(ret['result'])

        print(type(ret['result']))
        print(type(ret['result'][0]))
        def default_serializer(z):
            # if isinstance(z, list) and isinstance(z[0], complex):
            #     return {'re': [x.real() for x in z], 'im': [x.imag() for x in z]}
            if isinstance(z, complex):
                return (z.real, z.imag)

            type_name = z.__class__.__name__
            raise TypeError(f"Object of type '{type_name}' is not JSON serializable")


        # return ret
        return json.dumps(ret, default = default_serializer)

    # {"flowgraph": "Foo", "src": ["copy_0",0], "snk": ["copy_1",1] }
    # {"flowgraph": "Foo", "src": ["copy_0",0], "snk": ["copy_1",1] }
    def connect_blocks(self, fg_name, payload):
        print(self.blocks)
        src  = (self.blocks[payload['src'][0]],payload['src'][1])
        dest = (self.blocks[payload['dest'][0]],payload['dest'][1])
        print(src)
        print(dest)
        edge = self.fgs[fg_name].connect(src, dest)

        # if the edge is named in the payload, the store the 
        if 'edge_name' in payload:
            edge_name = payload['edge_name']
        else:
            edge_name = edge.identifier()

        return f"{{\"status\": 0, \"edge\": {edge_name}}}"

    '{"src": ["src",0], "ipaddr":"127.0.0.1", "port":1234 }'
    def create_edge(self, fg_name, payload):

        src  = (self.blocks[payload['src'][0]],payload['src'][1]) if payload['src'] else None
        dest = (self.blocks[payload['dest'][0]],payload['dest'][1]) if payload['dest'] else None

        # edge = self.fgs[fg_name].connect(src, dest)
        if (src and dest):
            edge = gr.edge(src[0], src[0].get_port(src[1], gr.port_type_t.STREAM, gr.port_direction_t.OUTPUT),
                           dest[0], dest[0].get_port(dest[1], gr.port_type_t.STREAM, gr.port_direction_t.INPUT))
        elif (src):
            edge = gr.edge(src[0], src[0].get_port(src[1], gr.port_type_t.STREAM, gr.port_direction_t.OUTPUT),
                           None, None)
        elif (dest):
            edge = gr.edge(None, None,
                           dest[0], dest[0].get_port(dest[1], gr.port_type_t.STREAM, gr.port_direction_t.INPUT))

        if (not (src and dest)):
            edge.set_custom_buffer(gr.buffer_net_zmq_properties.make(payload['ipaddr'], payload['port']))

        self.fgs[fg_name].add_edge(edge)

        # if the edge is named in the payload, the store the 
        if 'edge_name' in payload:
            edge_name = payload['edge_name']
        else:
            edge_name = edge.identifier()

        return f"{{\"status\": 0, \"edge\": {edge_name}}}"

    def create_fgm_proxy(self, fg_name, payload):
        ipaddr = payload['ipaddr']
        port = payload['port']
        upstream = payload['upstream']

        proxy2 = gr.fgm_proxy(ipaddr, port, upstream)
        self.fgs[fg_name].add_fgm_proxy(proxy2)

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

    @app.post("/flowgraph/{fg_name}/create")
    async def create_flowgraph(fg_name: str):
        return session.create_flowgraph(fg_name)

    @app.post("/flowgraph/{fg_name}/start")
    async def start_flowgraph(fg_name: str):
        return session.start_flowgraph(fg_name)

    @app.post("/flowgraph/{fg_name}/wait")
    async def wait_flowgraph(fg_name: str):
        return session.wait_flowgraph(fg_name)

    @app.post("/flowgraph/{fg_name}/stop")
    async def stop_flowgraph(fg_name: str):
        return session.stop_flowgraph(fg_name)

    @app.post("/block/{block_name}/create")
    async def create_block(block_name: str, payload: dict = Body(...)):
        return session.create_block(block_name, payload)

    @app.post("/block/{block_name}/{method}")
    async def block_method(block_name: str, method: str, payload: dict = Body(...)):
        return session.block_method(block_name, method, payload)

    @app.post("/flowgraph/{fg_name}/connect")
    async def connect_blocks(fg_name: str, payload: dict = Body(...)):
        return session.connect_blocks(fg_name, payload)

    @app.post("/flowgraph/{fg_name}/edge/create")
    async def create_edge(fg_name: str, payload: dict = Body(...)):
        return session.create_edge(fg_name, payload)

    @app.post("/flowgraph/{fg_name}/proxy/create")
    async def connect_blocks(fg_name: str, payload: dict = Body(...)):
        return session.create_fgm_proxy(fg_name, payload)


    return app


app = create_app

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/items/")
# async def create_item(item: Item):
#     return item
