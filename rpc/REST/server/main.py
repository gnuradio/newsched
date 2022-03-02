from fastapi import FastAPI, Request, Body
from typing import Optional
from pydantic import BaseModel

import importlib
import time
from gnuradio import gr
import json
import numpy as np

import string
import random
import pmtf
from base64 import b64encode, b64decode

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
        self.rts = {}
        self.blocks = {}
        self.edges = {}
        self.proxies = {}

    def execute(self, payload):
        print(payload)
        return self.__getattribute__(payload['command'])(**payload)

    # {"name": "Foo"}
    def flowgraph_create(self, **kwargs):
        self.fgs[kwargs['fg_name']] = gr.flowgraph(kwargs['fg_name'])
        # self.fgs[fg_name] = gr.flowgraph(fg_name)
        return {'status': 0}

    def flowgraph_start(self, **kwargs):
        self.fgs[kwargs['fg_name']].start()
        return {'status': 0}

    def flowgraph_wait(self, **kwargs):
        self.fgs[kwargs['fg_name']].wait()
        return {'status': 0}

    def flowgraph_stop(self, **kwargs):
        self.fgs[kwargs['fg_name']].stop()
        return {'status': 0}

    def flowgraph_connect(self, **kwargs):
        src = (self.blocks[kwargs['src'][0]],
               kwargs['src'][1]) if 'src' in kwargs and kwargs['src'] else None
        dst = (self.blocks[kwargs['dst'][0]],
                kwargs['dst'][1]) if 'dst' in kwargs and  kwargs['dst'] else None
        print(src)
        print(dst)
        if (src and dst):
            edge = self.fgs[kwargs['fg_name']].connect(src, dst)
        elif (dst):
            edge = self.fgs[kwargs['fg_name']].connect((None,""), dst)
        else:
            edge = self.fgs[kwargs['fg_name']].connect(src, (None,""))

        # if the edge is named in the payload, the store the
        if 'edge_name' in kwargs:
            edge_name = kwargs['edge_name']
        else:
            edge_name = edge.identifier()

        self.edges[edge_name] = edge

        return {"status": 0, "edge": edge_name}

    def flowgraph_create_edge(self, fg_name, payload):

        src = (self.blocks[payload['src'][0]],
               payload['src'][1]) if 'src' in payload and payload['src'] else None
        dest = (self.blocks[payload['dest'][0]],
                payload['dest'][1]) if 'dest' in payload and  payload['dest'] else None

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

        self.fgs[fg_name].add_edge(edge)

        # if the edge is named in the payload, the store the
        if 'edge_name' in payload:
            edge_name = payload['edge_name']
        else:
            edge_name = edge.identifier()

        self.edges[edge_name] = edge
        
        return {"status": 0, "edge": edge_name}

    def runtime_create(self, **kwargs):
        self.rts[kwargs['rt_name']] = gr.runtime()
        return {'status': 0}

    def runtime_initialize(self, **kwargs):
        self.rts[kwargs['rt_name']].initialize(self.fgs[kwargs['fg_name']])
        return {'status': 0}

    def runtime_start(self, **kwargs):
        self.rts[kwargs['rt_name']].start()
        return {'status': 0}

    def runtime_wait(self, **kwargs):
        self.rts[kwargs['rt_name']].wait()
        return {'status': 0}

    def runtime_stop(self, **kwargs):
        self.rts[kwargs['rt_name']].stop()
        return {'status': 0}



    def block_create(self, **kwargs):
        if 'format' in kwargs and kwargs['format'] == 'b64':
            self.blocks[kwargs['block_name']] = importlib.import_module(
                'gnuradio.' + kwargs['module']).__getattribute__(kwargs['id']).make_from_params(json.dumps(kwargs['parameters']))
        else:
            self.blocks[kwargs['block_name']] = importlib.import_module(
                'gnuradio.' + kwargs['module']).__getattribute__(kwargs['id'])(**kwargs['parameters'])

        return {'status': 0}

    def block_method(self, **kwargs): #block_name, method, payload):

        ret = {}
        ret['result'] = self.blocks[kwargs['block_name']].__getattribute__(
            kwargs['method'])(**kwargs['params'])

        def default_serializer(z):
            # if isinstance(z, list) and isinstance(z[0], complex):
            #     return {'re': [x.real() for x in z], 'im': [x.imag() for x in z]}
            if isinstance(z, complex):
                return (z.real, z.imag)

            type_name = z.__class__.__name__
            raise TypeError(
                f"Object of type '{type_name}' is not JSON serializable")

        return ret
        # return json.dumps(ret, default=default_serializer)

    def block_parameter_query(self, **kwargs): #block_name, parameter):

        ret = {}
        pmt_res = self.blocks[kwargs['block_name']].request_parameter_query(kwargs['parameter'])
        b64str = pmt_res.to_base64()

        ret['result'] = b64str
        return ret

    def block_parameter_change(self, **kwargs): #block_name, parameter, payload):

        newvalue = pmtf.pmt.from_base64(kwargs['encoded_value'])

        # print(newvalue())

        # request_parameter_change(int param_id, pmtf::pmt new_value, bool block = true);
        self.blocks[kwargs['block_name']].request_parameter_change(kwargs['parameter'], newvalue, False)
        return {}

    def block_create_message_port_proxy(self, block_name, port_name, payload):
        upstream = payload['upstream']
        proxy_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
        mp = self.blocks[block_name].get_message_port(port_name)
        port = payload['port']

        if (upstream):
            proxy = gr.message_port_proxy_upstream()
            proxy.connect(payload['ipaddr'], payload['port'])
            mp.connect(proxy)
        else:
            proxy = gr.message_port_proxy_downstream(payload['port'])
            port = proxy.port()
            proxy.set_gr_port(mp)

        if block_name not in self.proxies:
            self.proxies[block_name] = {}
        self.proxies[block_name][port_name] = proxy
        return {"status": 0, "port": port, "proxy_name": proxy_name}

    def edge_set_custom_buffer(self, edge_name, payload):
        print(payload)
        buf_props = gr.__getattribute__(payload['id']).make_from_params(json.dumps(payload['parameters']))
        self.edges[edge_name].set_custom_buffer(buf_props)

        return {"status": 0, "edge": edge_name}

    def runtime_create_proxy(self, **kwargs): #rt_name, payload):
        svr_port = kwargs['svr_port']
        upstream = kwargs['upstream']

        proxy2 = gr.runtime_proxy(svr_port, upstream)
        proxy_port = proxy2.svr_port()
        proxy_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
        self.rts[kwargs['rt_name']].add_proxy(proxy2)

        self.proxies[proxy_name] = proxy2
        return {"status": 0, "name": proxy_name, "port": proxy_port}

    def runtime_connect_proxy(self, **kwargs): #proxy_name, payload):
        ipaddr = kwargs['ipaddr']
        port = kwargs['port']
        
        proxy2 = self.proxies[kwargs['proxy_name']]
        proxy2.client_connect(ipaddr, port)

        return {"status": 0}


    def start_downstream_message_port_rx(self, block_name, port_name):

        self.proxies[block_name][port_name].start_rx()

        return {"status": 0}

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

    @app.post("/execute")
    async def execute(payload: dict = Body(...)):
        return session.execute(payload)

    # @app.post("/flowgraph/{fg_name}/create")
    # async def create_flowgraph(fg_name: str):
    #     return session.create_flowgraph(fg_name)

    # @app.post("/flowgraph/{fg_name}/initialize")
    # async def initialize_flowgraph(fg_name: str):
    #     return session.initialize_flowgraph(fg_name)

    # @app.post("/flowgraph/{fg_name}/start")
    # async def start_flowgraph(fg_name: str):
    #     return session.start_flowgraph(fg_name)

    # @app.post("/flowgraph/{fg_name}/wait")
    # async def wait_flowgraph(fg_name: str):
    #     return session.wait_flowgraph(fg_name)

    # @app.post("/flowgraph/{fg_name}/stop")
    # async def stop_flowgraph(fg_name: str):
    #     return session.stop_flowgraph(fg_name)

    # @app.post("/block/{block_name}/create")
    # async def create_block(block_name: str, payload: dict = Body(...)):
    #     return session.create_block(block_name, payload)

    # @app.post("/block/{block_name}/{method}")
    # async def block_method(block_name: str, method: str, payload: dict = Body(...)):
    #     return session.block_method(block_name, method, payload)

    # @app.post("/block/{block_name}/get_parameter/{parameter_name}")
    # async def request_parameter_query(block_name: str, parameter_name: str):
    #     return session.request_parameter_query(block_name, parameter_name)

    # @app.post("/block/{block_name}/set_parameter/{parameter_name}")
    # async def request_parameter_change(block_name: str, parameter_name: str, payload: dict = Body(...)):
    #     return session.request_parameter_change(block_name, parameter_name, payload)

    # @app.post("/flowgraph/{fg_name}/connect")
    # async def connect_blocks(fg_name: str, payload: dict = Body(...)):
    #     return session.connect_blocks(fg_name, payload)

    # @app.post("/flowgraph/{fg_name}/edge/create")
    # async def create_edge(fg_name: str, payload: dict = Body(...)):
    #     return session.create_edge(fg_name, payload)

    # @app.post("/flowgraph/{fg_name}/edge/{edge_name}/set_custom_buffer")
    # async def set_custom_buffer(fg_name: str, edge_name : str, payload: dict = Body(...)):
    #     return session.set_custom_buffer(fg_name, edge_name, payload)

    # @app.post("/flowgraph/{fg_name}/proxy/create")
    # async def create_runtime_proxy(fg_name: str, payload: dict = Body(...)):
    #     return session.create_runtime_proxy(fg_name, payload)

    # @app.post("/proxy/{proxy_name}/connect")
    # async def connect_runtime_proxy(proxy_name: str, payload: dict = Body(...)):
    #     return session.connect_runtime_proxy(proxy_name, payload)

    # @app.post("/block/{block_name}/message_port/{port_name}/proxy/create")
    # async def create_message_port_proxy(block_name: str, port_name: str, payload: dict = Body(...)):
    #     return session.create_message_port_proxy(block_name, port_name, payload)

    # @app.post("/block/{block_name}/message_port/{port_name}/proxy/start_rx")
    # async def create_message_port_proxy(block_name: str, port_name: str):
    #     return session.start_downstream_message_port_rx(block_name, port_name)

    return app


app = create_app

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/items/")
# async def create_item(item: Item):
#     return item
