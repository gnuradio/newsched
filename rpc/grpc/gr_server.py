# Copyright 2020 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python AsyncIO implementation of the GRPC helloworld.Greeter server."""

import asyncio
import logging
import importlib
import json

import grpc
import gnuradio_pb2
import gnuradio_pb2_grpc

from newsched import gr

class GrServer(gnuradio_pb2_grpc.GNURadioServicer):
    def __init__(self):
        self.flowgraphs = {}
        self.blocks = {}
        self.edges = {}

    async def CreateFlowgraph(
            self, request: gnuradio_pb2.FlowgraphCreateRequest,
            context: grpc.aio.ServicerContext) -> gnuradio_pb2.GenericReply:

        try:
            self.flowgraphs[request.name] = gr.flowgraph(request.name)
            return gnuradio_pb2.GenericReply(message='{status: ok}')
        except Exception as e:
            return gnuradio_pb2.GenericReply(message=f'{{status: exception, details:{str(e)} }}')

    async def StartFlowgraph(
            self, request: gnuradio_pb2.FlowgraphActionRequest,
            context: grpc.aio.ServicerContext) -> gnuradio_pb2.GenericReply:

        try:
            self.flowgraphs[request.name].start()
            return gnuradio_pb2.GenericReply(message='{status: ok}')
        except Exception as e:
            return gnuradio_pb2.GenericReply(message=f'{{status: exception, details:{str(e)} }}')

    async def WaitFlowgraph(
            self, request: gnuradio_pb2.FlowgraphActionRequest,
            context: grpc.aio.ServicerContext) -> gnuradio_pb2.GenericReply:

        try:
            self.flowgraphs[request.name].wait()
            return gnuradio_pb2.GenericReply(message='{status: ok}')
        except Exception as e:
            return gnuradio_pb2.GenericReply(message=f'{{status: exception, details:{str(e)} }}')




    async def CreateBlock(
            self, request: gnuradio_pb2.BlockCreateRequest,
            context: grpc.aio.ServicerContext) -> gnuradio_pb2.GenericReply:

        try:
            parms = json.loads(request.parameters)
            self.blocks[request.name] = importlib.import_module(
            'newsched.' + request.module).__getattribute__(request.block)(**parms)
            return gnuradio_pb2.GenericReply(message='{status: ok}')
        except Exception as e:
            return gnuradio_pb2.GenericReply(message=f'{{status: exception, details:{str(e)} }}')

    async def ConnectBlocks(
            self, request: gnuradio_pb2.FlowgraphConnectBlocksRequest,
            context: grpc.aio.ServicerContext) -> gnuradio_pb2.FlowgraphConnectBlocksReply:

        try:
            src  = (self.blocks[request.src_block],request.src_port_index)
            dest = (self.blocks[request.dest_block],request.dest_port_index)
            edge = self.flowgraphs[request.name].connect(src, dest)

            if request.HasField('edge_name'):
                edge_name = request.edge_name
            else:
                edge_name = edge.identifier()
            return gnuradio_pb2.FlowgraphConnectBlocksReply(message='{status: ok}', edge_name=edge_name)
        except Exception as e:
            return gnuradio_pb2.FlowgraphConnectBlocksReply(message=f'{{status: exception, details:{str(e)} }}')


async def serve() -> None:
    server = grpc.aio.server()
    gnuradio_pb2_grpc.add_GNURadioServicer_to_server(GrServer(), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())