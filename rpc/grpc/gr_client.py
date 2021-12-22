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
"""The Python AsyncIO implementation of the GRPC helloworld.GNURadio client."""

import asyncio
import logging

import grpc
import gnuradio_pb2
import gnuradio_pb2_grpc


async def run() -> None:
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = gnuradio_pb2_grpc.GNURadioStub(channel)
        response = await stub.CreateFlowgraph(gnuradio_pb2.FlowgraphCreateRequest(name='foo'))
        print("GNURadio client received: " + response.message)

        response = await stub.CreateBlock(
            gnuradio_pb2.BlockCreateRequest(
                name='src', module='blocks',
                block='vector_source_f', 
                parameters=f'{{"data": [1,2,3,4,5], "repeat":false }}'))
        print("GNURadio client received: " + response.message)

        response = await stub.CreateBlock(
            gnuradio_pb2.BlockCreateRequest(
                name='copy', module='blocks',
                block='copy', parameters=f'{{ }}'))
        print("GNURadio client received: " + response.message)

        response = await stub.CreateBlock(
            gnuradio_pb2.BlockCreateRequest(
                name='snk', module='blocks',
                block='vector_sink_f', parameters=f'{{ }}'))
        print("GNURadio client received: " + response.message)

        response = await stub.ConnectBlocks(
            gnuradio_pb2.FlowgraphConnectBlocksRequest(
                name='foo', 
                src_block='src',
                src_port_index=0,
                dest_block='copy',
                dest_port_index=0))
        print("GNURadio client received: " + response.edge_name)

        response = await stub.ConnectBlocks(
            gnuradio_pb2.FlowgraphConnectBlocksRequest(
                name='foo', 
                src_block='copy',
                src_port_index=0,
                dest_block='snk',
                dest_port_index=0))
        print("GNURadio client received: " + response.edge_name)

        response = await stub.StartFlowgraph(
            gnuradio_pb2.FlowgraphActionRequest(
                name='foo'))
        print("GNURadio client received: " + response.message)

        response = await stub.WaitFlowgraph(
            gnuradio_pb2.FlowgraphActionRequest(
                name='foo'))
        print("GNURadio client received: " + response.message)

if __name__ == '__main__':
    logging.basicConfig()
    asyncio.run(run())
