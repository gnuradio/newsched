from gnuradio import gr
import uuid
# from python_on_whales import docker
from jinja2 import FileSystemLoader, Environment
import os
from gnuradio import zeromq

class runtime:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.docker_compose_filename:
            os.system(f'docker compose -f {self.docker_compose_filename} down')
        # pass

    def __init__(self, docker_compose_filename):
        self.service_blocks_map = {}
        self.service_client_map = {}
        self.service_sched_map = {}
        self.block_client_map = {}
        self.client_fgname_map = {}
        self.docker_compose_filename = docker_compose_filename


    def assign_rpc_client(self, service, client):
        self.service_client_map[service] = client

    def assign_blocks(self, service, blocks):
        self.service_blocks_map[service] = blocks

    def assign_scheduler(self, service, blocks):
        self.service_blocks_map[service] = blocks

    def initialize(self, fg):
        gr.flowgraph.check_connections(fg)

        # For now, we have a single scheduler per rpc client
        #  Create those objects now if 

        # Partition the flowgraph+
        partition_config =  [val for _, val in self.service_blocks_map.items()]
        graphs, crossings = gr.graph_utils.partition(fg, partition_config)


        # # The python_on_whales up command doesn't support -f afaik
        # # docker.compose.up(["-d", "-f", compose_file.name])
        # os.system(f'docker compose -f {self.docker_compose_filename} up -d')
        # time.sleep(2)

        # For each host configuration
        
        for service, blocks in self.service_blocks_map.items():
            # Create the blocks in this client
            client = self.service_client_map[service]

            for b in blocks:
                randstr = uuid.uuid4().hex[:6]
                newblockname = b.name() + "_" + randstr

                # Setting this information is necessary to indicate to the block that
                # methods need to be serialized when called directly on the block
                # e.g. sink.data()
                b.set_rpc(newblockname, client)
                client.block_create(newblockname, b)

        # Create ZMQ connections between the ports involved in domain crossings
        for c, src_graph, dst_graph in crossings:
            src_zmq_block = zeromq.push_sink( c.src().port().itemsize(), "tcp://127.0.0.1:0")
            dst_zmq_block = zeromq.pull_source( c.dst().port().itemsize(), src_zmq_block.last_endpoint())
            src_graph.connect( (c.src().node(), c.src().port().index() ), (src_zmq_block, 0))
            dst_graph.connect( (dst_zmq_block, 0), (c.dst().node(), c.dst().port().index() ))

            src_client = list(self.service_client_map.items())[graphs.index(src_graph)][1]
            dst_client = list(self.service_client_map.items())[graphs.index(dst_graph)][1]
            randstr = uuid.uuid4().hex[:6]
            newblockname = src_zmq_block.name() + "_" + randstr
            src_zmq_block.set_rpc(newblockname, src_client)
            src_client.block_create(newblockname, src_zmq_block)
            randstr = uuid.uuid4().hex[:6]
            newblockname = src_zmq_block.name() + "_" + randstr
            dst_zmq_block.set_rpc(newblockname, dst_client)
            dst_client.block_create(newblockname, dst_zmq_block)

        for cnt, g in enumerate(graphs):
            fgname = uuid.uuid4().hex[:6]
            client = list(self.service_client_map.items())[cnt][1]
            client.flowgraph_create(fgname)
            self.client_fgname_map[client] = fgname

            # Connect the Blocks
            # in this runtime, everything should be remote
            for e in g.edges():
                src_client = e.src().node().rpc_client()
                dst_client = e.dst().node().rpc_client()
                edge_name = e.identifier()

                # This should always be the case for this runtime
                if src_client == dst_client:
                    print(f"same remote client {e.identifier()}")
                    src_client.flowgraph_connect(fgname,
                                               e.src().node().rpc_name(),
                                               e.src().port().name(),
                                               e.dst().node().rpc_name(),
                                               e.dst().port().name(),
                                               edge_name)
 
                else:
                    print("There should be no domain crossings yet")


        for cnt, g in enumerate(graphs):
            # Create a runtime for each container
            client = list(self.service_client_map.items())[cnt][1]
            fgname = self.client_fgname_map[client]

            client.runtime_create(fgname)
            client.runtime_initialize(fgname, fgname)


    def start(self):
        # Call start on each container
        for client, fgname in self.client_fgname_map.items():
            client.runtime_start(fgname)

    def wait(self):
        # Launch a thread and call wait
        # When one breaks out of wait() tell the others to stop
        #TODO
        pass
