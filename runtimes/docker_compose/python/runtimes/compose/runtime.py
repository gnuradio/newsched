from gnuradio import gr
import uuid
# from python_on_whales import docker
from jinja2 import FileSystemLoader, Environment
import os
from gnuradio import zeromq
import time

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


        # Create ZMQ connections between the ports involved in domain crossings
        # These must be created directly on the remote host because
        # if they are created locally, the ports are assigned locally and cannot
        # be transferred
        for c, src_graph, dst_graph in crossings:

            # src_zmq_block = zeromq.push_sink( c.src().port().itemsize(), "tcp://127.0.0.1:0")

            src_client = list(self.service_client_map.items())[graphs.index(src_graph)][1]
            randstr = uuid.uuid4().hex[:6]
            src_blockname = 'zmq_push_sink' + "_" + randstr
            src_client.block_create_params(src_blockname, {'module': 'zeromq', 'id': 'push_sink', 
                'parameters': {'itemsize': c.src().port().itemsize(), 'address':"tcp://127.0.0.1:0"  }})
    
            lastendpoint = src_client.block_method(src_blockname, 'last_endpoint', {})
            dst_client = list(self.service_client_map.items())[graphs.index(dst_graph)][1]
            randstr = uuid.uuid4().hex[:6]
            dst_blockname = 'zmq_pull_source' + "_" + randstr
            dst_client.block_create_params(dst_blockname, {'module': 'zeromq', 'id': 'pull_source', 
                'parameters': {'itemsize': c.dst().port().itemsize(), 'address': lastendpoint  }})

            fgname = self.client_fgname_map[src_client]
            src_client.flowgraph_connect(fgname,
                                        c.src().node().rpc_name(),
                                        c.src().port().name(),
                                        src_blockname,
                                        "in",
                                        None)
            fgname = self.client_fgname_map[dst_client]
            dst_client.flowgraph_connect(fgname,
                                        dst_blockname, 
                                        "out",
                                        c.dst().node().rpc_name(),
                                        c.dst().port().name(),
                                        None)

        #TODO - figure out ipaddrs from the config
        ipaddr_a = "127.0.0.1"
        ipaddr_b = "127.0.0.1"

        # Create the remote runtimes and proxies back to the first
        client_a = list(self.service_client_map.items())[0][1]
        fgname_a = self.client_fgname_map[client_a]
        client_a.runtime_create(fgname_a)
        for cnt, g in enumerate(graphs):
            # Create a runtime for each container
            if (cnt > 0):
                client_b = list(self.service_client_map.items())[cnt][1]
                fgname_b = self.client_fgname_map[client_b]
                client_b.runtime_create(fgname_b)
                proxy_name_a, port_a = client_a.runtime_create_proxy(fgname_a, 0, True)
                proxy_name_b, port_b = client_b.runtime_create_proxy(fgname_b, 0, False)
                client_a.runtime_connect_proxy(proxy_name_a, ipaddr_b, port_b)
                client_b.runtime_connect_proxy(proxy_name_b, ipaddr_a, port_a)

        # Initialize the remote runtimes
        for cnt, g in enumerate(graphs):
            # Create a runtime for each container
            client = list(self.service_client_map.items())[cnt][1]
            fgname = self.client_fgname_map[client]
            client.runtime_initialize(fgname, fgname)


    def start(self):
        # # Call start on each container
        # for client, fgname in list(self.client_fgname_map.items())[::-1]:
        #     client.runtime_start(fgname)
        #     time.sleep(0.25)
        client, fgname = list(self.client_fgname_map.items())[0]
        client.runtime_start(fgname)

    def wait(self):
        # Launch a thread and call wait
        # When one breaks out of wait() tell the others to stop
        #TODO
        client, fgname = list(self.client_fgname_map.items())[0]
        client.runtime_wait(fgname)

    def stop(self):
        # Call start on each container
        # for client, fgname in self.client_fgname_map.items():
        #     client.runtime_stop(fgname)
        #     # time.sleep(0.25)
        client, fgname = list(self.client_fgname_map.items())[0]
        client.runtime_stop(fgname)
