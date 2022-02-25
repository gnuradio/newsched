from gnuradio import gr
import uuid
# from python_on_whales import docker
from jinja2 import FileSystemLoader, Environment
import os
import tempfile
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
        graph_info = gr.graph_utils.partition(fg, partition_config)


        # The python_on_whales up command doesn't support -f afaik
        # docker.compose.up(["-d", "-f", compose_file.name])
        os.system(f'docker compose -f {self.docker_compose_filename} up -d')
        time.sleep(2)

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
            

    def start(self):
        # Call start on each container
        for _, client in self.service_client_map.items():
            # fgname = self.host_flowgraph_names[hi]
            # client.start(fgname)
            pass


    def wait(self):
        # Launch a thread and call wait
        # When one breaks out of wait() tell the others to stop

        pass
