from gnuradio.runtimes import docker_compose 
from gnuradio import blocks, math, gr
from gnuradio.schedulers import nbt
from gnuradio.rpc import rest

import os

nsamples = 1000
input_data = list(range(nsamples))

src = blocks.vector_source_f(input_data, False)
cp1 = blocks.copy(gr.sizeof_float)
mc = math.multiply_const_ff(1.0)
cp2 = blocks.copy(gr.sizeof_float)
snk = blocks.vector_sink_f()

fg1 = gr.flowgraph("FG On Local Host")
fg1.connect([src, cp1, mc, cp2, snk])

client1 = rest.client("127.0.0.1", 8000)
client2 = rest.client("127.0.0.1", 8001)

with docker_compose.runtime(os.path.join(os.path.dirname(__file__), 'docker-compose.yml')) as rt1:
    rt1.assign_rpc_client("newsched1", client1)
    rt1.assign_rpc_client("newsched2", client2)
    rt1.assign_blocks("newsched1", [src, cp1, mc])
    rt1.assign_blocks("newsched2", [cp2, snk])
    rt1.initialize(fg1)

    mc.set_k(2.0)
    print(mc.k())

    rt1.start()
    rt1.wait()

    import time
    time.sleep(3)
    # Will need to get the data serialized from the remote host
    rx_data = snk.data()

    print(rx_data)