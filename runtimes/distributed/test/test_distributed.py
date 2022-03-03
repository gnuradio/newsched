from gnuradio.runtimes import distributed 
from gnuradio import blocks, math, gr
from gnuradio.schedulers import nbt


import os

nsamples = 1000
# input_data = [x%256 for x in list(range(nsamples))]
input_data = list(range(nsamples))

src = blocks.vector_source_f(input_data, False)
cp1 = blocks.copy()
mc = math.multiply_const_ff(1.0)
cp2 = blocks.copy()
snk = blocks.vector_sink_f()

fg1 = gr.flowgraph("FG On Local Host")
fg1.connect([src, cp1, mc, cp2, snk])
# fg1.connect([src, cp1, cp2, snk])

with distributed.runtime(os.path.join(os.path.dirname(__file__), 'test_config.yml')) as rt1:
    rt1.assign_blocks("newsched1", [src, cp1, mc])
    # rt1.assign_blocks("newsched1", [src, cp1])
    rt1.assign_blocks("newsched2", [cp2, snk])
    rt1.initialize(fg1)

    mc.set_k(2.0)
    print(mc.k())

    rt1.start()
    rt1.wait()
    # rt1.stop()

    # import time
    # time.sleep(1)
    # Will need to get the data serialized from the remote host
    rx_data = snk.data()

    print(f'Received {len(rx_data)} samples')