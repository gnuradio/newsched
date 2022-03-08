import os
from gnuradio.runtimes import distributed 
from gnuradio import blocks, math, gr
from gnuradio.schedulers import nbt

# An example of distributing a flowgraph across remote nodes
# These nodes could be containerized and/or live on remote hosts


nsamples = 1000
# input_data = [x%256 for x in list(range(nsamples))]
input_data = list(range(nsamples))

# Blocks are created locally, but will be replicated on the remote host
src = blocks.vector_source_f(input_data, False)
cp1 = blocks.copy()
mc = math.multiply_const_ff(1.0)
cp2 = blocks.copy()
snk = blocks.vector_sink_f()

fg1 = gr.flowgraph("FG On Local Host")
fg1.connect([src, cp1, mc, cp2, snk])

with distributed.runtime(os.path.join(os.path.dirname(__file__), 'test_config.yml')) as rt1:
    # There are 2 remote hosts defined in the config yml
    #  We assign groups of blocks where we want them to go
    rt1.assign_blocks("newsched1", [src, cp1, mc])
    rt1.assign_blocks("newsched2", [cp2, snk])
    rt1.initialize(fg1)

    # These calls on the local block are serialized to the remote block
    # This in effect means the local blocks are acting as proxy blocks
    mc.set_k(2.0)
    print(mc.k())

    # Run the flowgraph
    rt1.start()
    rt1.wait()

    # Get data from the block running on the remote host
    #  using the local block as a proxy
    rx_data = snk.data()

    print(f'Received {len(rx_data)} samples')