
#
# Copyright 2011-2013 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#

import numpy as np
from newsched import gr, gr_unittest, blocks, math


class add_2_f32_1_f32(gr.sync_block):
    def __init__(self, dims=[1]):
        gr.sync_block.__init__(
            self,
            name="add 2 f32")

        self.add_port(gr.port_f("in1", gr.INPUT, dims))
        self.add_port(gr.port_f("in2", gr.INPUT, dims))
        self.add_port(gr.port_f("out", gr.OUTPUT, dims))

    def work(self, inputs, outputs):
        noutput_items = outputs[0].n_items
        
        outputs[0].produce(noutput_items)

        inbuf1 = self.get_input_array(inputs, 0)
        inbuf2 = self.get_input_array(inputs, 1)
        outbuf1 = self.get_output_array(outputs, 0)

        outbuf1[:] = inbuf1 + inbuf2

        return gr.work_return_t.WORK_OK

class add_ff_numpy(math.add_ff):
    def __init__(self, dims=[1]):
        math.add_ff.__init__(self, impl = math.add_ff.available_impl.cpu)

    def work(self, inputs, outputs):
        noutput_items = outputs[0].n_items
        
        outputs[0].produce(noutput_items)

        inbuf1 = gr.get_input_array(self, inputs, 0)
        inbuf2 = gr.get_input_array(self, inputs, 1)
        outbuf1 = gr.get_output_array(self, outputs, 0)

        outbuf1[:] = inbuf1 + inbuf2

        return gr.work_return_t.WORK_OK

class test_block_gateway(gr_unittest.TestCase):

    def test_add_ff_deriv(self):
        tb = gr.flowgraph()
        src0 = blocks.vector_source_f([1, 3, 5, 7, 9], False)
        src1 = blocks.vector_source_f([0, 2, 4, 6, 8], False)
        adder = add_ff_numpy()
        sink = blocks.vector_sink_f()
        tb.connect((src0, 0), (adder, 0))
        tb.connect((src1, 0), (adder, 1))
        tb.connect(adder, sink)
        tb.run()
        self.assertEqual(sink.data(), [1, 5, 9, 13, 17])

    # def test_add_f32(self):
    #     tb = gr.flowgraph()
    #     src0 = blocks.vector_source_f([1, 3, 5, 7, 9], False)
    #     src1 = blocks.vector_source_f([0, 2, 4, 6, 8], False)
    #     adder = add_2_f32_1_f32()
    #     sink = blocks.vector_sink_f()
    #     tb.connect((src0, 0), (adder, 0))
    #     tb.connect((src1, 0), (adder, 1))
    #     tb.connect(adder, sink)
    #     tb.run()
    #     self.assertEqual(sink.data(), [1, 5, 9, 13, 17])


    # def test_add_f32_vector(self):
    #     tb = gr.flowgraph()
    #     src0 = blocks.vector_source_f(10*[1, 3, 5, 7, 9], False, 5)
    #     src1 = blocks.vector_source_f(10*[0, 2, 4, 6, 8], False, 5)
    #     adder = add_2_f32_1_f32(dims=[5])
    #     sink = blocks.vector_sink_f(5)
    #     tb.connect((src0, 0), (adder, 0))
    #     tb.connect((src1, 0), (adder, 1))
    #     tb.connect(adder, sink)
    #     tb.run()
    #     self.assertEqual(sink.data(), 10*[1, 5, 9, 13, 17])


if __name__ == '__main__':
    gr_unittest.run(test_block_gateway)
