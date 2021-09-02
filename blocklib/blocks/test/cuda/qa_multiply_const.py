#!/usr/bin/env python3
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


from newsched import gr, gr_unittest, blocks
# import pmt


class test_multiply_const(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()

    def tearDown(self):
        self.tb = None

    def test_000(self):
        k = 4.0
        tb = self.tb
        src_data = [float(x) for x in range(0, 100)]
        expected_data = [k * float(x) for x in range(0, 100)]

        src = blocks.vector_source_f(src_data)
        op = blocks.multiply_const_ff(k, impl=blocks.multiply_const_ff.cuda)
        dst = blocks.vector_sink_f()

        tb.connect(src, op).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2H))
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_data, dst_data)

    def test_001(self):
        k = 2.0 + 3.0j
        tb = self.tb
        src_data = [complex(x,-x) for x in range(0, 100)]
        expected_data = [k * x for x in src_data]

        src = blocks.vector_source_c(src_data)
        op = blocks.multiply_const_cc(k, impl=blocks.multiply_const_cc.cuda)
        dst = blocks.vector_sink_c()

        tb.connect(src, op).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2H))
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_data, dst_data)

if __name__ == '__main__':
    gr_unittest.run(test_multiply_const)
