#!/usr/bin/env python3
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


from newsched import gr, gr_unittest, math, blocks
# import pmt


class test_multiply_cuda(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()

    def tearDown(self):
        self.tb = None

    def test_multiply_ss(self):
        tb = self.tb
        src_data1 = [x for x in range(0, 100)]
        src_data2 = [x for x in range(0, 100)]
        expected_data = [src_data1[x] * src_data2[x] for x in range(0, 100)]

        src1 = blocks.vector_source_s(src_data1)
        src2 = blocks.vector_source_s(src_data2)
        op = math.multiply_ss(impl=math.multiply_ss.cuda)
        dst = blocks.vector_sink_s()

        tb.connect(src1,0, op,0).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(src2,0, op,1).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.D2H))
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_data, dst_data)

    def test_multiply_ii(self):
        tb = self.tb
        src_data1 = [x for x in range(0, 100)]
        src_data2 = [x for x in range(0, 100)]
        expected_data = [src_data1[x] * src_data2[x] for x in range(0, 100)]

        src1 = blocks.vector_source_i(src_data1)
        src2 = blocks.vector_source_i(src_data2)
        op = math.multiply_ii(impl=math.multiply_ii.cuda)
        dst = blocks.vector_sink_i()

        tb.connect(src1,0, op,0).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(src2,0, op,1).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.D2H))
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_data, dst_data)

    def test_multiply_ff(self):
        tb = self.tb
        src_data1 = [float(x) for x in range(0, 100)]
        src_data2 = [float(x) for x in range(0, 100)]
        expected_data = [src_data1[x] * src_data2[x] for x in range(0, 100)]

        src1 = blocks.vector_source_f(src_data1)
        src2 = blocks.vector_source_f(src_data2)
        op = math.multiply_ff(impl=math.multiply_ff.cuda)
        dst = blocks.vector_sink_f()

        tb.connect(src1,0, op,0).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(src2,0, op,1).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.D2H))
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_data, dst_data)

    def test_multiply_cc(self):
        tb = self.tb
        src_data1 = [complex(x,-x) for x in range(0, 100)]
        src_data2 = [complex(x,-x) for x in range(0, 100)]
        expected_data = [src_data1[x] * src_data2[x] for x in range(0, 100)]
        
        src1 = blocks.vector_source_c(src_data1)
        src2 = blocks.vector_source_c(src_data2)
        op = math.multiply_cc(impl=math.multiply_cc.cuda)
        dst = blocks.vector_sink_c()

        tb.connect(src1,0, op,0).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(src2,0, op,1).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.D2H))
        tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_data, dst_data)


if __name__ == '__main__':
    gr_unittest.run(test_multiply_cuda)
