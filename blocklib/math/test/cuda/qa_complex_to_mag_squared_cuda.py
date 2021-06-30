#!/usr/bin/env python3
#
# Copyright 2012,2013 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


from newsched import gr, gr_unittest, blocks, math


class test_complex_to_mag_squared (gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()

    def tearDown(self):
        self.tb = None

    def test_complex_to_mag_squared(self):
        tb = self.tb
        src_data = [-2 - 2j, -1 - 1j, -2 + 2j, -1 + 1j,
                    2 - 2j, 1 - 1j, 2 + 2j, 1 + 1j,
                    0 + 0j]

        expected_data = [abs(i)**2 for i in src_data]

        src = blocks.vector_source_c(src_data)
        op = math.complex_to_mag_squared(impl=math.complex_to_mag_squared.cuda)
        dst = blocks.vector_sink_f()

        tb.connect(src, op).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.H2D))
        tb.connect(op, dst).set_custom_buffer(gr.cuda_buffer_properties.make(gr.cuda_buffer_type.D2H))
        tb.run()
        dst_data = dst.data()

        for exp, dst in zip(expected_data, dst_data):
            self.assertAlmostEqual(exp, dst)


if __name__ == '__main__':
    gr_unittest.run(test_complex_to_mag_squared)
