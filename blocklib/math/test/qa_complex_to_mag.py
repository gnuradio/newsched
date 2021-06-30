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


class test_complex_to_mag (gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()

    def tearDown(self):
        self.tb = None

    def test_complex_to_mag(self):
        src_data = [-2 - 2j, -1 - 1j, -2 + 2j, -1 + 1j,
                    2 - 2j, 1 - 1j, 2 + 2j, 1 + 1j,
                    0 + 0j]

        exp_data = [abs(i) for i in src_data]

        src = blocks.vector_source_c(src_data)
        op = math.complex_to_mag()
        dst = blocks.vector_sink_f()

        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()
        result_data = dst.data()

        for exp, result in zip(exp_data, result_data):
            self.assertAlmostEqual(exp, result)


if __name__ == '__main__':
    gr_unittest.run(test_complex_to_mag)
