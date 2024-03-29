#!/usr/bin/env python3
#
# Copyright 2012,2013 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


import cmath

from gnuradio import gr, gr_unittest, analog, blocks


class test_quadrature_demod(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_quad_demod_001(self):
        f = 1000.0
        fs = 8000.0

        src_data = []
        for i in range(200):
            ti = i / fs
            src_data.append(cmath.exp(2j * cmath.pi * f * ti))

        src_data = [0, ] + src_data # to account for history

        # f/fs is a quarter turn per sample.
        # Set the gain based on this to get 1 out.
        gain = 1.0 / (cmath.pi / 4)

        expected_result = [0, ] + 199 * [1.0]

        src = blocks.vector_source_c(src_data)
        op = analog.quadrature_demod(gain)
        dst = blocks.vector_sink_f()

        self.tb.connect(src, op)
        self.tb.connect(op, dst)
        self.tb.run()

        result_data = dst.data()
        self.assertComplexTuplesAlmostEqual(expected_result, result_data, 5)


if __name__ == '__main__':
    gr_unittest.run(test_quadrature_demod)
