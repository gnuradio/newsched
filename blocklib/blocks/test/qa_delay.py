#!/usr/bin/env python3
#
# Copyright 2004,2007,2010,2013 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


from newsched import gr, gr_unittest, blocks
# import pmt


class test_delay(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.flowgraph()
        self.rt = gr.runtime()

    def tearDown(self):
        self.tb = None
        self.rt = None

    def test_000(self):
        delta_t = 0
        tb = self.tb
        src_data = [float(x) for x in range(0, 100)]
        expected_result = list(delta_t * [0.0] + src_data)

        src = blocks.vector_source_f(src_data)
        op = blocks.delay(delta_t)
        dst = blocks.vector_sink_f()

        tb.connect(src, op)
        tb.connect(op, dst)
        self.rt.initialize(self.tb)
        self.rt.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    def test_010(self):
        delta_t = 3
        tb = self.tb
        src_data = [1.0+float(x) for x in range(0, 100)]
        expected_result = list(delta_t * [0.0] + src_data)

        src = blocks.vector_source_f(src_data)
        op = blocks.delay(delta_t)
        dst = blocks.vector_sink_f()

        tb.connect(src, op)
        tb.connect(op, dst)
        self.rt.initialize(self.tb)
        self.rt.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    # def test_020(self):
    #     tb = self.tb
    #     vector_sink = blocks.vector_sink_f(1)
    #     ref_sink = blocks.vector_sink_f(1)
    #     tags_strobe = blocks.tags_strobe(
    #         gr.sizeof_float * 1,
    #         pmt.intern("TEST"),
    #         100,
    #         pmt.intern("strobe"))
    #     head = blocks.head(gr.sizeof_float * 1, 10**5)
    #     delay = blocks.delay(gr.sizeof_float * 1, 100)
    #     tb.connect((delay, 0), (head, 0))
    #     tb.connect((head, 0), (vector_sink, 0))
    #     tb.connect((tags_strobe, 0), (delay, 0))
    #     tb.connect((tags_strobe, 0), (ref_sink, 0))
    #     tb.run()

    #     tags = vector_sink.tags()
    #     self.assertNotEqual(len(tags), 0)
    #     lastoffset = tags[0].offset - 100
    #     for tag in tags:
    #         newoffset = tag.offset
    #         self.assertEqual(newoffset, lastoffset + 100)
    #         lastoffset = newoffset


if __name__ == '__main__':
    gr_unittest.run(test_delay)
