#!/usr/bin/env python
#
# Copyright 2014 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#


from gnuradio import gr, gr_unittest, blocks, zeromq
import time


class qa_zeromq_pushpull (gr_unittest.TestCase):

    def setUp(self):
        self.send_tb = gr.flowgraph()
        self.recv_tb = gr.flowgraph()
        self.send_rt = gr.runtime()
        self.recv_rt = gr.runtime()

    def tearDown(self):
        self.send_tb = None
        self.recv_tb = None

    def test_001(self):
        vlen = 10
        src_data = list(range(vlen)) * 100
        src = blocks.vector_source_f(src_data, False, vlen)
        zeromq_push_sink = zeromq.push_sink(gr.sizeof_float * vlen, "tcp://127.0.0.1:0")
        address = zeromq_push_sink.last_endpoint()
        zeromq_pull_source = zeromq.pull_source(gr.sizeof_float * vlen, address, 0)
        sink = blocks.vector_sink_f(vlen)
        self.send_tb.connect(src, zeromq_push_sink)
        self.recv_tb.connect(zeromq_pull_source, sink)
        self.send_rt.initialize(self.send_tb)
        self.recv_rt.initialize(self.recv_tb)
        self.recv_rt.start()
        time.sleep(0.5)
        self.send_rt.start()
        time.sleep(0.5)
        self.recv_rt.stop()
        self.send_rt.stop()
        # self.recv_rt.wait()
        # self.send_rt.wait()
        self.assertFloatTuplesAlmostEqual(sink.data(), src_data)


if __name__ == '__main__':
    gr_unittest.run(qa_zeromq_pushpull)
