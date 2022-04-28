#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: josh
# GNU Radio version: 0.2.0

from gnuradio import blocks
from gnuradio import gr
#from gnuradio.filter import firdes
#from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
#from gnuradio.eng_arg import eng_float, intx
#from gnuradio import eng_notation
from gnuradio import math
from gnuradio import pdu
from gnuradio.kernel import digital as digitalk
import numpy


def snipfcn_snippet_0(fg, rt=None):
    from matplotlib import pyplot as plt

    plt.plot(fg.snk1.data())
    plt.plot(fg.snk2.data(), 'r:')
    plt.show()


def snippets_main_after_stop(fg, rt=None):
    snipfcn_snippet_0(fg, rt)


class ofdm_tx(gr.flowgraph):

    def __init__(self):
        gr.flowgraph.__init__(self, "Not titled yet")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000
        self.header_mod = header_mod = digitalk.constellation_bpsk()

        ##################################################
        # Blocks
        ##################################################
        self.snk2 = blocks.vector_sink_f(
        1,1024, impl=blocks.vector_sink_f.cpu)
        self.snk1 = blocks.vector_sink_f(
        1,1024, impl=blocks.vector_sink_f.cpu)
        self.pdu_stream_to_pdu_0 = pdu.stream_to_pdu_f(
        96, impl=pdu.stream_to_pdu_f.cpu)
        self.pdu_pdu_to_stream_0 = pdu.pdu_to_stream_f(
         impl=pdu.pdu_to_stream_f.cpu)
        self.math_multiply_const_0_0 = math.multiply_const_ff(
        1.0,1, impl=math.multiply_const_ff.cpu)
        self.math_multiply_const_0 = math.multiply_const_ff(
        1.0,1, impl=math.multiply_const_ff.cpu)
        self.math_multiply_const_0.set_work_mode(gr.work_mode_t.PDU)
        self.blocks_vector_source_0 = blocks.vector_source_f(
        numpy.random.randint(0, 255, 96*10),False,1,[], impl=blocks.vector_source_f.cpu)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_vector_source_0, 0), (self.math_multiply_const_0_0, 0))
        self.connect((self.blocks_vector_source_0, 0), (self.pdu_stream_to_pdu_0, 0))
        self.connect((self.math_multiply_const_0_0, 0), (self.snk1, 0))
        self.connect((self.pdu_pdu_to_stream_0, 0), (self.snk2, 0))
        self.msg_connect((self.math_multiply_const_0, 'pdus_out'), (self.pdu_pdu_to_stream_0, 'pdus'))
        self.msg_connect((self.pdu_stream_to_pdu_0, 'pdus'), (self.math_multiply_const_0, 'pdus_in'))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod




def main(flowgraph_cls=ofdm_tx, options=None):
    fg = flowgraph_cls()
    rt = gr.runtime()


    rt.initialize(fg)

    rt.start()

    try:
        rt.wait()
    except KeyboardInterrupt:
        rt.stop()
        rt.wait()
    snippets_main_after_stop(fg, rt)

if __name__ == '__main__':
    main()
