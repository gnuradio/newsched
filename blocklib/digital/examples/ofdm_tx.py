#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: josh
# GNU Radio version: 0.2.0

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr
#from gnuradio.filter import firdes
#from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
#from gnuradio.eng_arg import eng_float, intx
#from gnuradio import eng_notation
from gnuradio import pdu
import numpy



from gnuradio import qtgui

class ofdm_tx(Qt.QWidget):
    def start(self):
        self.fg.start()

    def stop(self):
        self.fg.stop()

    def wait(self):
        self.fg.wait()

    def connect(self,*args):
        return self.fg.connect(*args)

    def msg_connect(self,*args):
        return self.fg.connect(*args)

    def __init__(self):
        self.fg = gr.flowgraph("Not titled yet")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "ofdm_tx")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.pdu_stream_to_pdu_0 = pdu.stream_to_pdu_b( 96, impl=pdu.stream_to_pdu_b.cpu)
        self.digital_crc_append_0 = digital.crc_append( 32,79764919,4294967295,4294967295,True,True,False,0, impl=digital.crc_append.cpu)
        self.blocks_vector_source_0 = blocks.vector_source_b( numpy.random.randint(0, 255, 1000),True,1,[], impl=blocks.vector_source_b.cpu)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_vector_source_0, 0), (self.pdu_stream_to_pdu_0, 0))
        self.msg_connect((self.pdu_stream_to_pdu_0, 'pdus'), (self.digital_crc_append_0, 'in'))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "ofdm_tx")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(flowgraph_cls=ofdm_tx, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    fg = flowgraph_cls()
    rt = gr.runtime()

    rt.initialize(fg.fg)
    rt.start()

    fg.show()

    def sig_handler(sig=None, frame=None):
        rt.stop()
        rt.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
