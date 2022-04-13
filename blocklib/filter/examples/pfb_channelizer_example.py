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

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.kernel.filter import firdes
import sip
from gnuradio import analog
from gnuradio import filter
from gnuradio import gr
#from gnuradio.filter import firdes
#from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
#from gnuradio.eng_arg import eng_float, intx
#from gnuradio import eng_notation
from gnuradio import math
from gnuradio import streamops
from gnuradio.kernel import fft as fftk
from gnuradio.kernel import filter as filterk



from gnuradio import qtgui

class pfb_channelizer_example(Qt.QWidget):
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

        self.settings = Qt.QSettings("GNU Radio", "pfb_channelizer_example")

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
        self.streamops_deinterleave_0 = streamops.deinterleave( 2,1,0, impl=streamops.deinterleave.cpu)
        self.qtgui_freq_sink_0_1 = qtgui.freq_sink_c(1024,5,0,samp_rate,"ALL",1)
        self.qtgui_freq_sink_0_1.set_update_time(0.10)
        self.qtgui_freq_sink_0_1.set_y_axis(-140, 10)
        self.qtgui_freq_sink_0_1.set_y_label('Relative Gain', 'dB')
        # self.qtgui_freq_sink_0_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_0_1.enable_autoscale(False)
        self.qtgui_freq_sink_0_1.enable_grid(False)
        self.qtgui_freq_sink_0_1.set_fft_average(1.0)
        self.qtgui_freq_sink_0_1.enable_axis_labels(True)
        self.qtgui_freq_sink_0_1.enable_control_panel(False)
        self.qtgui_freq_sink_0_1.set_fft_window_normalized(False)
        self._qtgui_freq_sink_0_1_win = sip.wrapinstance(self.qtgui_freq_sink_0_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_0_1_win)
        self.qtgui_freq_sink_0_0_0 = qtgui.freq_sink_c(1024,5,0,samp_rate / 2,"2",1)
        self.qtgui_freq_sink_0_0_0.set_update_time(0.10)
        self.qtgui_freq_sink_0_0_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_0_0_0.set_y_label('Relative Gain', 'dB')
        # self.qtgui_freq_sink_0_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_0_0_0.enable_autoscale(False)
        self.qtgui_freq_sink_0_0_0.enable_grid(False)
        self.qtgui_freq_sink_0_0_0.set_fft_average(1.0)
        self.qtgui_freq_sink_0_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_0_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_0_0_0.set_fft_window_normalized(False)
        self._qtgui_freq_sink_0_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_0_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_0_0_0_win)
        self.qtgui_freq_sink_0_0 = qtgui.freq_sink_c(1024,5,0,samp_rate / 2,"1",1)
        self.qtgui_freq_sink_0_0.set_update_time(0.10)
        self.qtgui_freq_sink_0_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_0_0.set_y_label('Relative Gain', 'dB')
        # self.qtgui_freq_sink_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_0_0.enable_autoscale(False)
        self.qtgui_freq_sink_0_0.enable_grid(False)
        self.qtgui_freq_sink_0_0.set_fft_average(1.0)
        self.qtgui_freq_sink_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_0_0.set_fft_window_normalized(False)
        self._qtgui_freq_sink_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_0_0_win)
        self.math_add_0 = math.add_cc( 3,1, impl=math.add_cc.cpu)
        self.filter_pfb_channelizer_0 = filter.pfb_channelizer_cc( 2,filterk.firdes.low_pass(1.0, 2.0, 0.45, 0.01),1, impl=filter.pfb_channelizer_cc.cpu)
        self.analog_sig_source_0_0 = analog.sig_source_c( samp_rate,analog.waveform_type.cos,1000,1.0,0,0, impl=analog.sig_source_c.cpu)
        self.analog_sig_source_0 = analog.sig_source_c( samp_rate,analog.waveform_type.cos,10000,1.0,0,0, impl=analog.sig_source_c.cpu)
        self.analog_noise_source_0 = analog.noise_source_c( analog.noise_type.gaussian,0.1,0, impl=analog.noise_source_c.cpu)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_0, 0), (self.math_add_0, 2))
        self.connect((self.analog_sig_source_0, 0), (self.math_add_0, 1))
        self.connect((self.analog_sig_source_0_0, 0), (self.math_add_0, 0))
        self.connect((self.filter_pfb_channelizer_0, 0), (self.qtgui_freq_sink_0_0, 0))
        self.connect((self.filter_pfb_channelizer_0, 1), (self.qtgui_freq_sink_0_0_0, 0))
        self.connect((self.math_add_0, 0), (self.qtgui_freq_sink_0_1, 0))
        self.connect((self.math_add_0, 0), (self.streamops_deinterleave_0, 0))
        self.connect((self.streamops_deinterleave_0, 1), (self.filter_pfb_channelizer_0, 1))
        self.connect((self.streamops_deinterleave_0, 0), (self.filter_pfb_channelizer_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "pfb_channelizer_example")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(flowgraph_cls=pfb_channelizer_example, options=None):

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
