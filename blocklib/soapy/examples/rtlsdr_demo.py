import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.console
from pyqtgraph.dockarea import *

from gnuradio import gr, soapy, analog

from gnuradio.pyqtgraph.numpy import *

app = pg.mkQApp('DockArea Example')
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('pyqtgraph example: dockarea')

samp_rate = 1920000
freq = 751000000
gain = 40

fg = gr.flowgraph()

src = soapy.rtlsdr_source_c(samp_rate, freq, gain)
# src = analog.sig_source_f(samp_rate, analog.waveform_type.cos, 12, 1.0)
# snk = pg_plot_widget_f(100000, 'hello world')
snk = pg_time_sink_c('hello world', 100000, 1)
# snk2 = pg_time_sink_f('hello world 2', 100000)
fg.connect(src, 0, snk, 0)

d1 = Dock('a')
area.addDock(d1,'top')
d1.addWidget(snk.widget())

fg.start()

win.show()

# fg.wait()

if __name__ == '__main__':
    pg.exec()