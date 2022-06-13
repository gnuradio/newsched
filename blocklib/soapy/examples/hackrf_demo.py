import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.console
from pyqtgraph.dockarea import *

from gnuradio import gr, soapy, analog

from gnuradio.pyqtgraph.numpy import *

import threading, time

app = pg.mkQApp('DockArea Example')
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('pyqtgraph example: dockarea')

samp_rate = 2000000
freq = 751000000
gain = 40

fg = gr.flowgraph()

src = soapy.hackrf_source_c(samp_rate, freq, gain, vga=30.0, amp=True)
# src = analog.sig_source_f(samp_rate, analog.waveform_t.cos, 12, 1.0)
# snk = pg_plot_widget_f(100000, 'hello world')
snk = pg_time_sink_c('hello world', 100000, 1)
# snk2 = pg_time_sink_f('hello world 2', 100000)
fg.connect(src, 0, snk, 0)

d1 = Dock('a')
area.addDock(d1,'top')
d1.addWidget(snk.widget())
d2 = Dock('console')

namespace = {'pg': pg, 'src': src }
c = pyqtgraph.console.ConsoleWidget(namespace=namespace)
d2.addWidget(c)
area.addDock(d2,'bottom')
fg.start()


# def thread_function(src):
#     newgain = 0.0
#     while True:
#         time.sleep(2)
#         src.set_gain(newgain)
#         newgain += 10.0

# x = threading.Thread(target=thread_function, args=(src,))
# x.daemon = True
# x.start()

win.show()

# fg.wait()

if __name__ == '__main__':
    pg.exec()