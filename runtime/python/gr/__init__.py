
import os

try:
    from .runtime_python import *
except ImportError:
    dirname, filename = os.path.split(os.path.abspath(__file__))
    __path__.append(os.path.join(dirname, "bindings"))
    from .runtime_python import *

from .gateway import block, sync_block #, decim_block, interp_block

# CUDA specific python code will exception out as these objects aren't compiled in
try:
    buffer_cuda_properties.__init__ = buffer_cuda_properties.make
except:
    pass

# For newsched, connect and msg_connect are the same
graph.msg_connect = graph.connect
