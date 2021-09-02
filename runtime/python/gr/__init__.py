
import os

try:
    from .runtime_python import *
except ImportError:
    dirname, filename = os.path.split(os.path.abspath(__file__))
    __path__.append(os.path.join(dirname, "bindings"))
    from .runtime_python import *

# CUDA specific python code will exception out as these objects aren't compiled in
try:
    buffer_cuda_properties.__init__ = buffer_cuda_properties.make
except:
    pass
