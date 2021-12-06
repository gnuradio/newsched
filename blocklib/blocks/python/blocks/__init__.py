
import os

try:
    from .blocks_python import *
except ImportError:
    dirname, filename = os.path.split(os.path.abspath(__file__))
    __path__.append(os.path.join(dirname, "bindings"))
    from .blocks_python import *

# Import pure python code here
from .pytest_numpy import *
