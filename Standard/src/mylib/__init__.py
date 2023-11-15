import sys
sys.path.append(__file__[:-12])
import Mycoord
import Mymodels
import MapPalette
import Mymap
import Myfit
import Mysigmap
import Myspec
import Myspectrum

import importlib

importlib.reload(Mycoord)
importlib.reload(MapPalette)
importlib.reload(Mymap)
importlib.reload(Mymodels)
importlib.reload(Myfit)
importlib.reload(Mysigmap)
importlib.reload(Myspec)
importlib.reload(Myspectrum)

from Mycoord import *
from Mymodels import *
from MapPalette import *
from Mymap import *
from Myfit import *
from Mysigmap import *
from Myspec import *
from Myspectrum import *

print("Yourlib init successful!!!")