import sys
import time
from tqdm import tqdm

progress_bar = tqdm(total=28, desc="Start!!!", position=0, leave=True)


sys.path.append(__file__[:-12])

progress_bar.update(1); progress_bar.set_description("Load Mycoord")
import Mycoord
progress_bar.update(1); progress_bar.set_description("Load Mymodels")
import Mymodels
progress_bar.update(1); progress_bar.set_description("Load MapPalette")
import MapPalette
progress_bar.update(1); progress_bar.set_description("Load Mymap")
import Mymap
progress_bar.update(1); progress_bar.set_description("Load Myfit")
import Myfit
progress_bar.update(1); progress_bar.set_description("Load Mysigmap")
import Mysigmap
progress_bar.update(1); progress_bar.set_description("Load Myspec")
import Myspec
progress_bar.update(1); progress_bar.set_description("Load Myspectrum")
import Myspectrum
progress_bar.update(1); progress_bar.set_description("Load Mycatalog")
import Mycatalog

progress_bar.update(1); progress_bar.set_description("Load importlib")
import importlib

progress_bar.update(1); progress_bar.set_description("ReLoad Mycatalog")
importlib.reload(Mycoord)
progress_bar.update(1); progress_bar.set_description("ReLoad MapPalette")
importlib.reload(MapPalette)
progress_bar.update(1); progress_bar.set_description("ReLoad Mymap")
importlib.reload(Mymap)
progress_bar.update(1); progress_bar.set_description("ReLoad Mymodels")
importlib.reload(Mymodels)
progress_bar.update(1); progress_bar.set_description("ReLoad Myfit")
importlib.reload(Myfit)
progress_bar.update(1); progress_bar.set_description("ReLoad Mysigmap")
importlib.reload(Mysigmap)
progress_bar.update(1); progress_bar.set_description("ReLoad Myspec")
importlib.reload(Myspec)
progress_bar.update(1); progress_bar.set_description("ReLoad Myspectrum")
importlib.reload(Myspectrum)
progress_bar.update(1); progress_bar.set_description("ReLoad Mycatalog")
importlib.reload(Mycatalog)

progress_bar.update(1); progress_bar.set_description("Load sub from Mycoord")
from Mycoord import *
progress_bar.update(1); progress_bar.set_description("Load sub from Mymodels")
from Mymodels import *
progress_bar.update(1); progress_bar.set_description("Load sub from MapPalette")
from MapPalette import *
progress_bar.update(1); progress_bar.set_description("Load sub from Mymap")
from Mymap import *
progress_bar.update(1); progress_bar.set_description("Load sub from Myfit")
from Myfit import *
progress_bar.update(1); progress_bar.set_description("Load sub from Mysigmap")
from Mysigmap import *
progress_bar.update(1); progress_bar.set_description("Load sub from Myspec")
from Myspec import *
progress_bar.update(1); progress_bar.set_description("Load sub from Myspectrum")
from Myspectrum import *
progress_bar.update(1); progress_bar.set_description("Load sub from Mycatalog")
from Mycatalog import *

print("Yourlib init successful!!!")