from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals


# the following modules are frequently used modules
# will not take huge time to import so clumped together
# you can add your module over here
# and can import easily with from pixutils import *

import os, re, time, shutil
from datetime import datetime as dt
from glob import glob
from itertools import *

try:
    from queue import Queue
except:
    from Queue import Queue
try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

from collections import defaultdict, OrderedDict, deque
from os.path import exists, isfile, basename, dirname, join

import json, cv2, dlib
import numpy as np
from multiprocessing.pool import ThreadPool
from threading import Thread