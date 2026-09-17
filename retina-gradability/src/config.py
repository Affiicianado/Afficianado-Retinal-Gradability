import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

EYEQ = Path("/kaggle/input/eyeq-512-preprocessed")
IMAGES = EYEQ / "eyeq_512"

WORK = Path("/kaggle/working")
RESULTS = WORK / "results"
FIGURES = WORK / "figures"
for d in (RESULTS, FIGURES):
    d.mkdir(exist_ok=True)

CLASSES = ["Good", "Usable", "Reject"]
GOOD, USABLE, REJECT = 0, 1, 2
SEED = 42

# confirm what is actually attached
print(sorted(p.name for p in EYEQ.iterdir()))