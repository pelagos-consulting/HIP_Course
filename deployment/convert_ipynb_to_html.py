#!/usr/bin/env python3

import os
import subprocess

for root, dirs, files in os.walk(".", topdown=False):
    for fname in files:
        base, dot, ext = fname.rpartition(".")
        #print(base)
        if ext == "ipynb" and not ".ipynb_checkpoints" in root:
            subprocess.run(f"jupyter nbconvert --to html --embed-images {os.path.join(root,fname)}", shell=True)
