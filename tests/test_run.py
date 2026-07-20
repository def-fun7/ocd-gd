# test_run.py
import sys
import os

# Temporary local path injection so python finds your src folder
sys.path.insert(0, os.path.abspath("src"))

import ocd_gd

print("Package imported successfully!")
print("Exposed items:", ocd_gd.__all__)
