import os
import numpy as np
import sys
import json
sys.path.insert(0, os.path.abspath("../common"))
from local_opt import LocalOpt
from py_helper import MatMul
                
# Helper class to specify an experiment
class Exp:
    def __init__(
        self,
        cmds,
        local0=np.uint32(2**np.arange(0,10,1)),
        local1=np.uint32(2**np.arange(0,10,1)),
        local2=np.uint32(2**np.arange(0,1,1))):
        
        self.local0 = local0
        self.local1 = local1
        self.local2 = local2
        self.cmds = cmds.split()

### Modify this section          
output_file = "benchmark.json"
                
# Specify device types along with indices to try
dev_indices_gpu = [0]
dev_types={"gpu" : dev_indices_gpu}

mat_size = {}
# Read mat_size.hpp to extract the size
with open("mat_size.hpp") as fd:
    for line in fd:
        if line.startswith("#define"):
            splt=line.strip().split()
            mat_size[splt[1]]=int(splt[2])

alg_float=MatMul(mat_size["NCOLS_A"], mat_size["NROWS_C"], mat_size["NCOLS_C"], np.float32)
alg_double=MatMul(mat_size["NCOLS_A"], mat_size["NROWS_C"], mat_size["NCOLS_C"], np.float64)

# Specify programs to benchmark
experiments = {
    "Naive double precision" : "mat_mult_naive.exe",
    "Double precision" : "mat_mult_double.exe",
    "Single precision" : "mat_mult_float.exe",
    "Single with restrict" : "mat_mult_float_restrict.exe",
    "Shared A" : "mat_mult_shared_A.exe",
    "Shared B" : "mat_mult_shared_B.exe",
    "Transpose B" : "mat_mult_BT.exe",
    "Transpose A" : "mat_mult_AT.exe",
    "Tile shared A" : "mat_mult_tile_shared_A.exe",
    "Tile shared B" : "mat_mult_tile_shared_B.exe",
    "Tile shared AB" : "mat_mult_tile_shared_AB.exe",
    "Tile shared A vector" : "mat_mult_tile_shared_A_vector.exe",
    "Tile shared B vector" : "mat_mult_tile_shared_B_vector.exe",
    "Tile shared AB vector" : "mat_mult_tile_shared_AB_vector.exe",
    "HIPBlas" : "mat_mult_float_hipblas.exe",
    "HIPBlas MD" : "mat_mult_float_md_hipblas.exe"
}

# Make up the input specification
inputSpec={}
for exp, executable in experiments.items():
    for d, indices in dev_types.items():
        for i in indices:
            label=f"{exp} ({d.upper()})[{i}]"
            cmds=f"{executable} {i}"
            if "HIPBlas" in exp:
                inputSpec[label]=Exp(cmds, local0=2**(np.arange(0,1,1)), local1=2**(np.arange(0,1,1)))
            else:
                inputSpec[label]=Exp(cmds)
                
# Special cases here
#inputSpec["CLBlast MD (GPU)"]=Exp("mat_mult_clblast_md.exe -gpu", local0=2**(np.arange(0,1,1)), local1=2**(np.arange(0,1,1)))
        
### Don't modify any more commands
       
results = {}
    
# Now process all commands
for label, spec in inputSpec.items():
    print(f"{label}, {spec.cmds}")
    # Make and run an optimisation experiment
    alg=alg_float
    if "double" in label.lower():
        alg=alg_double

    temp=LocalOpt(
        cmds=spec.cmds, 
        local0=spec.local0,
        local1=spec.local1,
        local2=spec.local2,
        alg=alg
    )
    
    if temp.has_data:
        # Results contains dictionary of timing results
        results[label]=temp.export_result()

# Put all results here
result_json=json.dumps(results)

with open(output_file, "w") as fd:
    fd.write(result_json)
                
