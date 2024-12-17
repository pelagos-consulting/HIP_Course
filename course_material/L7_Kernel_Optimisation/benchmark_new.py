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

# Algorithm to test
alg=MatMul(1024,1024,1024,np.float32)

# Specify programs to benchmark
experiments = {
    "Single precision naive" : "mat_mult_naive.exe",
    "Single precision coalesced1" : "mat_mult_coalesced1.exe",
    "Single precision coalesced2" : "mat_mult_coalesced2.exe",
    "Double precision" : "mat_mult_double.exe",
    "Single precision" : "mat_mult_float.exe",
    "Shared A" : "mat_mult_shared_A.exe",
    "Shared A col" : "mat_mult_shared_A_col.exe",
    "Shared A flat alt" : "mat_mult_shared_A_flat_alt.exe",
    "Shared A flat col" : "mat_mult_shared_A_flat_col.exe",
    "Tile shared_AB_flat_col" : "mat_mult_tile_shared_AB_flat_col.exe",
    "Tile shared_AB" : "mat_mult_tile_shared_AB.exe"
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
    temp=LocalOpt(
        cmds=spec.cmds, 
        local0=spec.local0,
        local1=spec.local1,
        local2=spec.local2,
        alg=alg)
    
    if temp.has_data:
        # Results contains dictionary of timing results
        results[label]=temp.export_result()

# Put all results here
result_json=json.dumps(results)

with open(output_file, "w") as fd:
    fd.write(result_json)
                
