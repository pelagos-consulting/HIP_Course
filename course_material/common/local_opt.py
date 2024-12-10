import numpy as np
import subprocess
from collections.abc import Iterable
from collections import OrderedDict

class LocalOpt():
    """Class to capture the result of an optimisation exercise for different algorithms."""
    
    def __init__(self,
            timings=None, 
            cmds=None, 
            local0=np.uint32(2**np.arange(0,10,1)),
            local1=np.uint32(2**np.arange(0,10,1)),
            local2=np.uint32(2**np.arange(0,1,1)),
            alg=None
                ):
        
        # Does the class have any data?
        self.has_data=False

        if timings is not None:
            assert(cmds is None)
            self.import_result(timings)
            
        if cmds is not None:
            
            assert(timings is None)
            assert(alg is not None)
            
            # Import the algorithm class
            self.alg = alg
            # Make the result
            self.make_result(cmds, local0, local1, local2)
        
    def make_mesh(self, local0, local1, local2):
        return np.meshgrid(local0, local1, local2, indexing="ij") 
        
    def insert_local(self, local0, local1, local2, times_ms, times_stdev, tflops, tflops_stdev):
        
        # Set variables
        self.local0 = np.array(local0)
        self.local1 = np.array(local1)
        self.local2 = np.array(local2)
            
        self.L0, self.L1, self.L2 = self.make_mesh(local0, local1, local2)
            
        # Data to plot
        self.times_ms = np.array(times_ms).reshape(self.L0.shape)
        self.times_stdev = np.array(times_stdev).reshape(self.L0.shape)

        self.tflops = np.array(tflops).reshape(self.L0.shape)
        self.tflops_stdev = np.array(tflops_stdev).reshape(self.L0.shape)
        
        # Signal that we have data
        self.has_data=True 
          
    def import_result(self, timing_data):
        self.insert_local(
            timing_data["local0"],
            timing_data["local1"],
            timing_data["local2"],
            timing_data["times_ms"],
            timing_data["times_stdev"],
            timing_data["tflops"],
            timing_data["tflops_stdev"]
        )
        
    def report_timings(self):
        
        """Find the minimum and maximum of timing data obtained from an experiement"""
        
        timing_data=self.export_result()
        
        print(f"Min time is {timing_data['min_ms']:.3f} ms, at the local size of" 
            f" ({timing_data['L0_min']},{timing_data['L1_min']},{timing_data['L2_min']}).")
        print(f"Max TFlop/S is {timing_data['max_tflops']} +/- {timing_data['max_tflops_stdev']}.")
        print(f"Max time is {timing_data['max_ms']:.3f} ms, at the local size of" 
            f" ({timing_data['L0_max']},{timing_data['L1_max']},{timing_data['L2_max']}).")
        print(f"Max time / min time == {timing_data['max_ms']/timing_data['min_ms']:.3f}")
  
    def make_result(self, 
                    cmds,
                    # Vectors of local sizes
                    local0=np.uint32(2**np.arange(0,10,1)),
                    local1=np.uint32(2**np.arange(0,10,1)),
                    local2=np.uint32(2**np.arange(0,1,1))):
        
        """Prepare the input file for local optimisation and run the problem"""
        
        # Make up the optimisation grid
        L0, L1, L2 = self.make_mesh(local0, local1, local2)   
    
        input_local = np.zeros((*L0.shape, 3), dtype=np.uint32)
        input_local[...,0] = L0
        input_local[...,1] = L1
        input_local[...,2] = L2
        
        # Write out to file
        input_local.tofile("input_block.dat")        
    
        # Add the --local-file flag
        if isinstance(cmds, Iterable) and not isinstance(cmds, str):
            temp_cmds = list(cmds) + ["--block_file"]
        else:
            temp_cmds=[cmds,"--block_file"]
        
        # Run the program 
        result = subprocess.run(temp_cmds)
        print(f"returncode is {result.returncode}")
        
        if (result.returncode==0):
        
            # Get the output data
            output_local = np.fromfile("output_block.dat", dtype=np.float64).reshape(
                local0.size, local1.size, local2.size, 2, order="C"
            )
            
            # Data to plot
            times_ms = output_local[...,0]
            times_stdev = output_local[...,1]

            # Calculate tflops here
            tflops, tflops_stdev = self.alg.get_tflops(times_ms, times_stdev)
            
            self.insert_local(local0, local1, local2, times_ms, times_stdev, tflops, tflops_stdev)
   
    def export_result(self):
        
        assert(self.has_data==True)
            
        # Find the minimum time
        index_min = np.nanargmin(self.times_ms)
        index_max = np.nanargmax(self.times_ms)

        timing_data = {
            "min_ms" : self.times_ms.ravel()[index_min],
            "std_ms" : self.times_stdev.ravel()[index_min],
            "max_tflops" : self.tflops.ravel()[index_min],
            "max_tflops_stdev" : self.tflops_stdev.ravel()[index_min],
            "L0_min" : int(self.L0.ravel()[index_min]),
            "L1_min" : int(self.L1.ravel()[index_min]),
            "L2_min" : int(self.L2.ravel()[index_min]),
            "max_ms" : self.times_ms.ravel()[index_max],
            "std_ms_max" : self.times_stdev.ravel()[index_max],
            "L0_max" : int(self.L0.ravel()[index_max]),
            "L1_max" : int(self.L1.ravel()[index_max]),
            "L2_max" : int(self.L2.ravel()[index_max]),
            "times_ms" : list(self.times_ms.ravel()),
            "times_stdev" : list(self.times_stdev.ravel()),
            "tflops" : list(self.tflops.ravel()),
            "tflops_stdev" : list(self.tflops_stdev.ravel()),
            "local0" : [int(n) for n in self.local0],
            "local1" : [int(n) for n in self.local1],
            "local2" : [int(n) for n in self.local2]
        }
            
        # Report timings
        #self.report_timings(timing_data)
        return timing_data