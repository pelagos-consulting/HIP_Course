import numpy as np
import ast
import math
from matplotlib import pyplot as plt
from ipywidgets import widgets

# Import axes machinery
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from collections.abc import Iterable
from collections import OrderedDict

def load_defines(fname):
    '''Load all defines from a header file'''
    defines = {}
    with open(fname, "r") as fd:
        for line in fd:
            if line.startswith("#define"):
                tokens=line.split()
                defines[tokens[1]]=ast.literal_eval(tokens[2])
    return defines

class MatMul:
    '''Implements the means to define and test and run a matrix multiplication'''
    def __init__(self, NCOLS_A, NROWS_C, NCOLS_C, dtype):
        self.NCOLS_A = NCOLS_A
        self.NROWS_C = NROWS_C
        self.NCOLS_C = NCOLS_C
        self.dtype = dtype
  
    def run_compute(self):
        # Run the compute
        self.C = np.matmul(self.A, self.B, dtype = self.dtype)

    def load_data(self):
        # Load binary arrays from file if they have already been written
        self.A = np.fromfile("array_A.dat", dtype=self.dtype).reshape((self.NROWS_C, self.NCOLS_A))
        self.B = np.fromfile("array_B.dat", dtype=self.dtype).reshape((self.NCOLS_A, self.NCOLS_C))
        self.run_compute()

    def make_data(self):
    
        # A is of size (NROWS_C, NCOLS_A)
        # B is of size (NCOLS_A, NCOLS_C)    
        # C is of size (NROWS_C, NCOLS_C)

        # Make up the arrays A, B, and C
        self.A = np.random.random(size = (self.NROWS_C, self.NCOLS_A)).astype(self.dtype)
        self.B = np.random.random(size = (self.NCOLS_A, self.NCOLS_C)).astype(self.dtype)

        # Make up the answer
        self.run_compute()

        # Write out the arrays as binary files
        self.A.tofile("array_A.dat")
        self.B.tofile("array_B.dat")

    def check_data(self):
        # Make sure we have the solution
        assert hasattr(self, "C"), "Must run make_data() before check_data()."
        
        # Read in the output from file
        self.C_out = np.fromfile("array_C.dat", dtype=self.dtype).reshape((self.NROWS_C, self.NCOLS_C))

        # Make plots
        fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)

        # Data to plot
        data = [self.C, self.C_out, np.abs(self.C-self.C_out)]

        # Labels to plot
        labels = ["Numpy", "Program", "Absolute residual"]

        for n, value in enumerate(data):
            # Plot the graph
            ax = axes[n]
            im = ax.imshow(value)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            # Set labels on things
            ax.set_xlabel("Dimension 1 (columns)")
            ax.set_ylabel("Dimension 0 (rows)")
            ax.set_title(labels[n])

            # Put a color bar on the plot
            plt.colorbar(mappable=im, cax=cax)

        fig.tight_layout()
        plt.show()
        
class Hadamard:
    
    def __init__(self, NROWS_F, NCOLS_F, dtype):
        self.NROWS_F = NROWS_F
        self.NCOLS_F = NCOLS_F
        self.dtype = dtype
        
    def run_compute(self):
        # Compute the transformation
        self.F = self.D*self.E

    def load_data(self):
        # Read in the output from OpenCL
        self.D = np.fromfile("array_D.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))
        self.E = np.fromfile("array_E.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))
        self.run_compute()

    def make_data(self):
    
        # D is of size (NROWS_F, NCOLS_F)
        # E is of size (NCOLS_F, NCOLS_F)    
        # F is of size (NROWS_F, NCOLS_F)

        # Make up the arrays A, B, and C
        self.D = np.random.random(size = (self.NROWS_F, self.NCOLS_F)).astype(self.dtype)
        self.E = np.random.random(size = (self.NROWS_F, self.NCOLS_F)).astype(self.dtype)

        # Make up the answer using Hadamard multiplication
        self.run_compute()

        # Write out the arrays as binary files
        self.D.tofile("array_D.dat")
        self.E.tofile("array_E.dat")

    def check_data(self):
        # Make sure we have the solution
        assert hasattr(self, "F"), "Must run make_data() or load_data before check_data()."

        # Read in the output from OpenCL
        self.F_out = np.fromfile("array_F.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))

        # Make plots
        fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)

        # Data to plot
        data = [self.F, self.F_out, np.abs(self.F-self.F_out)]

        # Labels to plot
        labels = ["Numpy", "Program", "Absolute residual"]

        for n, value in enumerate(data):
            # Plot the graph
            ax = axes[n]
            im = ax.imshow(value)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            # Set labels on things
            ax.set_xlabel("Dimension 1 (columns)")
            ax.set_ylabel("Dimension 0 (rows)")
            ax.set_title(labels[n])

            # Put a color bar on the plot
            plt.colorbar(mappable=im, cax=cax)

        fig.tight_layout()
        plt.show()
        
class LocalOpt():

    def report_timings(self):
        assert hasattr(self, "timing_data"), "Must execute run_problem() before report_timings()."
        
        print(f"Min time is {self.timing_data['min_ms']:.3f} ms, at the local size of" 
            f" ({self.timing_data['L0_min']},{self.timing_data['L1_min']},{self.timing_data['L2_min']}).")
        print(f"Max time is {self.timing_data['max_ms']:.3f} ms, at the local size of" 
            f" ({self.timing_data['L0_max']},{self.timing_data['L1_max']},{self.timing_data['L2_max']}).")
        print(f"Max time / min time == {self.timing_data['max_ms']/self.timing_data['min_ms']:.3f}")
        
    def run_problem(self, 
                    cmds,
                    # Vectors of local sizes
                    local0=np.uint32(2**np.arange(0,10,1)),
                    local1=np.uint32(2**np.arange(0,10,1)),
                    local2=np.uint32(2**np.arange(0,1,1)),
                    plot=True):
        
        self.local0 = local0
        self.local1 = local1
        self.local2 = local2
        
        # Make up the optimisation grid
        self.L0, self.L1, self.L2 = np.meshgrid(self.local0, self.local1, self.local2, indexing="ij")    
    
        self.nexperiments = self.L0.size
        self.input_local = np.zeros((*self.L0.shape, 3), dtype=np.uint32)
        self.input_local[...,0] = self.L0
        self.input_local[...,1] = self.L1
        self.input_local[...,2] = self.L2
        
        # Write out to file
        self.input_local.tofile("input_local.dat")        
    
        # Add the --local-file flag
        if isinstance(cmds, Iterable) and not isinstance(cmds, str):
            temp_cmds = list(cmds) + ["--local_file"]
        else:
            temp_cmds=[cmds,"--local_file"]
        
        # Run the program 
        result = subprocess.run(temp_cmds)
        print(f"returncode is {result.returncode}")
        
        if (result.returncode==0):
        
            # Get the output data
            self.output_local = np.fromfile("output_local.dat", dtype=np.float64).reshape(
                self.local0.size, self.local1.size, self.local2.size, 2, order="C"
            )
            
            # Data to plot
            #data = [self.output_local[:,:,0], self.output_local[:,:,1]]
            times_ms = self.output_local[...,0]
            times_stdev = self.output_local[...,1]
            data = [times_ms] #, times_stdev]
            
            # Find the minimum time
            index_min = np.nanargmin(times_ms)
            index_max = np.nanargmax(times_ms)
            
            self.timing_data = {
                "min_ms" : times_ms.ravel()[index_min],
                "std_ms" : times_stdev.ravel()[index_min],
                "L0_min" : self.L0.ravel()[index_min],
                "L1_min" : self.L1.ravel()[index_min],
                "L2_min" : self.L2.ravel()[index_min],
                "max_ms" : times_ms.ravel()[index_max],
                "std_ms_max" : times_stdev.ravel()[index_max],
                "L0_max" : self.L0.ravel()[index_max],
                "L1_max" : self.L1.ravel()[index_max],
                "L2_max" : self.L2.ravel()[index_max]
            }
            
            # Report timings
            self.report_timings()

            if plot:
            
                # Make plots
                fig, axes = plt.subplots(1, 1, figsize=(6,6), sharex=True, sharey=True)

                # Labels to plot
                labels = ["Time (ms)", "Error (ms)"]

                for n, value in enumerate(data):
                    # Plot the graph
                    #ax = axes[n]
                    ax=axes
                
                    indices = np.where(~np.isnan(value))
                    bad_indices=np.where(np.isnan(value))
    
                    #value[bad_indices]=1.0
                    value=np.log10(value)
                    #value[bad_indices]=np.nan
                
                    min_data = np.min(value[indices])
                    max_data = np.max(value[indices])
                
                    im = ax.imshow(value[...,0], vmin=min_data, vmax=max_data, origin="lower")
                    #ax.contour(value, 20, origin="lower")
                
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)

                    # Set labels on things
                    ax.set_xticks(np.arange(0,self.local1.size,1))
                    ax.set_yticks(np.arange(0,self.local0.size,1))
                    ax.set_xticklabels([str(x) for x in self.local1])
                    ax.set_yticklabels([str(x) for x in self.local0])
                    ax.set_xlabel("Local size (dimension 1)")
                    ax.set_ylabel("Local size (dimension 0)")
                    ax.set_title(labels[n])

                    # Put a color bar on the plot
                    plt.colorbar(mappable=im, cax=cax)

                fig.tight_layout()
                plt.show()
                
            return self.timing_data

        
class TimingPlotData:
    def __init__(self):
        self.labels=[]
        self.colours=[]
        self.speedups=[]
        self.errors=[]
        
    def ingest(self, speedup, error, label, colour):
        self.labels.append(label)
        self.colours.append(colour)
        self.speedups.append(speedup)
        self.errors.append(error)
        
    def num_items(self):
        return len(self.speedups)
        
class TimingResults:
    
    def __init__(self):
        self.results = OrderedDict()
    
    def add_result(self, result, label, benchmark=False):
        if result is not None:
            if len(self.results)==0 or benchmark:
                self.benchmark_label = label
            self.results[label] = result
    
    def delete_result(self, label):
        if label in self.results:
            del self.results["label"]
    
    def plot_results(self, highlight_key=None):
        
        if len(self.results)>0:
            
            if highlight_key is None:
                highlight_key = self.benchmark_label
            
            # Sort by GPU results and CPU results
            
            # Make up timing results
            [fig, ax] = plt.subplots(2, 1, figsize=(6,6))
            
            t_bench = self.results[self.benchmark_label]["min_ms"]
            dt_bench = self.results[self.benchmark_label]["std_ms"]
            
            gpu_data = TimingPlotData()
            cpu_data = TimingPlotData()
            
            for key, result in self.results.items():
                
                # Get time
                t = result["min_ms"]
                # Get error in time
                dt = result["std_ms"]
                
                # Calculate speedup and associated error
                speedup  = t_bench / t
                err = (1/t)**2.0 * dt_bench**2.0
                err += (-t_bench/(t**2.0))**2.0 * dt**2.0
                err = math.sqrt(err)
                
                output=gpu_data
                if "CPU" in key:
                    output=cpu_data
                
                colour="Orange"       
                if highlight_key in key:
                    colour="Purple"
             
                output.ingest(speedup, err, key, colour)
            
            
            total_data = [*gpu_data.speedups, *cpu_data.speedups]
            
            for n, data in enumerate([cpu_data, gpu_data]):
                if data.num_items()>0:
                    
                    # Sort in descending order
                    sort_indices = (np.argsort(data.speedups))[::-1]
                    
                    ax[n].barh(np.array(data.labels)[sort_indices], 
                               np.array(data.speedups)[sort_indices], 
                               0.8, 
                               xerr=np.array(data.errors)[sort_indices], 
                               color=np.array(data.colours)[sort_indices])
                    ax[n].set_xlabel("Speedup, more is better")
                    ax[n].set_xlim((0,1.1*np.max(data.speedups)))
    
            fig.tight_layout()
            plt.show()
            
def plot_slices(images):
    
    # Get the dimensions
    nslices, N0, N1 = images.shape
    
    # Animate the result
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    extent=[ -0.5, N1-0.5, -0.5, N0-0.5]
    img = ax.imshow(
        images[0,...], 
        extent=extent, 
        vmin=np.min(images), 
        vmax=np.max(images),
        cmap=plt.get_cmap("Greys_r")
    )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 0")
    ax.set_title("Images")

    def update(n=0):
        img.set_data(images[n,...])
        plt.show()
    
    # Run the interaction
    result = widgets.interact(
        update,
        n=(0, nslices-1, 1)
    )
