#
# Vadd
#
# Element wise addition of two vectors (c = a + b)
# Asks the user to select a device at runtime
#
# History: C version written by Tim Mattson, December 2009
#          C version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#

# Import the Python OpenCL API
import pyopencl as cl
# Import the Python Maths Library (for vectors)
import numpy

# Import a library to print out the device information
import deviceinfo

# Import Standard Library to time the execution
from time import time
#------------------------------------------------------------------------------

# tolerance used in floating point comparisons
TOL = 0.001
# length of vectors a, b and c
LENGTH = 1024

#------------------------------------------------------------------------------
#
# Kernel: vadd
#
# To compute the elementwise sum c = a + b
#
# Input: a and b float vectors of length count
# Output c float vector of length count holding the sum a + b

kernel_source = """
__kernel void vadd_b(
    __global float* n0,
    __global float* n1,
    __global float* n2,
    __global float* n3,
    __global float* out0,
    const unsigned int count)
    {
    
        float t0, t1;
    int i = get_global_id(0);
    if (i < count) 
    {
        t0 = n0[i] + n1[i];
        t1 = t0 + n2[i];
        out0[i] = t1 + n3[i];
    }
}
"""

# ------------------------------------------------------------------------------

# Main procedure

# Create a compute context
# Ask the user to select a platform/device on the CLI
context = cl.create_some_context()

# Print out device info
deviceinfo.output_device_info(context.devices[0])

# Create a command queue
queue = cl.CommandQueue(context)

print("Starting to build program.")

# Create the compute program from the source buffer
# and build it
program = cl.Program(context, kernel_source).build()

print("Program built!")

# Create a and b vectors and fill with random float values
h_n0 = numpy.random.rand(LENGTH).astype(numpy.float32)
h_n1 = numpy.random.rand(LENGTH).astype(numpy.float32)
h_n2 = numpy.random.rand(LENGTH).astype(numpy.float32)
h_n3 = numpy.random.rand(LENGTH).astype(numpy.float32)
# Create an empty c vector (a+b) to be returned from the compute device
h_out0 = numpy.empty(LENGTH).astype(numpy.float32)

# Create the input (a, b) arrays in device memory and copy data from host
d_n0 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_n0)
d_n1 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_n1)
d_n2 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_n2)
d_n3 = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_n3)
# Create the output (c) array in device memory
d_out0 = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_out0.nbytes)

# Start the timer
r_time = time()

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
vadd = program.vadd_b
vadd.set_scalar_arg_dtypes([None, None, None, None, None, numpy.uint32])
vadd(queue, h_n0.shape, None, d_n0, d_n1, d_n2, d_n3, d_out0, LENGTH)

# Wait for the commands to finish before reading back
queue.finish()
r_time = time() - r_time
print("The kernel ran in", r_time, "seconds")

# Read back the results from the compute device
cl.enqueue_copy(queue, h_out0, d_out0)

# Test the results
correct1 = 0
for a, b, c, d, e in zip(h_n0, h_n1, h_n2, h_n3, h_out0):
    # assign element i of a+b to tmp
    tmp = a + b + c + d
    # compute the deviation of expected and output result
    tmp -= e
    # correct if square deviation is less than tolerance squared
    if tmp*tmp < TOL*TOL:
        correct1 += 1
    else:
        print("tmp", tmp, "h_a", a, "h_b", b, "h_c", c)

# Summarize results
print("E = A+B+C+D :", correct1, "out of", LENGTH, "results were correct.")
