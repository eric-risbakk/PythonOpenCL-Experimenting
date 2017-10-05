#
# This program will numerically compute the integral of
#
#                4/(1+x*x)
#
# from 0 to 1.  The value of this integral is pi -- which
# is great since it gives us an easy way to check the answer.
#
# This the OpenCL version.
#
# History: Written by Eric Risbakk.

import pyopencl as cl
import numpy as np

# Here starts primary details of the program.
# TODO: Finish a program version.
num_steps = 1000000

# Beginning OpenCL setup.

# Obtain OpenCL platform.
platform = cl.get_platforms()[0]

# Obtain device id for an accelerator.
device = platform.get_devices()[0]

# Get context.
context = cl.Context([device])

# Create program from source code.
# TODO: Write program.
# Build program.
# Create kernel from program function.
kernel_source = open('pif.cl').read()
program = cl.Program(context, kernel_source).build()

# Create command queue for device.
queue = cl.CommandQueue(context)

# Allocate device memory and move input data from host to the device memory.
# TODO: Write these read/write thingies.

# TODO: The rest of the needed OpenCL setup.

# TODO: Output result, plus an estimate of the error in the result.
# TODO: Report runtime.
