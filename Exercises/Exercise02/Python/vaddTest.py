
import pyopencl as cl
import numpy


TOL = 0.001  # Tolerance in floating point comparisons.
COUNT = 5000  # Length of vectors.

vector_a = numpy.random.rand(COUNT).astype(numpy.float32)
vector_b = numpy.random.rand(COUNT).astype(numpy.float32)

print("Vector a: ", vector_a)
print("Vector b: ", vector_b)

# Step #1. Obtain an OpenCL platform.
platform = cl.get_platforms()[0]

# It would be necessary to add some code to check the check the support for
# the necessary platform extensions with platform.extensions

# Step #2. Obtain a device id for at least one device (accelerator).
device = platform.get_devices()[0]

# It would be necessary to add some code to check the check the support for
# the necessary device extensions with device.extensions

# Step #3. Create a context for the selected device.
context = cl.Context([device])

# Step #4. Create the accelerator program from source code.
# Step #5. Build the program.
# Step #6. Create one or more kernels from the program functions.
program = cl.Program(context, """
        __kernel void vaddTest(
        __global const float *a,
        __global const float *b, 
        __global float *c)
        {
          int gid = get_global_id(0);
          c[gid] = a[gid] + b[gid];
        }
        """).build()

# Step #7. Create a command queue for the target device.
queue = cl.CommandQueue(context)

# Step #8. Allocate device memory and move input data from the host to the device memory.
mem_flags = cl.mem_flags
matrix_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector_b)
vector_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector_a)
end_vector = numpy.empty_like(vector_a)
destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, vector_a.nbytes)

# Step #9. Associate the arguments to the kernel with kernel object.
# Step #10. Deploy the kernel for device execution.
program.vaddTest(queue, end_vector.shape, None, matrix_buf, vector_buf, destination_buf)

# Step #11. Move the kernel’s output data to host memory.
cl.enqueue_copy(queue, end_vector, destination_buf)

# Step #12. Release context, program, kernels and memory.
# PyOpenCL performs this step for you, and therefore,
# you don't need to worry about cleanup code

print("End vector: ", end_vector)

print("Let us check if calculations are matching!")

correct = 0

for i in range(COUNT):
    tmp = 0
    tmp = vector_a[i] + vector_b[i]
    tmp -= end_vector[i]
    if tmp**2 < TOL**2:
        correct += 1
    else:
        print("{} (Tmp) {} (vector a), {} (vector b), {} (end vector)".format(tmp, vector_a[i], vector_b[i], end_vector[i]))


print("Numbers correct out of total: ({} / {})".format(correct, COUNT))



