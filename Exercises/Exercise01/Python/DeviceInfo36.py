#
# Display Device Information
#
# Script to print out some information about the OpenCL devices
# and platforms available on your system
#
# History: C++ version written by Tom Deakin, 2012
#          Ported to Python by Tom Deakin, July 2013
#

# Import the Python OpenCL API
import pyopencl as cl

# Create a list of all the platform IDs
platforms = cl.get_platforms()

print("\nNumber of OpenCL platforms: ", len(platforms))

print("\n---------------------------")

# Investigate each platform.
for p in platforms:
    # Print out some information about the platforms.
    print("Platform: ", p.name)
    print("Vendor: ", p.version)
    print("Version: ", p.version)

    # Discover all devices.
    devices = p.get_devices()
    print("Number of services", len(devices))

    # Investigate each device:
    for d in devices:
        print("\t---------------------------")
        # Print some information about the devices.
        print("\t\tName: ", d.name)
        print("\t\tVersion: ", d.version)
        print("\t\tMaximum compute units: ", d.max_compute_units)
        print("\t\tLocal memory size: ", d.local_mem_size/1024, "KB")
        print("\t\tGlobal memory size: ", d.global_mem_size/(1024**2), "MB")
        print("\t\tMax alloc size: ", d.max_mem_alloc_size/(1024**2), "MB")
        print("\t\tMax work group total size: ", d.max_work_group_size)

        # Find the maximum dimensions of the work group.
        dim = d.max_work_item_sizes
        print("\t\tMax work group dims: ", dim[0], " ".join(map(str, dim[1:])), ")")

        print("\t---------------------------")

    print("\n---------------------------")
