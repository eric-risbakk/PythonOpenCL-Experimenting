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



