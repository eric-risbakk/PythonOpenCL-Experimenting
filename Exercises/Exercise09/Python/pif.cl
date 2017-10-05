// This function calculates the integral values of some interval.
__kernel void pi_function(
    const int START, // Starting step (inclusive).
    const int END, // End step (exclusive).
    const float step, // Step size.
    __local float* localSum, // Local array with the sum of each work item's interval.
    __global float* Sum )  // The final sum value for the work group returned.
    {

    float x;
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    tmp = 0.0;
    for (int i = START; i < END; ++i) {
        x = (i+0.5)*step;
        tmp = tmp + (4.0/(1.0 + x*x));
    }
    localSum[lid] = sum;
}