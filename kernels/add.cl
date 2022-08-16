__kernel void add(__global double *X, __global double *Y, __global double *result) {
    int id = get_global_id(0);
    res[id] = X[id] + Y[id];
}