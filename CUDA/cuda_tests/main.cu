#include <bits/stdc++.h>

using namespace std;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


class CudaClass {
public:
    double *data;
    int a;

    CudaClass(double x) {
        data = new double[1];
        data[0] = x;
        a = x * 2;
    }
};

__global__ void useClass(CudaClass *cudaClass) {
    printf("%g, %d\n", cudaClass->data[0], cudaClass->a);
};

CudaClass *copyToGPU(CudaClass c) {
    // create class storage on device and copy top level class
    CudaClass *d_c;
    cudaMalloc((void **) &d_c, sizeof(CudaClass));
    cudaCheckErrors("cudaMalloc");
    cudaMemcpy(d_c, &c, sizeof(CudaClass), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");

    // make an allocated region on device for use by pointer in class
    double *hostdata;
    cudaMalloc((void **) &hostdata, sizeof(double));
    cudaCheckErrors("cudaMalloc");
    printf("%g\n", c.data[0]);
    cudaMemcpy(hostdata, c.data, sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");

    // copy pointer to allocated device storage to device class
    cudaMemcpy(&(d_c->data), &hostdata, sizeof(double *), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");
    return d_c;
}

int main() {
    CudaClass c(1);

    CudaClass *d_c = copyToGPU(c);

    useClass<<<1, 1>>>(d_c);
    cudaDeviceSynchronize();
    return 0;
}