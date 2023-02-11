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

const int N = 1023;

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

__global__ void useClass(CudaClass **cudaClass) {
    int i = threadIdx.x;
    printf("kernel: %d\n", i);
//        printf("%d\n", i);
//        printf("%g, %d\n", cudaClass[i]->data[0], cudaClass[i]->a);
    float x = cudaClass[i]->data[0];
    for (int j = 0; j < 2000000000; j++) {
        x = x + 2304 + i / 2000;
    }
    cudaClass[i]->data[0] = x;
    printf("kernel: %d, result: %g\n", i, cudaClass[i]->data[0]);
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
    cudaMemcpy(hostdata, c.data, sizeof(double), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");

    // copy pointer to allocated device storage to device class
    cudaMemcpy(&(d_c->data), &hostdata, sizeof(double *), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy");
    return d_c;
}

int main() {
    CudaClass *h_classes[N];

    for (int i = 0; i < N; i++) {
        CudaClass c(i * rand());
        h_classes[i] = copyToGPU(c);
    }

    CudaClass **d_classes;
    cudaMalloc(&d_classes, N * sizeof(CudaClass *));
    cudaMemcpy(d_classes, h_classes, N * sizeof(CudaClass *), cudaMemcpyHostToDevice);

    // start kernel
    useClass<<<1, N>>>(d_classes);
    cudaDeviceSynchronize();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));


//    float **h_array_list, **d_array_list;
//    // allocate array lists
//    h_array_list = (float *) malloc(num_arrays * sizeof(float *));
//    cudaMalloc((void **) &d_array_list, num_arrays * sizeof(float *));
//    // allocate arrays on the device
//    for (int i = 0; i < num_arrays; i++)
//        cudaMalloc((void **) &h_array_list[i], data_size);
//    // copy array list to the device
//    cudaMemcpy(d_array_list, h_array_list, num_arrays * sizeof(float *), cudaMemcpyHostToDevice);
//    // allocate array list on the host
//    float **array_list;
//    array_list = (float **) malloc(num_arrays * sizeof(float *));
//// allocate arrays on the host
//    for (int i = 0; i < num_arrays; i++)
//        array_list[i] = malloc(data_size);
//// ****fill out data here
//// populate data arrays on the device
//    for (int i = 0; i < num_arrays; i++)
//        cudaMemcpy(h_array_list[i], array_list[i], data_size, cudaMemcpyDeviceToHost);


    return 0;
}