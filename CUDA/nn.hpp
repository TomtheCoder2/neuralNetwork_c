#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#ifndef NN_NN_CPP
#define NN_NN_CPP

#define printf printf

class Matrix {
public:
    int rows;
    int cols;
    double *data;
};

const double l_rate = 0.000056;
const int train_count = 100;
const int layerCount = 4;


// allocate a matrix with given dimensions
__device__ __host__ Matrix *allocMatrix(int rows, int cols) {
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double *) malloc(rows * cols * sizeof(double));
    return m;
}

// multiply two matrices together
__device__ __host__ Matrix *matrixMult(Matrix *a, Matrix *b) {
    Matrix *c = allocMatrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            c->data[i * c->cols + j] = sum;
        }
    }
    return c;
}

// add two matrices together
__device__ __host__ Matrix *add(Matrix *a, Matrix *b) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Matrix sizes do not match\n");
        return NULL;
    }
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
        }
    }
    return c;
}


// subtract two matrices a - b
__device__ __host__ Matrix *sub(Matrix *a, Matrix *b) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] - b->data[i * b->cols + j];
        }
    }
    return c;
}

// multiply a scalar to a each matrix element
__device__ __host__ Matrix *matrixMultScalar(Matrix *a, double scalar) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] * scalar;
        }
    }
    return c;
}

// add a scalar to each matrix element
__device__ __host__ Matrix *matrixAddScalar(Matrix *a, double scalar) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] + scalar;
        }
    }
    return c;
}


// compute the sigmoid function on each element of a matrix
__device__ __host__ Matrix *matrixSigmoid(Matrix *a) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = 1 / (1 + exp(-a->data[i * a->cols + j]));
        }
    }
    return c;
}

// compute the derivative of the sigmoid function on each element of a matrix
__device__ __host__ Matrix *matrixSigmoidDerivative(Matrix *a) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] * (1 - a->data[i * a->cols + j]);
        }
    }
    return c;
}

// transpose a matrix
__device__ __host__ Matrix *matrixTranspose(Matrix *a) {
    Matrix *c = allocMatrix(a->cols, a->rows);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[j * c->cols + i] = a->data[i * a->cols + j];
        }
    }
    return c;
}

// convert an array of doubles into a matrix with given dimensions, row-major
__device__ __host__ Matrix *init_matrix_array(int n, int m, const double *A) {
    double *B = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < n * m; i++) {
        B[i] = A[i];
    }
    Matrix *mat = (Matrix *) malloc(sizeof(Matrix));
    mat->cols = m;
    mat->rows = n;
    mat->data = B;
    return mat;
}

//// convert an array of doubles into a matrix with given dimensions, row-major
//Matrix *init_matrix_from_array(int n, int m, const double A[n]) {
//    double *B = (double *) malloc(n * m * sizeof(double));
//    for (int i = 0; i < n * m; i++) {
//        B[i] = A[i];
//    }
//    Matrix *mat = malloc(sizeof(Matrix));
//    mat->cols = m;
//    mat->rows = n;
//    mat->data = B;
//    return mat;
//}

// print a matrix
__device__ __host__ void print_matrix(Matrix *mat);

// init a matrix with random values
__host__ Matrix *init_matrix_r(int n, int m, int seed) {
    double *A = (double *) malloc(n * m * sizeof(double));
    //    double A[n * m];
    srand(seed);

    for (int i = 0; i < n * m; i++) {
        A[i] = (((rand() / (double) RAND_MAX) * 2.0) - 1.0);
        rand();
        //        printf("%f ", rand() / (double) RAND_MAX * 2.0 - 1.0);
    }
    Matrix *mat = static_cast<Matrix *>(malloc(sizeof(Matrix)));
    mat->cols = m;
    mat->rows = n;
    mat->data = A;
    return mat;
}

// init a matrix with value 0
__device__ __host__ Matrix *init_matrix(int n, int m) {
    double *A = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < n * m; i++) {
        A[i] = 0;
    }
    Matrix *mat = static_cast<Matrix *>(malloc(sizeof(Matrix)));
    mat->cols = m;
    mat->rows = n;
    mat->data = A;
    return mat;
}

// free a matrix
__device__ __host__ void matrix_release(Matrix *mat) {
    if (mat == NULL) {
        return;
    }
    if (mat->data != NULL) {
        free(mat->data);
    }
    free(mat);
}

// print matrix
__device__ __host__ void print_matrix(Matrix *mat) {
    printf("Matrix: (%d x %d)\n", mat->rows, mat->cols);
    if (mat == NULL || mat->data == NULL) {
        printf("NULL\n");
        return;
    }
    if (mat->rows == 0 || mat->cols == 0) {
        printf("Empty\n");
        return;
    }
    for (int i = 0; i < mat->rows; i++) {
        printf("\t");
        for (int j = 0; j < mat->cols; j++) {
            printf("%g ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

// print a matrix with a description before it
__device__ __host__ void print_matrix_desc(Matrix *mat, char desc[]) {
    printf("%s", desc);
    print_matrix(mat);
}

// correct the error of one layer of a neural network
__device__ __host__ void
correctError(const int i, Matrix *layers[], Matrix *error, int layerCount,
             Matrix *weights[],
             Matrix *biases[]) {
    // compute the gradient for gradient descend
    Matrix *gradient = matrixSigmoidDerivative(layers[i]);
    // add the current error of this layer to the gradient
    for (int j = 0; j < layers[i]->rows; j++) {
        for (int k = 0; k < layers[i]->cols; k++) {
            gradient->data[j * gradient->cols + k] *= error->data[j * error->cols + k];
        }
    }
    //    Matrix *g_temp = matrixMult(gradient, error);
    //    matrix_release(gradient);
    //    gradient = g_temp;
    // add the learning rate for controlled learning
    Matrix *temp = matrixMultScalar(gradient, l_rate);
    matrix_release(gradient);
    gradient = temp;
    Matrix *layerTransposed = matrixTranspose(layers[i - 1]);
    // compute the delta weight
    Matrix *delta = matrixMult(gradient, layerTransposed);
    //    print_matrix_desc(delta, "delta: ");
    // apply correction
    Matrix *temp2 = add(weights[i - 1], delta);
    matrix_release(weights[i - 1]);
    weights[i - 1] = temp2;
    Matrix *temp3 = add(biases[i - 1], gradient);
    matrix_release(biases[i - 1]);
    biases[i - 1] = temp3;
    // free memory
    matrix_release(gradient);
    matrix_release(layerTransposed);
    matrix_release(delta);
    //    print_matrix_desc(weights[i - 1], "weights: ");
}

// predict the output of a neural network for a specific input
__device__ __host__ Matrix *predict(size_t x_n, double X[], int _layerCount, Matrix *weights[], Matrix *biases[]) {
    // array of matrices resembling the layers of the network and their output
    Matrix *layers[layerCount + 1];
    // init the input layer
    layers[0] = init_matrix(x_n, 1);
    for (int i = 0; i < x_n; i++) {
        layers[0]->data[i] = X[i];
        //        printf("%g ", layers[0]->data[i]);
    }
    //    printf("\n");
    // compute the output of each layer
    Matrix *temp2;
    for (int i = 1; i < layerCount; i++) {
        layers[i] = matrixMult(weights[i - 1], layers[i - 1]);
        Matrix *temp = add(layers[i], biases[i - 1]);
        matrix_release(layers[i]);
        layers[i] = temp;
        temp2 = matrixSigmoid(layers[i]);
        matrix_release(layers[i]);
        layers[i] = temp2;
        //        matrix_release(temp);
    }
    //    matrix_release(temp2);
    // free memory
    for (int i = 0; i < layerCount - 1; i++) {
        matrix_release(layers[i]);
    }
    // return output (last layer)
    return layers[layerCount - 1];
}

// one training iteration
__device__ __host__ Matrix *train(size_t x_n, const double X[], size_t y_n, const double Y[], int _layerCount, Matrix *weights[], Matrix *biases[]) {
    // predict the output of the network for the input -----------------------------------------------------------------
    // array of matrices resembling the layers of the network and their output
    Matrix *layers[layerCount + 1];
    // init the input layer
    layers[0] = init_matrix(x_n, 1);
    //    printf("Input: ");
    for (int i = 0; i < x_n; i++) {
        layers[0]->data[i] = X[i];
        //        printf("%g, ", layers[0]->data[i]);
    }
    //    printf("\n");
    //    printf("Target: ");
    for (int i = 0; i < y_n; i++) {
        //        printf("%g, ", Y[i]);
    }
    //    printf("\n");
    // compute the output of each layer
    Matrix *temp2;
    for (int i = 1; i < layerCount; i++) {
        layers[i] = matrixMult(weights[i - 1], layers[i - 1]);
        //        print_matrix_desc(matrixTranspose(weights[i - 1]), "weights[i - 1]: ");
        Matrix *temp = add(layers[i], biases[i - 1]);
        matrix_release(layers[i]);
        layers[i] = temp;
        temp2 = matrixSigmoid(layers[i]);
        matrix_release(layers[i]);
        layers[i] = temp2;
        //        matrix_release(temp);
        //        print_matrix_desc(matrixTranspose(layers[i]), "layers[i]: ");
    }
    //    print_matrix_desc(layers[layerCount - 1], "Output: ");
    // compute the error of the output layer and how to correct for it---------------------------------------------------------------
    // compare output the expected output (target) => error
    Matrix *target = init_matrix(y_n, 1);
    for (int i = 0; i < y_n; i++) {
        target->data[i] = Y[i];
        //        printf("%g ", target->data[i]);
    }
    //    printf("\n");
    Matrix *error = sub(target, layers[layerCount - 1]);
    Matrix *transposed;
    //    print_matrix_desc(error, "Error: ");
    // correct the error of each layer
    correctError(layerCount - 1, layers, error, layerCount, weights, biases);
    Matrix *temp;
    for (int i = layerCount - 2; i > 0; i--) {
        // compute the error of each layer
        transposed = matrixTranspose(weights[i]);
        temp = matrixMult(transposed, error);
        matrix_release(transposed);
        matrix_release(error);
        error = temp;
        // apply correction
        correctError(i, layers, error, layerCount, weights, biases);
        //        matrix_release(temp);
    }
    // free all memory
    //    matrix_release(transposed);
    matrix_release(target);
    matrix_release(temp2);
    matrix_release(temp);
    //    matrix_release(error);
    for (int i = 0; i < layerCount - 1; i++) {
        matrix_release(layers[i]);
    }
    // print last layer weights
    //    for (int i = 0; i < weights[layerCount - 2]->rows; i++) {
    //        for (int j = 0; j < weights[layerCount - 2]->cols; j++) {
    ////            printf("%g ", weights[layerCount - 2]->data[i * weights[layerCount - 2]->cols + j]);
    //        }
    ////        printf("\n");
    //    }
    //    printf("\n");
    // return output for later stats (WIP)
    return layers[layerCount - 1];
}


__device__ __host__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// A function to generate a random permutation of arr[]
__device__ __host__ void randomize(int arr[], int n) {
    // Use a different seed value so that we don't get same
    // result each time we run this program
    //    srand(time(NULL));

    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = n - 1; i > 0; i--) {
        // Pick a random index from 0 to i
        int j = (n + 928392 * 1 + 292329) % (i + 1); // too lazy for real randomness

        // Swap arr[i] with the element at random index
        swap(&arr[i], &arr[j]);
    }
}

#undef printf

// just call the train function for each learn_set each epoch
__device__ __host__ void fit(Matrix *train_set[], Matrix *target_set[],
         int epochs, int layerCount, Matrix *weights[], Matrix *biases[]) {
    printf("fit\n");
//    Matrix *output;
    for (int i = 0; i < epochs; i++) {
        //        shuffle(train_count, train_set);
        //        randomize(train_set, train_count);
        int arr[train_count];
        for (int j = 0; j < train_count; j++) {
            arr[j] = j;
        }
        randomize(arr, train_count);
        Matrix *train_set_temp[train_count];
        Matrix *target_set_temp[train_count];
        for (int j = 0; j < train_count; j++) {
            train_set_temp[j] = train_set[arr[j]];
            target_set_temp[j] = target_set[arr[j]];
        }
        if (i % 100 == 0) {
            printf("epoch %d\n", i);
        }
        //        printf("epoch %d\n", i);
        for (int j = 0; j < train_count; j++) {
            train(train_set_temp[j]->rows, train_set_temp[j]->data, target_set_temp[j]->rows, target_set_temp[j]->data, layerCount, weights, biases);
            //            matrix_release(output);
        }
    }
}

//// Matrix *weights[contestantCount][layerCount (index i)];
//// Matrix *biases][contestantCount][layerCount (index i);
//// training data is the same for all contestants
//// int layerSizes[contestantCount][layerCount];
//// int layerCounts[contestantCount];
//__kernel __device__ __host__ void fit(__global const double *input, __global const double *target, const int test_count,
//                  __global double *weights, __global double *biases,
//                  __global const int *layerSizes, __global const int *layerCounts, const int contestantCount, const int *epochs, __global const double *learning_rates) {
//__device__ __host__ void fitK(const double *input, const double *target, const int test_count,
//          double *weights, double *biases,
//          const int *layerSizes, const int *layerCounts, const int contestantCount, const int *epochs, const double *learning_rates) {
//    int id = get_global_id(0);
//    // get the training data for this contestant
//    Matrix *train_set[test_count];
//    Matrix *target_set[test_count];
//    for (int i = 0; i < test_count; i++) {
//        train_set[i] = init_matrix(4, 1);
//        for (int j = 0; j < 4; j++) {
//            train_set[i]->data[j] = input[i * 4 + j];
//        }
//        target_set[i] = init_matrix(7, 1);
//        for (int j = 0; j < 7; j++) {
//            target_set[i]->data[j] = target[i * 7 + j];
//        }
//    }
//    // get the layer sizes for this contestant
//    int layerCount = layerCounts[contestantCount];
//    int layerSizes_curr[layerCount];
//    // calculate where current layerSizes are in the main array
//    int layer_index = 0;
//    // go through each neural network layer
//    for (int i = 0; i < id; i++) {
//        // go through each layer size
//        for (int j = 0; j < layerCounts[i]; j++) {
//            layer_index++;
//        }
//    }
//    for (int i = 0; i < layerCount; i++) {
//        layerSizes_curr[i] = layerSizes[layer_index + i];
//    }
//    // get the weights and biases_curr for this contestant
//    Matrix *weights_curr[layerCount];
//    Matrix *biases_curr[layerCount];
//    for (int i = 0; i < layerCount; i++) {
//        // layer
//        weights_curr[i] = init_matrix(layerSizes[i], layerSizes[i - 1] + 1);
//        for (int j = 0; j < weights_curr[i]->rows * weights_curr[i]->cols; j++) {
//            // calc index where weight is located in the array
//            // go through each post neural network
//            int index = 0;
//            layer_index = 0;
//            for (int k = 0; k < id; k++) {
//                // go through each layer
//                for (int l = 0; l < layerCounts[k]; l++) {
//                    index += layerSizes[layer_index] * layerSizes[layer_index + 1];
//                    layer_index++;
//                }
//            }
//            weights_curr[i]->data[j] = weights[index + i * weights_curr[i]->rows * weights_curr[i]->cols + j];
//        }
//        biases_curr[i] = init_matrix(layerSizes[i], 1);
//        for (int j = 0; j < biases_curr[i]->rows * biases_curr[i]->cols; j++) {
//            // calc layer_index where bias is located in the array
//            // go through each post neural network
//            int index = 0;
//            layer_index = 0;
//            for (int k = 0; k < id; k++) {
//                // go through each layer
//                for (int l = 0; l < layerCounts[k]; l++) {
//                    index += layerSizes[layer_index] * layerSizes[layer_index + 1];
//                    layer_index++;
//                }
//            }
//            biases_curr[i]->data[j] = biases[index + i * biases_curr[i]->rows * biases_curr[i]->cols + j];
//        }
//    }
//    // get the learning rate for this contestant
//    l_rate = learning_rates[contestantCount];
//    // fit the model for this contestant
//    fit(test_count, train_set, target_set, *epochs, layerCount, weights_curr, biases_curr);
//}
//


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H