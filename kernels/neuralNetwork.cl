#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


#define printf printf
double l_rate = 0.000056;

typedef struct Matrix {
    int rows;
    int cols;
    double *data;
} Matrix;


// allocate a matrix with given dimensions
Matrix *allocMatrix(int rows, int cols) {
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (double *) malloc(rows * cols * sizeof(double));
    return m;
}

// multiply two matrices together
Matrix *matrixMult(Matrix *a, Matrix *b) {
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
Matrix *add(Matrix *a, Matrix *b) {
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
Matrix *sub(Matrix *a, Matrix *b) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] - b->data[i * b->cols + j];
        }
    }
    return c;
}

// multiply a scalar to a each matrix element
Matrix *matrixMultScalar(Matrix *a, double scalar) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] * scalar;
        }
    }
    return c;
}

// add a scalar to each matrix element
Matrix *matrixAddScalar(Matrix *a, double scalar) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] + scalar;
        }
    }
    return c;
}


// compute the sigmoid function on each element of a matrix
Matrix *matrixSigmoid(Matrix *a) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = 1 / (1 + exp(-a->data[i * a->cols + j]));
        }
    }
    return c;
}

// compute the derivative of the sigmoid function on each element of a matrix
Matrix *matrixSigmoidDerivative(Matrix *a) {
    Matrix *c = allocMatrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[i * c->cols + j] = a->data[i * a->cols + j] * (1 - a->data[i * a->cols + j]);
        }
    }
    return c;
}

// transpose a matrix
Matrix *matrixTranspose(Matrix *a) {
    Matrix *c = allocMatrix(a->cols, a->rows);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            c->data[j * c->cols + i] = a->data[i * a->cols + j];
        }
    }
    return c;
}

// convert an array of doubles into a matrix with given dimensions, row-major
Matrix *init_matrix_array(int n, int m, const double *A) {
    double *B = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < n * m; i++) {
        B[i] = A[i];
    }
    Matrix *mat = malloc(sizeof(Matrix));
    mat->cols = m;
    mat->rows = n;
    mat->data = B;
    return mat;
}

// convert an array of doubles into a matrix with given dimensions, row-major
Matrix *init_matrix_from_array(int n, int m, const double A[n]) {
    double *B = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < n * m; i++) {
        B[i] = A[i];
    }
    Matrix *mat = malloc(sizeof(Matrix));
    mat->cols = m;
    mat->rows = n;
    mat->data = B;
    return mat;
}

// print a matrix
void print_matrix(Matrix *mat);

// init a matrix with random values
Matrix *init_matrix_r(int n, int m, int seed) {
    double *A = (double *) malloc(n * m * sizeof(double));
    //    double A[n * m];
    srand(seed);

    for (int i = 0; i < n * m; i++) {
        A[i] = (((rand() / (double) RAND_MAX) * 2.0) - 1.0);
        rand();
        //        printf("%f ", rand() / (double) RAND_MAX * 2.0 - 1.0);
    }
    Matrix *mat = malloc(sizeof(Matrix));
    mat->cols = m;
    mat->rows = n;
    mat->data = A;
    return mat;
}

// init a matrix with value 0
Matrix *init_matrix(int n, int m) {
    double *A = (double *) malloc(n * m * sizeof(double));
    for (int i = 0; i < n * m; i++) {
        A[i] = 0;
    }
    Matrix *mat = malloc(sizeof(Matrix));
    mat->cols = m;
    mat->rows = n;
    mat->data = A;
    return mat;
}

// free a matrix
void matrix_release(Matrix *mat) {
    if (mat == NULL) {
        return;
    }
    if (mat->data != NULL) {
        free(mat->data);
    }
    free(mat);
}

// print matrix
void print_matrix(Matrix *mat) {
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
void print_matrix_desc(Matrix *mat, char desc[]) {
    printf("%s", desc);
    print_matrix(mat);
}

// correct the error of one layer of a neural network
void
correctError(const int i, Matrix *layers[], Matrix *error, int layerCount,
             Matrix *weights[layerCount],
             Matrix *biases[layerCount]) {
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
Matrix *predict(size_t x_n, double X[], int layerCount, Matrix *weights[layerCount], Matrix *biases[layerCount]) {
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
Matrix *train(size_t x_n, const double X[], size_t y_n, const double Y[], int layerCount, Matrix *weights[layerCount], Matrix *biases[layerCount]) {
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


void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// A function to generate a random permutation of arr[]
void randomize(int arr[], int n) {
    // Use a different seed value so that we don't get same
    // result each time we run this program
    //    srand(time(NULL));

    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = n - 1; i > 0; i--) {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);

        // Swap arr[i] with the element at random index
        swap(&arr[i], &arr[j]);
    }
}

#undef printf

// just call the train function for each learn_set each epoch
void fit(size_t train_count, Matrix *train_set[train_count], Matrix *target_set[train_count], int epochs, int layerCount, Matrix *weights[layerCount], Matrix *biases[layerCount]) {
    printf("fit\n");
    Matrix *output;
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
            output = train(train_set_temp[j]->rows, train_set_temp[j]->data, target_set_temp[j]->rows, target_set_temp[j]->data, layerCount, weights, biases);
            //            matrix_release(output);
        }
    }
}

enum {
    test_count = 100
};

// resembles a train_set but only contains one input and one output
typedef struct {
    Matrix *input[test_count * 7];
    Matrix *target[test_count * 7];
} TrainSet;

double max_train_set = 0;

TrainSet *getTrainingData();


// Matrix *weights[contestantCount][layerCount (index i)];
// Matrix *biases][contestantCount][layerCount (index i);
// training data is the same for all contestants
// int layerSizes[contestantCount][layerCount];
// int layerCounts[contestantCount];
void fit(__global const double *input, __global const double *target, const int test_count,
                  __global double *weights, __global double *biases,
                  __global const int *layerSizes, __global const int *layerCounts, const int contestantCount, const int *epochs, __global const double *learning_rates) {
    int id = get_global_id(0);
    // get the training data for this contestant
    Matrix *train_set[test_count];
    Matrix *target_set[test_count];
    for (int i = 0; i < test_count; i++) {
        train_set[i] = getTrainingData()->input[i];
        target_set[i] = getTrainingData()->target[i];
    }
    // get the layer sizes for this contestant
    int layerCount = layerCounts[contestantCount];
    int layerSizes[layerCount];
    for (int i = 0; i < layerCount; i++) {
        layerSizes[i] = layerSizes[contestantCount * layerCount + i];
    }
    // get the weights and biases for this contestant
    Matrix *weights[layerCount];
    Matrix *biases[layerCount];
    for (int i = 0; i < layerCount; i++) {
        weights[i] = init_matrix(layerSizes[i], layerSizes[i - 1] + 1);
        for (int j = 0; j < weights[i]->rows * weights[i]->cols; j++) {
            weights[i]->data[j] = weights[contestantCount * layerCount + i]->data[j];
        }
        biases[i] = init_matrix(layerSizes[i], 1);
        for (int j = 0; j < biases[i]->rows * biases[i]->cols; j++) {
            biases[i]->data[j] = biases[contestantCount * layerCount + i]->data[j];
        }
    }
    // get the learning rate for this contestant
    l_rate = learning_rates[contestantCount];
    // fit the model for this contestant
    fit(test_count, train_set, target_set, *epochs, layerCount, weights, biases);
}


int main() {
    srand(12345);
    // check for mem leak: cmake .; make; valgrind --tool=memcheck --leak-check=full ./train_p
    // how many layers of the network
    int layerCount = 4;
    // testing arrays
    double X[] = {17, 35, 27, 81};
    double Y[] = {1, 0, 0, 0, 0, 0, 0};
    // amount of nodes of each layer
    int layerSizes[] = {4, 74, 89, 7};
    //    int layerSizes[] = {4, 2, 3, 7};
    // weight and biases for the network
    Matrix *weights[layerCount - 1];
    Matrix *biases[layerCount - 1];
    // local train_set and target_set
    Matrix *train_set[test_count * 7];
    Matrix *target_set[test_count * 7];
    // init train_set and target_set
    TrainSet *ts = getTrainingData();
    for (int i = 0; i < test_count * 7; i++) {
        train_set[i] = ts->input[i];
        target_set[i] = ts->target[i];
        //        print_matrix_desc(train_set[i], "train_set");
    }
    printf("training data init done\n");
    // init weights and biases
    for (int i = 1; i < layerCount; i++) {
        printf("init i %d: ", i);
        printf("%d\n", layerSizes[i]);
        //        weights[i - 1] = init_matrix_r(layerSizes[i], layerSizes[i - 1], 12345);
        //        biases[i - 1] = init_matrix_r(layerSizes[i], 1, 12345);
    }
    printf("weights and biases init done\n");
    // train
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    fit(test_count * 7, train_set, target_set, 1000, layerCount, weights, biases);
    gettimeofday(&stop, NULL);
    printf("training took %ld us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    // print in ms
    printf("training took %ld ms\n", (stop.tv_sec - start.tv_sec) * 1000 + stop.tv_usec - start.tv_usec / 1000);
    // print in s
    printf("training took %ld s\n", (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000);
    // one train for debugging
    for (int i = 0; i < 20; i++) {
        (train(train_set[i]->rows, train_set[i]->data, target_set[i]->rows, target_set[i]->data, layerCount, weights, biases));
        //        (train(train_set[1]->rows, train_set[1]->data, target_set[1]->rows, target_set[1]->data, layerCount, weights, biases));
    }
    // print the first 20 training sets
    for (int i = 0; i < 20; i++) {
        printf("NN.train(new double[]{");
        for (int j = 0; j < train_set[i]->rows; j++) {
            printf("%g, ", train_set[i]->data[j]);
        }
        printf("}, new double[]{");
        for (int j = 0; j < target_set[i]->rows; j++) {
            printf("%g, ", target_set[i]->data[j]);
        }
        printf("});\n");
    }
    // testing the network with the test set, ik i should split it in train and test set, but this is just a test anyway
    ts = getTrainingData();
    for (int i = 0; i < test_count * 7; i++) {
        train_set[i] = ts->input[i];
        target_set[i] = ts->target[i];
        //        print_matrix_desc(train_set[i], "train_set");
    }
    for (int i = 0; i < 7 * test_count; i += test_count) {
        Matrix *result = predict(4, train_set[i]->data, layerCount, weights, biases);
        for (int k = 0; k < result->rows; k++) {
            printf("%g ", result->data[k]);
        }
        printf("\n");
        // get the index of the highest value in the result matrix
        int index = 0;
        for (int j = 0; j < 7; j++) {
            if (result->data[j] > result->data[index]) {
                index = j;
            }
        }
        //        printf("%d\n", index);
        double res[7];
        for (int j = 0; j < 7; j++) {
            res[j] = 0;
        }
        res[index] = 1.0;
        // check if it's correct
        int correct = 0;
        int bitarray[7];
        for (int j = 0; j < 7; j++) {
            bitarray[j] = 0;
        }
        for (int j = 0; j < 7; j++) {
            if (res[j] == target_set[i]->data[j]) {
                correct++;
            }
            if (result->data[j] > 0.5) {
                bitarray[j] = 1;
            } else {
                bitarray[j] = 0;
            }
        }
        if (correct == 7) {
            printf("correct\n");
        } else {
            printf("wrong expected ");
            for (int j = 0; j < 7; j++) {
                printf("%g ", target_set[i]->data[j]);
            }
            printf("got ");
            for (int j = 0; j < 7; j++) {
                printf("%d ", bitarray[j]);
            }
            printf("\n");
        }
        matrix_release(result);
    }
    // print neural network
    //    for (int i = 0; i < layerCount - 1; i++) {
    //        printf("layer %d\n", i);
    //        print_matrix_desc(weights[i], "weights");
    //        print_matrix_desc(biases[i], "biases");
    //        printf("\n");
    //    }
#define printf //printf
printf("weights_list = vec![");
    for (int i = 0; i < layerCount - 1; i++) {
        printf("vec![");
        for (int j = 0; j < weights[i]->rows; j++) {
            printf("vec![");
            for (int k = 0; k < weights[i]->cols; k++) {
                printf("%g, ", weights[i]->data[j * weights[i]->cols + k]);
            }
            printf("],");
        }
        printf("],");
    }
    printf("],\n");
    printf("biases_list = vec![");
    for (int i = 0; i < layerCount - 1; i++) {
        printf("vec![");
        for (int j = 0; j < biases[i]->rows; j++) {
            printf("vec![");
            for (int k = 0; k < biases[i]->cols; k++) {
                printf("%g, ", biases[i]->data[j * biases[i]->cols + k]);
            }
            printf("],");
        }
        printf("],");
    }
    printf("]\n");
    printf("for i in 0 .. biases_list.len() {\n"
           "    weights.push(Matrix::from_2d_array(weights_list[i].clone()));\n"
           "    biases.push(Matrix::from_2d_array(biases_list[i].clone()));\n"
           "}\n"
           "NN.weights = weights;\n"
           "NN.biases = biases;\n"
           "weights = vec![];\n"
           "biases = vec![];\n");
    printf("\n\n");

    // now for java
    printf("public static NeuralNetwork NN = new NeuralNetwork();\n");
    printf("NN.weights = new ArrayList<>();\n");
    printf("NN.biases = new ArrayList<>();\n");
    for (int i = 0; i < layerCount - 1; i++) {
        printf("// layer %d\n", i);
        printf("NN.weights.add(Matrix.from_2d_array(");
        printf("new double[][]{");
        for (int j = 0; j < weights[i]->rows; j++) {
            printf("new double[]{");
            for (int k = 0; k < weights[i]->cols; k++) {
                printf("%g, ", weights[i]->data[j * weights[i]->cols + k]);
            }
            printf("},");
        }
        printf("}));\n");
        printf("NN.biases.add(Matrix.from_2d_array(");
        printf("new double[][]{");
        for (int j = 0; j < biases[i]->rows; j++) {
            printf("new double[]{");
            for (int k = 0; k < biases[i]->cols; k++) {
                printf("%g, ", biases[i]->data[j * biases[i]->cols + k]);
            }
            printf("},");
        }
        printf("}));\n");
    }
    printf("\n");
#undef printf
    // release memory
    for (int i = 0; i < layerCount - 1; i++) {
        matrix_release(weights[i]);
        matrix_release(biases[i]);
    }
    printf("finished\n");
    return 0;
}