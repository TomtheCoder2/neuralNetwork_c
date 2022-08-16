#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


#define printf printf
const double l_rate = 0.000056;
const double lamda = 100;

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

// print a matrix
void print_matrix(Matrix *mat);

// init a matrix with random values
Matrix *init_matrix(int n, int m) {
    double *A = (double *) malloc(n * m * sizeof(double));
//    double A[n * m];
    for (int i = 0; i < n * m; i++) {
        A[i] = rand() / (double) RAND_MAX;
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
}

// predict the output of a neural network for a specific input
Matrix *predict(size_t x_n, double X[], int layerCount, Matrix *weights[layerCount], Matrix *biases[layerCount]) {
    // array of matrices resembling the layers of the network and their output
    Matrix *layers[layerCount + 1];
    // init the input layer
    layers[0] = init_matrix(x_n, 1);
    for (int i = 0; i < x_n; i++) {
        layers[0]->data[i] = X[i];
        printf("%g ", layers[0]->data[i]);
    }
    printf("\n");
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
    for (int i = 0; i < x_n; i++) {
        layers[0]->data[i] =
                X[i];
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
    // return output for later stats (WIP)
    return layers[layerCount - 1];
}


/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(size_t n, Matrix *array[n]) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            Matrix *t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
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
    srand(time(NULL));

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
        for (int j = 0; j < train_count; j++) {
            train_set[j] = train_set[arr[j]];
            target_set[j] = target_set[arr[j]];
        }
        if (i % 100 == 0) {
            printf("epoch %d\n", i);
        }
//        printf("epoch %d\n", i);
        for (int j = 0; j < train_count; j++) {
            output = train(train_set[j]->rows, train_set[j]->data, target_set[j]->rows, target_set[j]->data, layerCount, weights, biases);
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


int main() {
    // check for mem leak: cmake .; make; valgrind --tool=memcheck --leak-check=full ./train_p
    // how many layers of the network
    int layerCount = 4;
    // testing arrays
    double X[] = {17, 35, 27, 81};
    double Y[] = {1, 0, 0, 0, 0, 0, 0};
    // amount of nodes of each layer
//    int layerSizes[] = {4, 74, 89, 7};
    int layerSizes[] = {4, 10, 20, 7};
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
    layerCount = 4;
    for (int i = 1; i < layerCount; i++) {
        printf("init i %d: ", i);
        printf("%d\n", layerSizes[i]);
        weights[i - 1] = init_matrix(layerSizes[i], layerSizes[i - 1]);
        biases[i - 1] = init_matrix(layerSizes[i], 1);
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
//    matrix_release(train(train_set[0]->rows, train_set[0]->data, target_set[0]->rows, target_set[0]->data, layerCount, weights, biases));
    // testing the network with the test set, ik i should split it in train and test set, but this is just a test anyway
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
        res[index] = 1.0;
        // check if it's correct
        int correct = 0;
        int bitarray[7];
        for (int j = 0; j < 7; j++) {
            if (res[j] == target_set[i]->data[j]) {
                correct++;
            }
            if (result->data[j] > 0) {
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
    for (int i = 0; i < layerCount - 1; i++) {
        printf("layer %d\n", i);
        print_matrix_desc(weights[i], "weights");
        print_matrix_desc(biases[i], "biases");
        printf("\n");
    }
    // release memory
    for (int i = 0; i < layerCount - 1; i++) {
        matrix_release(weights[i]);
        matrix_release(biases[i]);
    }
    printf("finished\n");
    return 0;
}

// data
TrainSet *getTrainingData() {
    double training_set[][4] = {{17,  35,   27,  81},
                                {16,  34,   27,  80},
                                {17,  34,   27,  81},
                                {17,  36,   28,  83},
                                {16,  35,   27,  81},
                                {17,  34,   27,  82},
                                {17,  35,   27,  82},
                                {17,  35,   27,  84},
                                {19,  39,   30,  84},
                                {16,  35,   39,  124},
                                {17,  35,   27,  81},
                                {16,  34,   26,  80},
                                {14,  32,   24,  78},
                                {19,  38,   28,  88},
                                {29,  49,   35,  115},
                                {16,  33,   26,  80},
                                {16,  33,   25,  80},
                                {91,  162,  73,  213},
                                {67,  126,  83,  237},
                                {17,  35,   26,  83},
                                {15,  34,   26,  80},
                                {16,  34,   26,  81},
                                {16,  34,   26,  80},
                                {18,  35,   27,  82},
                                {14,  33,   27,  78},
                                {16,  34,   26,  81},
                                {17,  34,   27,  82},
                                {18,  37,   28,  84},
                                {15,  33,   25,  78},
                                {18,  36,   28,  84},
                                {16,  34,   26,  81},
                                {17,  35,   27,  81},
                                {16,  34,   27,  81},
                                {72,  132,  60,  187},
                                {17,  35,   27,  81},
                                {18,  36,   28,  84},
                                {20,  41,   30,  91},
                                {45,  78,   43,  129},
                                {76,  134,  99,  292},
                                {17,  34,   25,  80},
                                {17,  36,   27,  85},
                                {156, 287,  118, 339},
                                {16,  34,   26,  82},
                                {17,  35,   26,  83},
                                {17,  35,   26,  82},
                                {17,  35,   27,  83},
                                {18,  36,   28,  85},
                                {17,  35,   27,  82},
                                {245, 394,  152, 446},
                                {16,  34,   27,  79},
                                {17,  35,   29,  86},
                                {48,  87,   53,  191},
                                {13,  32,   27,  80},
                                {68,  118,  51,  149},
                                {18,  36,   28,  84},
                                {16,  33,   26,  79},
                                {16,  34,   27,  79},
                                {16,  35,   29,  90},
                                {15,  33,   25,  81},
                                {18,  34,   27,  81},
                                {17,  35,   26,  81},
                                {18,  35,   26,  83},
                                {20,  34,   26,  81},
                                {14,  31,   25,  78},
                                {18,  35,   27,  84},
                                {16,  35,   27,  83},
                                {16,  34,   26,  80},
                                {17,  37,   28,  86},
                                {17,  36,   27,  86},
                                {18,  36,   28,  84},
                                {16,  34,   26,  80},
                                {62,  105,  66,  191},
                                {17,  37,   27,  84},
                                {34,  63,   39,  115},
                                {23,  44,   33,  101},
                                {17,  35,   27,  82},
                                {103, 168,  88,  272},
                                {17,  36,   27,  82},
                                {17,  34,   28,  81},
                                {17,  34,   27,  80},
                                {18,  37,   29,  87},
                                {16,  33,   26,  80},
                                {20,  40,   29,  93},
                                {17,  35,   26,  80},
                                {18,  35,   27,  83},
                                {17,  37,   28,  86},
                                {16,  35,   27,  81},
                                {78,  129,  58,  170},
                                {16,  36,   27,  84},
                                {15,  33,   25,  79},
                                {54,  90,   53,  172},
                                {18,  36,   27,  84},
                                {22,  38,   28,  88},
                                {17,  36,   27,  85},
                                {17,  36,   28,  86},
                                {17,  34,   26,  81},
                                {16,  32,   25,  204},
                                {16,  33,   27,  81},
                                {17,  35,   29,  85},
                                {35,  59,   27,  81},
                                {189, 298,  180, 588},
                                {443, 706,  349, 1128},
                                {71,  108,  70,  226},
                                {506, 800,  390, 1279},
                                {35,  61,   43,  136},
                                {542, 859,  419, 1377},
                                {38,  64,   44,  186},
                                {61,  98,   65,  209},
                                {67,  108,  72,  47},
                                {299, 477,  294, 908},
                                {474, 753,  373, 1047},
                                {78,  120,  81,  277},
                                {415, 657,  334, 1095},
                                {190, 301,  174, 563},
                                {272, 724,  310, 948},
                                {199, 327,  192, 612},
                                {221, 312,  182, 586},
                                {131, 206,  128, 417},
                                {121, 174,  110, 361},
                                {340, 407,  227, 727},
                                {22,  41,   32,  95},
                                {35,  60,   42,  135},
                                {59,  93,   60,  195},
                                {436, 690,  353, 1143},
                                {436, 690,  353, 1144},
                                {132, 211,  132, 425},
                                {206, 317,  194, 629},
                                {113, 176,  117, 385},
                                {435, 692,  362, 1169},
                                {73,  117,  76,  250},
                                {56,  90,   61,  200},
                                {171, 273,  152, 484},
                                {396, 635,  314, 994},
                                {140, 226,  126, 394},
                                {159, 251,  150, 488},
                                {502, 798,  390, 1280},
                                {136, 208,  140, 468},
                                {550, 873,  428, 1409},
                                {106, 174,  106, 345},
                                {539, 874,  464, 1487},
                                {275, 434,  287, 911},
                                {459, 726,  374, 1216},
                                {30,  53,   39,  123},
                                {173, 269,  196, 650},
                                {131, 207,  134, 432},
                                {66,  106,  69,  228},
                                {138, 223,  126, 410},
                                {287, 461,  254, 805},
                                {265, 428,  233, 737},
                                {168, 265,  171, 560},
                                {93,  144,  93,  379},
                                {132, 206,  142, 381},
                                {47,  76,   51,  140},
                                {119, 186,  126, 493},
                                {518, 815,  410, 1622},
                                {36,  60,   47,  151},
                                {321, 512,  272, 867},
                                {292, 466,  251, 801},
                                {169, 268,  154, 495},
                                {43,  74,   405, 1321},
                                {270, 468,  261, 852},
                                {425, 664,  358, 1183},
                                {637, 940,  473, 1573},
                                {734, 937,  472, 1572},
                                {437, 693,  340, 1107},
                                {464, 741,  355, 1148},
                                {370, 573,  307, 1010},
                                {73,  117,  80,  255},
                                {109, 173,  115, 373},
                                {88,  146,  89,  288},
                                {108, 171,  109, 353},
                                {161, 259,  143, 447},
                                {57,  93,   64,  205},
                                {21,  41,   32,  96},
                                {449, 718,  373, 1200},
                                {67,  106,  66,  219},
                                {39,  66,   45,  148},
                                {574, 915,  458, 1528},
                                {56,  92,   58,  198},
                                {115, 179,  118, 401},
                                {137, 221,  136, 437},
                                {475, 754,  380, 1241},
                                {276, 432,  241, 783},
                                {183, 288,  168, 543},
                                {64,  101,  64,  211},
                                {68,  110,  75,  248},
                                {76,  123,  85,  282},
                                {99,  154,  98,  321},
                                {82,  127,  79,  260},
                                {242, 382,  213, 682},
                                {203, 315,  187, 611},
                                {768, 1249, 554, 1730},
                                {61,  97,   62,  203},
                                {490, 767,  389, 1278},
                                {229, 361,  193, 618},
                                {226, 356,  192, 618},
                                {45,  73,   53,  158},
                                {26,  45,   34,  92},
                                {165, 266,  159, 398},
                                {450, 713,  362, 1178},
                                {34,  126,  153, 276},
                                {22,  63,   77,  161},
                                {17,  40,   38,  101},
                                {23,  73,   93,  186},
                                {23,  67,   84,  172},
                                {28,  72,   87,  185},
                                {37,  140,  188, 333},
                                {28,  96,   124, 231},
                                {61,  115,  98,  132},
                                {20,  47,   51,  135},
                                {31,  82,   96,  172},
                                {23,  66,   80,  159},
                                {20,  49,   35,  92},
                                {17,  37,   31,  89},
                                {19,  47,   57,  129},
                                {25,  72,   89,  179},
                                {17,  40,   34,  92},
                                {18,  41,   39,  101},
                                {17,  39,   35,  93},
                                {19,  41,   37,  98},
                                {19,  43,   39,  103},
                                {20,  39,   36,  95},
                                {148, 326,  295, 616},
                                {24,  71,   88,  177},
                                {21,  62,   75,  156},
                                {18,  38,   33,  92},
                                {22,  69,   84,  172},
                                {20,  50,   52,  126},
                                {18,  38,   32,  91},
                                {62,  191,  219, 424},
                                {56,  168,  199, 380},
                                {54,  171,  204, 383},
                                {18,  37,   31,  89},
                                {25,  70,   77,  159},
                                {45,  111,  122, 248},
                                {20,  50,   55,  127},
                                {18,  39,   33,  92},
                                {29,  96,   110, 208},
                                {23,  73,   98,  192},
                                {21,  66,   82,  163},
                                {22,  61,   75,  156},
                                {39,  107,  126, 255},
                                {72,  180,  191, 379},
                                {36,  134,  169, 302},
                                {27,  71,   83,  174},
                                {37,  141,  181, 322},
                                {28,  87,   111, 215},
                                {73,  210,  229, 474},
                                {36,  64,   50,  148},
                                {13,  33,   29,  78},
                                {23,  67,   84,  171},
                                {73,  205,  223, 463},
                                {19,  46,   47,  130},
                                {19,  39,   35,  95},
                                {20,  58,   72,  140},
                                {28,  97,   123, 209},
                                {37,  139,  176, 313},
                                {41,  138,  159, 300},
                                {20,  53,   80,  165},
                                {25,  76,   84,  168},
                                {18,  40,   34,  92},
                                {17,  39,   34,  92},
                                {21,  48,   54,  124},
                                {19,  53,   63,  138},
                                {28,  76,   98,  190},
                                {21,  59,   72,  152},
                                {71,  166,  183, 394},
                                {22,  66,   71,  150},
                                {20,  44,   41,  106},
                                {21,  57,   72,  152},
                                {92,  239,  240, 478},
                                {21,  42,   37,  103},
                                {67,  202,  245, 484},
                                {136, 294,  283, 641},
                                {53,  119,  127, 280},
                                {61,  193,  234, 441},
                                {18,  43,   42,  107},
                                {17,  42,   44,  110},
                                {19,  46,   47,  115},
                                {28,  99,   132, 247},
                                {16,  37,   34,  90},
                                {16,  44,   48,  110},
                                {36,  131,  164, 299},
                                {18,  44,   44,  108},
                                {29,  103,  137, 254},
                                {22,  55,   62,  141},
                                {26,  70,   88,  179},
                                {17,  36,   30,  85},
                                {20,  55,   66,  142},
                                {21,  57,   70,  147},
                                {19,  52,   63,  135},
                                {37,  142,  183, 324},
                                {29,  100,  132, 243},
                                {17,  41,   38,  98},
                                {20,  50,   55,  127},
                                {17,  41,   37,  99},
                                {26,  79,   99,  202},
                                {84,  231,  248, 764},
                                {24,  77,   93,  187},
                                {23,  68,   85,  177},
                                {82,  193,  87,  264},
                                {154, 350,  170, 525},
                                {249, 508,  169, 496},
                                {18,  38,   29,  86},
                                {27,  128,  50,  177},
                                {58,  155,  68,  242},
                                {18,  50,   32,  101},
                                {19,  44,   34,  100},
                                {31,  129,  53,  184},
                                {36,  152,  61,  212},
                                {19,  53,   32,  106},
                                {24,  99,   45,  153},
                                {27,  124,  51,  177},
                                {23,  88,   41,  141},
                                {18,  43,   29,  93},
                                {29,  146,  54,  189},
                                {17,  37,   27,  87},
                                {19,  54,   33,  105},
                                {18,  38,   30,  88},
                                {26,  113,  46,  159},
                                {26,  52,   35,  92},
                                {18,  39,   29,  98},
                                {28,  52,   39,  123},
                                {20,  61,   35,  188},
                                {23,  89,   38,  126},
                                {20,  59,   37,  119},
                                {33,  152,  58,  204},
                                {24,  94,   37,  119},
                                {20,  62,   35,  114},
                                {25,  131,  54,  185},
                                {72,  64,   32,  108},
                                {21,  74,   38,  127},
                                {86,  180,  72,  246},
                                {19,  87,   40,  139},
                                {24,  120,  48,  166},
                                {58,  125,  90,  264},
                                {19,  46,   30,  96},
                                {138, 353,  133, 419},
                                {85,  212,  133, 436},
                                {18,  51,   32,  102},
                                {45,  184,  71,  247},
                                {29,  137,  53,  182},
                                {19,  54,   32,  107},
                                {19,  47,   30,  97},
                                {30,  76,   44,  144},
                                {169, 385,  136, 89},
                                {22,  69,   36,  116},
                                {261, 533,  175, 521},
                                {18,  44,   31,  94},
                                {29,  139,  52,  180},
                                {71,  153,  74,  249},
                                {19,  48,   31,  100},
                                {18,  45,   29,  96},
                                {20,  82,   39,  134},
                                {17,  56,   35,  113},
                                {20,  51,   33,  110},
                                {21,  74,   37,  126},
                                {17,  43,   29,  92},
                                {138, 327,  162, 502},
                                {17,  39,   29,  88},
                                {19,  45,   30,  97},
                                {28,  129,  51,  176},
                                {21,  61,   37,  119},
                                {60,  192,  84,  273},
                                {18,  50,   32,  101},
                                {17,  43,   30,  94},
                                {26,  108,  46,  147},
                                {59,  138,  73,  236},
                                {59,  138,  73,  238},
                                {102, 225,  158, 328},
                                {22,  74,   36,  114},
                                {17,  38,   28,  86},
                                {18,  51,   29,  92},
                                {19,  45,   29,  94},
                                {38,  109,  28,  89},
                                {29,  43,   30,  98},
                                {72,  249,  134, 448},
                                {36,  430,  152, 490},
                                {19,  43,   30,  96},
                                {18,  54,   32,  105},
                                {21,  82,   37,  125},
                                {18,  62,   31,  103},
                                {22,  82,   37,  126},
                                {34,  147,  58,  200},
                                {23,  76,   39,  129},
                                {35,  147,  59,  202},
                                {23,  80,   40,  132},
                                {69,  150,  111, 331},
                                {23,  86,   41,  136},
                                {32,  79,   48,  159},
                                {26,  117,  45,  155},
                                {18,  48,   31,  97},
                                {25,  86,   43,  141},
                                {18,  41,   29,  90},
                                {187, 390,  159, 494},
                                {19,  57,   34,  113},
                                {19,  40,   28,  90},
                                {157, 367,  142, 453},
                                {20,  69,   36,  121},
                                {17,  47,   30,  99},
                                {53,  72,   32,  150},
                                {202, 241,  55,  400},
                                {174, 215,  47,  307},
                                {48,  66,   31,  139},
                                {421, 493,  83,  717},
                                {322, 388,  67,  510},
                                {100, 125,  43,  251},
                                {24,  42,   30,  96},
                                {522, 632,  129, 958},
                                {522, 632,  129, 959},
                                {553, 681,  157, 1045},
                                {243, 293,  61,  456},
                                {312, 375,  72,  566},
                                {40,  60,   29,  134},
                                {56,  77,   34,  159},
                                {79,  102,  38,  204},
                                {429, 504,  84,  760},
                                {193, 235,  54,  351},
                                {39,  58,   31,  120},
                                {162, 198,  49,  340},
                                {24,  42,   28,  93},
                                {205, 246,  55,  419},
                                {195, 235,  54,  399},
                                {139, 171,  150, 941},
                                {346, 413,  98,  689},
                                {28,  44,   28,  100},
                                {388, 393,  76,  610},
                                {179, 214,  53,  367},
                                {170, 197,  50,  337},
                                {105, 142,  47,  292},
                                {473, 556,  93,  794},
                                {38,  56,   31,  122},
                                {53,  74,   36,  154},
                                {135, 167,  56,  321},
                                {327, 387,  74,  596},
                                {28,  45,   29,  104},
                                {127, 160,  45,  285},
                                {332, 414,  116, 626},
                                {88,  110,  39,  214},
                                {340, 407,  73,  601},
                                {332, 398,  71,  586},
                                {119, 146,  45,  286},
                                {48,  69,   35,  148},
                                {146, 181,  46,  300},
                                {132, 159,  44,  293},
                                {170, 205,  51,  352},
                                {87,  110,  38,  220},
                                {220, 262,  57,  438},
                                {456, 535,  88,  768},
                                {136, 165,  59,  428},
                                {172, 211,  52,  342},
                                {70,  91,   35,  181},
                                {415, 493,  83,  693},
                                {27,  46,   30,  105},
                                {322, 386,  71,  568},
                                {151, 184,  48,  313},
                                {82,  103,  38,  200},
                                {82,  103,  38,  200},
                                {148, 180,  49,  318},
                                {146, 199,  89,  456},
                                {121, 148,  44,  275},
                                {323, 381,  74,  665},
                                {502, 624,  150, 939},
                                {111, 138,  41,  320},
                                {156, 191,  47,  301},
                                {159, 195,  52,  351},
                                {474, 566,  111, 836},
                                {40,  58,   29,  120},
                                {22,  40,   27,  81},
                                {212, 273,  68,  406},
                                {320, 355,  66,  529},
                                {527, 608,  116, 894},
                                {211, 274,  59,  433},
                                {150, 178,  48,  315},
                                {106, 133,  44,  269},
                                {198, 241,  63,  410},
                                {490, 580,  102, 836},
                                {49,  67,   32,  137},
                                {60,  81,   41,  175},
                                {164, 212,  74,  381},
                                {205, 243,  56,  402},
                                {199, 237,  57,  397},
                                {503, 606,  118, 909},
                                {33,  51,   29,  113},
                                {253, 301,  60,  444},
                                {331, 390,  75,  604},
                                {64,  83,   33,  164},
                                {95,  118,  39,  238},
                                {215, 267,  82,  465},
                                {21,  38,   28,  90},
                                {417, 491,  84,  721},
                                {160, 197,  54,  343},
                                {72,  92,   35,  185},
                                {172, 208,  55,  367},
                                {497, 593,  107, 863},
                                {233, 296,  118, 589},
                                {576, 719,  180, 1087},
                                {581, 725,  185, 1089},
                                {132, 163,  45,  266},
                                {88,  111,  40,  217},
                                {101, 60,   36,  170},
                                {54,  48,   33,  128},
                                {51,  43,   29,  119},
                                {208, 81,   40,  276},
                                {144, 144,  88,  309},
                                {37,  42,   30,  111},
                                {26,  41,   31,  100},
                                {25,  41,   31,  101},
                                {71,  50,   32,  141},
                                {231, 224,  89,  341},
                                {102, 58,   34,  167},
                                {249, 141,  64,  358},
                                {68,  50,   34,  148},
                                {90,  55,   34,  155},
                                {275, 140,  67,  377},
                                {173, 115,  55,  268},
                                {43,  47,   34,  119},
                                {43,  47,   34,  119},
                                {93,  53,   31,  18},
                                {226, 87,   39,  297},
                                {55,  50,   35,  127},
                                {204, 78,   36,  250},
                                {21,  40,   33,  107},
                                {84,  54,   34,  153},
                                {245, 93,   36,  276},
                                {146, 83,   49,  235},
                                {240, 99,   44,  289},
                                {216, 79,   37,  269},
                                {140, 63,   39,  177},
                                {167, 159,  126, 465},
                                {120, 61,   34,  182},
                                {69,  47,   32,  128},
                                {75,  50,   33,  141},
                                {77,  52,   32,  157},
                                {83,  54,   34,  153},
                                {201, 81,   39,  251},
                                {335, 254,  87,  431},
                                {419, 366,  130, 550},
                                {95,  87,   56,  228},
                                {30,  41,   30,  106},
                                {219, 84,   41,  273},
                                {33,  43,   31,  110},
                                {131, 128,  71,  272},
                                {91,  56,   35,  160},
                                {119, 60,   35,  179},
                                {27,  43,   33,  105},
                                {278, 206,  117, 498},
                                {75,  65,   45,  181},
                                {65,  52,   34,  147},
                                {80,  55,   35,  168},
                                {25,  41,   31,  102},
                                {198, 79,   38,  244},
                                {63,  47,   31,  141},
                                {103, 59,   35,  171},
                                {23,  40,   32,  100},
                                {84,  53,   33,  151},
                                {388, 326,  137, 562},
                                {303, 173,  97,  499},
                                {301, 171,  96,  490},
                                {196, 86,   43,  243},
                                {68,  51,   35,  150},
                                {137, 143,  71,  266},
                                {124, 115,  95,  335},
                                {52,  51,   38,  155},
                                {298, 169,  90,  363},
                                {37,  43,   33,  104},
                                {297, 171,  88,  458},
                                {58,  48,   32,  132},
                                {63,  48,   32,  140},
                                {259, 127,  99,  467},
                                {251, 255,  112, 404},
                                {217, 185,  47,  217},
                                {89,  78,   38,  222},
                                {295, 196,  100, 440},
                                {95,  57,   34,  146},
                                {208, 112,  53,  324},
                                {121, 61,   35,  181},
                                {80,  53,   35,  148},
                                {60,  48,   33,  138},
                                {216, 86,   42,  286},
                                {96,  58,   36,  165},
                                {81,  53,   35,  151},
                                {288, 154,  82,  410},
                                {76,  52,   34,  144},
                                {192, 78,   38,  240},
                                {88,  56,   35,  174},
                                {140, 69,   37,  181},
                                {161, 71,   37,  217},
                                {338, 231,  126, 529},
                                {93,  58,   36,  154},
                                {38,  45,   33,  115},
                                {108, 57,   33,  175},
                                {114, 58,   33,  180},
                                {120, 59,   35,  189},
                                {88,  54,   33,  155},
                                {286, 166,  83,  424},
                                {118, 83,   47,  213},
                                {36,  42,   31,  111},
                                {29,  42,   31,  105},
                                {224, 83,   38,  271},
                                {19,  40,   33,  96},
                                {19,  40,   33,  95},
                                {21,  42,   33,  97},
                                {20,  42,   33,  97},
                                {21,  42,   33,  100},
                                {18,  40,   32,  96},
                                {19,  40,   33,  95},
                                {19,  40,   33,  96},
                                {19,  41,   33,  96},
                                {19,  40,   33,  96},
                                {19,  40,   33,  96},
                                {18,  40,   33,  93},
                                {19,  40,   32,  95},
                                {19,  40,   32,  96},
                                {21,  41,   32,  98},
                                {20,  41,   33,  97},
                                {20,  41,   33,  96},
                                {19,  40,   33,  96},
                                {20,  41,   33,  96},
                                {19,  40,   32,  96},
                                {19,  40,   32,  95},
                                {20,  41,   33,  97},
                                {19,  40,   33,  97},
                                {18,  40,   32,  95},
                                {19,  41,   33,  96},
                                {19,  40,   33,  93},
                                {18,  40,   32,  97},
                                {20,  41,   33,  97},
                                {18,  40,   33,  95},
                                {19,  40,   33,  97},
                                {21,  42,   33,  96},
                                {19,  40,   32,  96},
                                {19,  41,   33,  96},
                                {20,  39,   32,  95},
                                {20,  40,   32,  95},
                                {20,  40,   33,  96},
                                {19,  40,   33,  95},
                                {19,  41,   32,  97},
                                {20,  41,   32,  96},
                                {18,  39,   33,  95},
                                {20,  42,   33,  98},
                                {19,  40,   33,  95},
                                {19,  40,   32,  96},
                                {20,  40,   32,  96},
                                {19,  41,   32,  96},
                                {21,  41,   32,  96},
                                {19,  41,   32,  96},
                                {19,  41,   33,  96},
                                {20,  40,   33,  96},
                                {20,  40,   32,  94},
                                {19,  41,   33,  97},
                                {20,  40,   33,  97},
                                {20,  40,   33,  96},
                                {19,  40,   33,  96},
                                {18,  39,   33,  93},
                                {19,  40,   32,  95},
                                {19,  40,   33,  96},
                                {20,  41,   33,  96},
                                {19,  40,   33,  95},
                                {19,  40,   33,  96},
                                {18,  39,   33,  94},
                                {20,  41,   32,  96},
                                {19,  40,   32,  96},
                                {19,  40,   32,  96},
                                {20,  41,   33,  97},
                                {19,  40,   33,  95},
                                {20,  40,   33,  96},
                                {18,  40,   32,  94},
                                {20,  41,   33,  97},
                                {20,  41,   33,  95},
                                {18,  40,   32,  95},
                                {18,  40,   32,  95},
                                {20,  40,   32,  98},
                                {19,  40,   33,  96},
                                {19,  40,   33,  97},
                                {21,  42,   33,  96},
                                {19,  41,   33,  96},
                                {20,  41,   33,  96},
                                {20,  42,   34,  99},
                                {20,  40,   33,  96},
                                {19,  41,   33,  96},
                                {21,  42,   33,  98},
                                {18,  40,   33,  95},
                                {18,  41,   33,  98},
                                {21,  42,   33,  99},
                                {19,  41,   32,  97},
                                {19,  41,   33,  96},
                                {19,  40,   32,  96},
                                {19,  41,   32,  96},
                                {19,  41,   33,  97},
                                {19,  41,   33,  97},
                                {21,  41,   33,  97},
                                {19,  41,   33,  96},
                                {20,  40,   33,  97},
                                {19,  41,   33,  96},
                                {20,  42,   33,  98},
                                {19,  40,   33,  96},
                                {19,  41,   33,  97},
                                {25,  45,   34,  105},
                                {19,  40,   32,  95},};
    Matrix *train_set[test_count * 7];
    Matrix *target_set[test_count * 7];
    // get the max of the train_set
    max_train_set = 0;
    int max_train_set_4 = 0;
    for (int i = 0; i < test_count * 7; i++) {
        for (int j = 0; j < 3; j++) {
            // we don't want values less than 1000 cause they are too big and may cause overflow
            if (training_set[i][j] > 1000) {
                training_set[i][j] = 1000;
            }
            if (training_set[i][j] > max_train_set) {
                max_train_set = training_set[i][j];
            }
        }
        // the 4th value is often much larger than the others, so im normalizing them separately
        if (training_set[i][3] > max_train_set_4) {
            max_train_set_4 = training_set[i][3];
        }
    }
    printf("max_train_set: %f\n", max_train_set);
    // normalize the train_set
//    for (int i = 0; i < test_count * 7; i++) {
//        for (int j = 0; j < 3; j++) {
//            training_set[i][j] /= (max_train_set / 2);
//            training_set[i][j] -= 1;
////            printf("%f ", training_set[i][j]);
//        }
//        training_set[i][3] /= (max_train_set_4 / 2);
//        training_set[i][3] -= 1;
//    }
    for (int i = 0; i < test_count * 7; i++) {
//        target_set[i] = init_matrix(7, 1);
        for (int j = 0; j < 4; j++) {
//            training_set[i][j] = training_set[i][j] / max_train_set;
//            printf("%f ", training_set[i][j]);
        }
        train_set[i] = init_matrix_array(4, 1, training_set[i]);
//        print_matrix_desc(train_set[i], "train_set");
    }
    // create target set
    for (int i = 0; i < 7; i++) {
        // we want the network to set the value of the result to 1, so we can get the max value (index) of the output and thats the result
        double target[] = {0, 0, 0, 0, 0, 0, 0};
        target[i] = 1;
        for (int j = 0; j < test_count; j++) {
            target_set[i * test_count + j] = init_matrix_array(7, 1, target);
        }
    }
    // convert it to a TrainSet
    TrainSet *ts = malloc(sizeof(TrainSet));
    for (int i = 0; i < test_count * 7; i++) {
        ts->input[i] = train_set[i];
        ts->target[i] = target_set[i];
//        // free memory
//        matrix_release(train_set[i]);
//        matrix_release(target_set[i]);
    }
    return ts;
}