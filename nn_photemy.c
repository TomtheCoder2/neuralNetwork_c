#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


#define printf printf
double l_rate = 0.000056;
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

//// Matrix *weights[contestantCount][layerCount (index i)];
//// Matrix *biases][contestantCount][layerCount (index i);
//// training data is the same for all contestants
//// int layerSizes[contestantCount][layerCount];
//// int layerCounts[contestantCount];
//__kernel void fit(__global const double *input, __global const double *target, const int test_count,
//                  __global double *weights, __global double *biases,
//                  __global const int *layerSizes, __global const int *layerCounts, const int contestantCount, const int *epochs, __global const double *learning_rates) {
void fitK(const double *input, const double *target, const int test_count,
          double *weights, double *biases,
          const int *layerSizes, const int *layerCounts, const int contestantCount, const int *epochs, const double *learning_rates) {
//    int id = get_global_id(0);
    // get the training data for this contestant
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
//            layer_index ++;
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
//                    layer_index ++;
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
//                    layer_index ++;
//                }
//            }
//            biases_curr[i]->data[j] = biases[index + i * biases_curr[i]->rows * biases_curr[i]->cols + j];
//        }
//    }
//    // get the learning rate for this contestant
//    l_rate = learning_rates[contestantCount];
//    // fit the model for this contestant
//    fit(test_count, train_set, target_set, *epochs, layerCount, weights_curr, biases_curr);
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
    double a0[] = {0.4617563814065817, -0.17983837701559668, -0.5845703173805659, -0.33456588808097765, 0.9355118188482414, -0.9877656354684774, 0.9274095940464153, 0.8797307775638197, 0.8943898353263877, 0.8741642977919393, -0.2056513156305888, -0.3049639415937795, -0.41188593599192647, 0.012967254652470173, -0.7680658239346845, 0.5410717601583555, 0.31978541738683997, -0.6865062188603075, -0.24359590935788944, -0.7204746341924977, 0.38989595920498377, 0.6104555429474274, -0.9899496480150949, 0.04627031157666606, 0.4879689724746332, -0.7159545935681477, -0.03654339684880403, 0.08910961778734738, 0.154200522748553, -0.5901729084828768, 0.2467276212633316, -0.6305858194441027, -0.9786311780355423, -0.6779133532565955, -0.6438903067256505, 0.08079412956759291, 0.9476680198374003, -0.5091468928374088, -0.21095818741155647, -0.5647957501846304, -0.13598624021292172, -0.5336884104973332, 0.7798175033107542, -0.9233462133614492, 0.18475848772587744, 0.31034720625883283, -0.7603219153916208,
                   0.3049534137919063, 0.9686456013385898, -0.5865249852789853, -0.2507009994120597, -0.07330042486104849, -0.3327795776506606, -0.11357509936546184, 0.008271133409147868, 0.9979616295358575, 0.2608097278457078, 0.8191680711780485, 0.015319832139896183, -0.017097588876774816, -0.14243164250759865, -0.3838141695830404, 0.43464480324445387, 0.9248291854674928, -0.5810145330728713, -0.6547023581719895, 0.09790503386131122, 0.11093676417592313, 0.17563165780266687, 0.5783813156701916, 0.39801324486896106, -0.590728895588629, -0.49018109740036864, 0.5558327190390557, -0.5499552279832491, 0.9662063696674061, 0.607237212030854, 0.6726298955126573, -0.6737929452233706, 0.27499485971881277, -0.9824791653625367, 0.2623797766230498, -0.584044041914755, 0.7607145698140847, 0.41408845210793044, 0.44674329243160504, 0.018985839340738497, 0.9742042761925997, -0.6935075343674779, 0.4211495502966689, 0.6479617509063196, -0.7526575429941516, 0.23390277349257982, -0.03627528947880365,
                   -0.8004994373485674, 0.23429177972286364, -0.9437592540976656, 0.18287821190927622, -0.8539479595723121, -0.8519041454533085, 0.3598708403820501, -0.43999676843466795, 0.2079673753671345, 0.6992816140285527, -0.2923534344590508, 0.19726497654931419, -0.41581317728749156, 0.7528853066489665, 0.48849924027210045, -0.15353454226742125, -0.24838358251388248, -0.45741334965056835, 0.459647857870199, 0.7734888447426316, -0.9125946034063506, -0.0845996338529802, 0.724899509523101, 0.8038479114846022, -0.8750230801744465, 0.5276409763493284, -0.1445998371634829, 0.9983668615811874, -0.8761963954427474, 0.17264126398586965, 0.13458614180741013, 0.3240519226107552, 0.8145623802363859, -0.5070287373953819, -0.5030224658253748, 0.41371666950135744, -0.3245581454720592, 0.21467015792151267, -0.3691544270850342, -0.951055740329511, -0.09798299476910088, -0.26173136899734595, -0.0723244806805814, -0.6455145763740981, 0.3522463090507362, 0.43114496998271745, 0.48673924553053105,
                   0.8582052208550688, 0.17978705894087765, 0.28802457476710486, -0.7790202931650292, 0.5492082189178475, -0.3870344237893202, 0.7611973395786324, 0.26080288156998255, 0.857787426649109, -0.6066211981784508, 0.5092924905221334, -0.6847387723847258, -0.9156426320965521, -0.24275424927878264, 0.5732184915636163, 0.42550433553218703, -0.7794527769815014, 0.5242661394785624, -0.9398623282422005, -0.3667370420546159, 0.33414670639259225, -0.1471139051811421, -0.19888567737380747, 0.48752309522700377, -0.2693885244307648, 0.527509798143645, -0.3546616179089326, 0.025437646764806088, -0.04264825558681129, -0.7376623737501162, -0.9364070460294214, -0.11108677780932652, 0.38304242336296523, 0.43815137069360843, 0.5559005447604162, -0.42379856790026804, 0.5627094672588622, 0.9325079013644189, 0.9930469030238771, -0.5217871273450587, -0.10963747653244882, -0.7395561439767482, -0.5310518249898872, 0.3121648779767989, 0.14445918501177846, 0.11193326222583289, -0.4994074888102318,
                   0.840562385096266, 0.4475024168644697, 0.553981931358702, -0.5154477144827585, -0.6485960909997945, -0.11041198807748454, 0.6593824466983986, 0.20382927212011515, 0.2758937921094, -0.39326571048193704, -0.8692059577422617, 0.8760610024651603, -0.46716658534520294, -0.5416572832644959, -0.39204712487533855, -0.25611897302693687, 0.7142540853584027, 0.6141527310483261, -0.22254144849415747, -0.46950707793809965, -0.9352542931981471, 0.3395705269641569, -0.8512772474037791, 0.2182821628673659, -0.6426752811796594, -0.9407335772053, 0.4163076558810881, -0.44444345058469303, -0.7684231144477478, 0.3824518650005857, -0.3019785521812086, -0.5437289954810616, -0.6648000080008807, 0.1892622659742289, 0.6197796579677197, 0.02985828169711513, -0.3020372245568883, -0.8938130714882238, 0.2123804084067331, 0.3630529243217582, 0.1674866344119328, -0.2604950014005061, 0.8919840619507564, 0.3728611665601458, -0.5508522595776915, -0.11080113051030072, 0.41522708850908274,
                   -0.9404045646350769, 0.3177872479269317, -0.3681649705532324, 0.6315404829028237, -0.6888427297301514, -0.09319152829934141, -0.36058366622658444, 0.09938505757310678, 0.04451849300256239, -0.3693699674372739, 0.39222576450162183, -0.8102451896845759, 0.9176161100317948, -0.7406448221923483, -0.952802763757856, -0.8305583514396848, -0.8953584308861684, -0.060092800991414785, 0.6393647783990011, 0.9381944945543601, -0.3000286259712659, -0.4542271181418751, -0.977542944962007, 0.5880466869499925, -0.7992492604291472, 0.6378848064003326, 0.49571961454667535, -0.24027382652589435, -0.7845912736244673, 0.5320655384878354, -0.5824367631047411, 0.8902041803858236, -0.9448522188144981, 0.4938463541049951, 0.2726517766875447, 0.8770772247487733, -0.8122829453839511, 0.9193644982735831, 0.6375592204524125, -0.5703801304492622, 0.6675442057285212, -0.4885674170728609, 0.28114096218840445, 0.4329845654626039, 0.6035657632551754, -0.07723458209616907, -0.760720375079807,
                   0.021166654236003835, 0.485862421006954, -0.8862599023879809, 0.11968219083981557, -0.26400725354637866, 0.7644703964442963, 0.759954413758579, -0.8790802317447803, -0.8564689514905641, -0.5414934855402953, -0.554758540226437, 0.2909182221813966, -0.4140360075988738, 0.7710093572261563,};
    weights[0] = init_matrix_from_array(74, 4, a0);
    double b0[] = {0.4617563814065817, -0.17983837701559668, -0.5845703173805659, -0.33456588808097765, 0.9355118188482414, -0.9877656354684774, 0.9274095940464153, 0.8797307775638197, 0.8943898353263877, 0.8741642977919393, -0.2056513156305888, -0.3049639415937795, -0.41188593599192647, 0.012967254652470173, -0.7680658239346845, 0.5410717601583555, 0.31978541738683997, -0.6865062188603075, -0.24359590935788944, -0.7204746341924977, 0.38989595920498377, 0.6104555429474274, -0.9899496480150949, 0.04627031157666606, 0.4879689724746332, -0.7159545935681477, -0.03654339684880403, 0.08910961778734738, 0.154200522748553, -0.5901729084828768, 0.2467276212633316, -0.6305858194441027, -0.9786311780355423, -0.6779133532565955, -0.6438903067256505, 0.08079412956759291, 0.9476680198374003, -0.5091468928374088, -0.21095818741155647, -0.5647957501846304, -0.13598624021292172, -0.5336884104973332, 0.7798175033107542, -0.9233462133614492, 0.18475848772587744, 0.31034720625883283, -0.7603219153916208,
                   0.3049534137919063, 0.9686456013385898, -0.5865249852789853, -0.2507009994120597, -0.07330042486104849, -0.3327795776506606, -0.11357509936546184, 0.008271133409147868, 0.9979616295358575, 0.2608097278457078, 0.8191680711780485, 0.015319832139896183, -0.017097588876774816, -0.14243164250759865, -0.3838141695830404, 0.43464480324445387, 0.9248291854674928, -0.5810145330728713, -0.6547023581719895, 0.09790503386131122, 0.11093676417592313, 0.17563165780266687, 0.5783813156701916, 0.39801324486896106, -0.590728895588629, -0.49018109740036864, 0.5558327190390557,};
    biases[0] = init_matrix_from_array(74, 1, b0);
    double a1[] = {0.4617563814065817, -0.17983837701559668, -0.5845703173805659, -0.33456588808097765, 0.9355118188482414, -0.9877656354684774, 0.9274095940464153, 0.8797307775638197, 0.8943898353263877, 0.8741642977919393, -0.2056513156305888, -0.3049639415937795, -0.41188593599192647, 0.012967254652470173, -0.7680658239346845, 0.5410717601583555, 0.31978541738683997, -0.6865062188603075, -0.24359590935788944, -0.7204746341924977, 0.38989595920498377, 0.6104555429474274, -0.9899496480150949, 0.04627031157666606, 0.4879689724746332, -0.7159545935681477, -0.03654339684880403, 0.08910961778734738, 0.154200522748553, -0.5901729084828768, 0.2467276212633316, -0.6305858194441027, -0.9786311780355423, -0.6779133532565955, -0.6438903067256505, 0.08079412956759291, 0.9476680198374003, -0.5091468928374088, -0.21095818741155647, -0.5647957501846304, -0.13598624021292172, -0.5336884104973332, 0.7798175033107542, -0.9233462133614492, 0.18475848772587744, 0.31034720625883283, -0.7603219153916208,
                   0.3049534137919063, 0.9686456013385898, -0.5865249852789853, -0.2507009994120597, -0.07330042486104849, -0.3327795776506606, -0.11357509936546184, 0.008271133409147868, 0.9979616295358575, 0.2608097278457078, 0.8191680711780485, 0.015319832139896183, -0.017097588876774816, -0.14243164250759865, -0.3838141695830404, 0.43464480324445387, 0.9248291854674928, -0.5810145330728713, -0.6547023581719895, 0.09790503386131122, 0.11093676417592313, 0.17563165780266687, 0.5783813156701916, 0.39801324486896106, -0.590728895588629, -0.49018109740036864, 0.5558327190390557, -0.5499552279832491, 0.9662063696674061, 0.607237212030854, 0.6726298955126573, -0.6737929452233706, 0.27499485971881277, -0.9824791653625367, 0.2623797766230498, -0.584044041914755, 0.7607145698140847, 0.41408845210793044, 0.44674329243160504, 0.018985839340738497, 0.9742042761925997, -0.6935075343674779, 0.4211495502966689, 0.6479617509063196, -0.7526575429941516, 0.23390277349257982, -0.03627528947880365,
                   -0.8004994373485674, 0.23429177972286364, -0.9437592540976656, 0.18287821190927622, -0.8539479595723121, -0.8519041454533085, 0.3598708403820501, -0.43999676843466795, 0.2079673753671345, 0.6992816140285527, -0.2923534344590508, 0.19726497654931419, -0.41581317728749156, 0.7528853066489665, 0.48849924027210045, -0.15353454226742125, -0.24838358251388248, -0.45741334965056835, 0.459647857870199, 0.7734888447426316, -0.9125946034063506, -0.0845996338529802, 0.724899509523101, 0.8038479114846022, -0.8750230801744465, 0.5276409763493284, -0.1445998371634829, 0.9983668615811874, -0.8761963954427474, 0.17264126398586965, 0.13458614180741013, 0.3240519226107552, 0.8145623802363859, -0.5070287373953819, -0.5030224658253748, 0.41371666950135744, -0.3245581454720592, 0.21467015792151267, -0.3691544270850342, -0.951055740329511, -0.09798299476910088, -0.26173136899734595, -0.0723244806805814, -0.6455145763740981, 0.3522463090507362, 0.43114496998271745, 0.48673924553053105,
                   0.8582052208550688, 0.17978705894087765, 0.28802457476710486, -0.7790202931650292, 0.5492082189178475, -0.3870344237893202, 0.7611973395786324, 0.26080288156998255, 0.857787426649109, -0.6066211981784508, 0.5092924905221334, -0.6847387723847258, -0.9156426320965521, -0.24275424927878264, 0.5732184915636163, 0.42550433553218703, -0.7794527769815014, 0.5242661394785624, -0.9398623282422005, -0.3667370420546159, 0.33414670639259225, -0.1471139051811421, -0.19888567737380747, 0.48752309522700377, -0.2693885244307648, 0.527509798143645, -0.3546616179089326, 0.025437646764806088, -0.04264825558681129, -0.7376623737501162, -0.9364070460294214, -0.11108677780932652, 0.38304242336296523, 0.43815137069360843, 0.5559005447604162, -0.42379856790026804, 0.5627094672588622, 0.9325079013644189, 0.9930469030238771, -0.5217871273450587, -0.10963747653244882, -0.7395561439767482, -0.5310518249898872, 0.3121648779767989, 0.14445918501177846, 0.11193326222583289, -0.4994074888102318,
                   0.840562385096266, 0.4475024168644697, 0.553981931358702, -0.5154477144827585, -0.6485960909997945, -0.11041198807748454, 0.6593824466983986, 0.20382927212011515, 0.2758937921094, -0.39326571048193704, -0.8692059577422617, 0.8760610024651603, -0.46716658534520294, -0.5416572832644959, -0.39204712487533855, -0.25611897302693687, 0.7142540853584027, 0.6141527310483261, -0.22254144849415747, -0.46950707793809965, -0.9352542931981471, 0.3395705269641569, -0.8512772474037791, 0.2182821628673659, -0.6426752811796594, -0.9407335772053, 0.4163076558810881, -0.44444345058469303, -0.7684231144477478, 0.3824518650005857, -0.3019785521812086, -0.5437289954810616, -0.6648000080008807, 0.1892622659742289, 0.6197796579677197, 0.02985828169711513, -0.3020372245568883, -0.8938130714882238, 0.2123804084067331, 0.3630529243217582, 0.1674866344119328, -0.2604950014005061, 0.8919840619507564, 0.3728611665601458, -0.5508522595776915, -0.11080113051030072, 0.41522708850908274,
                   -0.9404045646350769, 0.3177872479269317, -0.3681649705532324, 0.6315404829028237, -0.6888427297301514, -0.09319152829934141, -0.36058366622658444, 0.09938505757310678, 0.04451849300256239, -0.3693699674372739, 0.39222576450162183, -0.8102451896845759, 0.9176161100317948, -0.7406448221923483, -0.952802763757856, -0.8305583514396848, -0.8953584308861684, -0.060092800991414785, 0.6393647783990011, 0.9381944945543601, -0.3000286259712659, -0.4542271181418751, -0.977542944962007, 0.5880466869499925, -0.7992492604291472, 0.6378848064003326, 0.49571961454667535, -0.24027382652589435, -0.7845912736244673, 0.5320655384878354, -0.5824367631047411, 0.8902041803858236, -0.9448522188144981, 0.4938463541049951, 0.2726517766875447, 0.8770772247487733, -0.8122829453839511, 0.9193644982735831, 0.6375592204524125, -0.5703801304492622, 0.6675442057285212, -0.4885674170728609, 0.28114096218840445, 0.4329845654626039, 0.6035657632551754, -0.07723458209616907, -0.760720375079807,
                   0.021166654236003835, 0.485862421006954, -0.8862599023879809, 0.11968219083981557, -0.26400725354637866, 0.7644703964442963, 0.759954413758579, -0.8790802317447803, -0.8564689514905641, -0.5414934855402953, -0.554758540226437, 0.2909182221813966, -0.4140360075988738, 0.7710093572261563, 0.08868715483308742, 0.40362185197409084, -0.6338976041733178, -0.3053344801905238, -0.9204112478252877, -0.5014300699616621, 0.593362607679703, 0.41585420017249497, -0.0678168917254871, 0.6066398889726063, -0.8358319871083821, -0.04078885930975695, 0.9223636956218182, -0.5237730174200228, -0.09362361593075197, -0.03782443553414039, -0.8685032291790877, -0.7823532007301461, 0.7864765742550983, 0.7287528662943221, 0.5978363044994466, -0.3004387865358291, -0.740062733569075, 0.7839879246135268, -0.1360703359664024, -0.7402549205104585, 0.33923187827797374, 0.03541276588258202, 0.10879520691077893, 0.8335257327328773, -0.2392958342417062, -0.6550048188585313, -0.7334556545329904,
                   -0.043255722716780465, 0.7505799471975545, -0.17511741572253237, 0.058511332327899845, 0.26610137326114214, -0.6779300180015784, -0.5075003126890016, 0.7038436325218125, -0.9017322953121019, 0.7796517360338735, -0.9446747387303911, 0.22389816722296274, -0.4198460762437439, -0.8367444080582236, 0.5429373153513242, -0.28132271049682855, 0.20625050403561307, -0.26577579064015633, 0.263684137807622, 0.7689307137181995, 0.31496408683598087, 0.2865532371047357, -0.8643989460574832, 0.20543900210088784, 0.10439199214576789, 0.06106168814262758, 0.8259616291751632, 0.8873857705804806, 0.8689926148048137, -0.6532178585906796, -0.6688846916915596, 0.8446155768889705, 0.9921245603305735, 0.47300071741863303, 0.6380300766818798, 0.16396586531348323, -0.10455643996500963, -0.1729822345961347, -0.06877578578197374, 0.3377512645671197, -0.3753384059915801, 0.7768403719106152, 0.4435373478338622, 0.4428696503882732, 0.04218769023983593, 0.9346589722418666, 0.49645519848127484,
                   0.4818898150939612, 0.7006870619845313, 0.06610090762259224, -0.46453011839092007, 0.4928710774641969, 0.3876087522264322, -0.868106217994175, 0.03822173798830253, 0.622556343870637, 0.05924937330418878, 0.4427129596431163, 0.05793522726276468, 0.2579697305072117, 0.4239353347697836, -0.285372502144996, 0.8619265945813017, -0.47565821896263705, -0.21257848901206233, 0.9316998514658994, -0.36527315220597223, -0.017325085220385228, 0.9396202840385481, 0.35393508757797476, 0.5376287774531514, -0.8018474656154688, -0.1597096167192229, 0.7529158270059466, 0.5512336090277055, 9.864339512639653E-4, -0.3476695330819941, -0.9858995135582167, -0.5251636700735824, -0.27581265182058545, -0.7251620171299673, -0.3860980191413501, 0.21625679994852565, -0.521866405676193, -0.055890195764195516, -0.8386945947268156, 0.07798773037047702, 0.609704658962309, 0.6992764155353963, -0.812076482456136, 0.13029646449166, 0.894135919760972, -0.4497228973625478, -0.32297012246989687,
                   -0.040815831976717565, 0.720642058084145, 0.039399522796289776, 0.44124938512551726, -0.913129915858155, -0.44175268996846273, 0.5536220392507769, -0.6097726471838123, 0.1847354089696971, -0.6738490912803419, 0.22351377096764224, -0.026261683719869255, 0.8334262707136533, -0.056653037754744284, 0.535461587845538, 0.6423367941140314, 0.03208146833942638, -0.7048910956801153, -0.11248933074675826, -0.9632074244754596, 0.9184153016710372, -0.8706614056267428, -0.7648809961689669, 0.7991938560330296, 0.2336529082846095, -0.016328097395461816, 0.7016432551773015, -0.6072576421085734, -0.048920470112241876, 0.1446212756320453, -0.4306116130452171, -0.8760956088910088, -0.8157679476940964, -0.14707094114558816, -0.5794309261038282, 0.6797237917918408, -0.9994739186478907, 0.4946937454493163, 0.3635469837486227, 0.9247420187247666, -0.9116223178326195, -0.6442774520991257, -0.7417773483731642, 0.4461120161383725, -0.639355738230065, -0.3330444891104416, 0.645477965463219,
                   -0.607867100619121, -0.47552435792047776, -0.8683051296075128, -0.31092101923671067, 0.518572955020401, -0.5925698995294806, -0.09923385767834181, -0.14553411175917796, 0.03878964260051876, 0.4019781697103719, -0.9503736708631689, 0.27737171727201826, -0.1322048145007273, -0.22471932391323746, -0.9881736596714608, 0.8162276457730837, -0.8429426429125628, 0.7883938571634896, 0.13612503034764178, 0.6358275930365096, 0.6359656857276144, 0.5788206673153771, 0.5599425363677661, 0.8357412255018737, 0.20753402474083948, 0.5825913148025064, -0.07026813851511182, -0.8845201354264969, 0.4900506056408893, 0.9068237285250427, -0.8507622193807312, 0.7594347060808979, 0.5530278048144603, 0.5109787241829884, 0.299502934628346, -0.671876164389478, 0.9649444748963754, -0.685209149603812, -0.8501284115000383, 0.35829841810365926, -0.5480863862633054, 0.42312192083773237, -0.732867710200992, 0.1029759189417172, -0.03944995170988741, 0.5427486778893642, -0.36966902225923626,
                   0.8834819756586672, 0.724951899803354, 0.3633634166936208, 0.8716769746398525, 0.9790996773829199, -0.9431484672252473, 0.18482999042936932, 0.7893550843837316, 0.14593968186272144, 0.009631615533212345, -0.6049950120350034, -0.3840529882517494, -0.16483916921641995, 0.8110429045705321, -0.4628012347408834, -0.3234434120273826, -0.028314028770306532, -0.9838998570148614, -0.3463839554134107, 0.5336234876586188, -0.24192615064738665, -0.29922503053686356, -0.6491171813880459, -0.5091986342809189, 0.3842236022640888, -0.3012573036323791, -0.39396939925394836, 0.4161658591572903, 0.7521027312034498, 0.8775829790888101, -0.2839243739074371, -0.7545902003670362, -0.6342924404621659, -0.24252641528954855, 0.9915953158988744, 0.23857250068975144, 0.38997815224162835, -0.6054429891121651, -0.19552901253261035, -0.9107583518628244, -0.45159332171499855, -0.6171982155764295, 0.42223078507581024, -0.6769441581380431, 0.00759419724783994, -0.9276815108513372,
                   -0.30380796705337954, -0.9272725855827195, -0.3119747495816658, -0.34126621434187476, -0.8821308709891407, 0.2796953588366484, -0.4561538698771319, 0.6673646226901524, -0.9821226430098648, -0.3520915640492801, -0.7295519295850796, -0.4624245380598375, 0.7538295413593057, 0.2631303795740023, 0.6271725760600964, -0.49315638493645064, -0.3815126586586235, 0.6915480463204711, -0.8965294905773702, 0.8006959120908563, -0.10330448591388719, 0.5843555047815325, 0.010364393175903208, -0.7461770556376985, 0.00939227612983995, 0.34037685332194423, -0.754923624319918, 0.9193159924791634, 0.005254846951199266, -0.7833189037391082, -0.04479766023508902, 0.9671134046085805, -0.20469470395736655, -0.5724658880611122, 0.11982629699117098, 0.22938346044068103, 0.4730661472707469, -0.7650786486055623, 0.6684865150649821, -0.8802888388988859, 0.8226227983882577, -0.19073198732031327, -0.8250811083914014, 0.3345976704260267, 0.6892134772240224, 0.7268721025348197, 0.3910200522279623,
                   0.7399207352106172, 0.2717342424659679, -0.6290429424350787, -0.07251264104068822, -0.848353794351494, -0.40618784546971076, 0.2161417094066418, -0.7588908973920898, 0.33922070907143187, -0.5801356304071028, -0.9984846408412742, 0.7546936820604886, 0.5117158056491005, -0.29517193588013013, 0.1353964829288823, -0.5858579181988721, -0.2090642637920841, 0.30101014553365313, -0.6747555925361473, -0.7218164054003873, 0.7323403152752526, -0.9072273031376121, -0.6109160659386819, 0.8398106188810868, 0.823776007794262, 0.9895347941323167, -0.5198068285025013, 0.6944638935796708, -0.9263519100027156, -0.7201629670645266, 0.5446958672289215, -0.10146558069665712, 0.294308013906023, 0.7339817752345161, 0.7320899297339862, 0.9027656528343644, -0.5069289682840381, -0.4599740356436812, -0.30836573079923113, -0.7013471445474342, 0.9024179729794466, 0.6676707627751233, 0.21290762172555655, -0.21804530520304577, -0.387421490451092, 0.5768566275791103, -0.49325803172605176,
                   -0.2840545791990332, 0.5138311578892838, 0.32263858042737015, 0.5731256829536848, 0.4680530099380771, -0.6875022413959091, -0.9448175848134761, 0.5655608805285528, 0.29704426398160444, -0.24625047995870175, 0.6734871059854508, -0.16324953185343505, 0.9839325259596392, -0.1340161526846857, 0.96370560544464, 0.4383807269750777, -0.3716995529924634, 0.035669692500730044, 0.9377010869752476, 0.24815712080704388, -0.9272553073572931, -0.9717981654719379, 0.2847409598947539, 0.6101394439360635, -0.7283332881933826, 0.689444235488935, 0.8589981915896283, -0.6001848120982507, -0.4133526708600188, -0.005717357711302551, -0.05220732359153346, -0.23676056291799608, 0.46915984471554517, 0.35264872681009574, 0.7002736562976659, -0.5700395946779857, 0.28239239332587673, 0.6401167407403248, 0.5435074204085659, 0.46234091636452157, -0.8825360223096717, 0.2893168005810065, -0.32729783285138114, 0.5792755342126303, 0.3753364627214939, -0.03995355907857889, 0.7745658663855679,
                   0.7801954936913502, 0.6142387056383138, -0.19989406945907695, 0.027559738451369276, -0.10435781795928478, 0.7034361411948704, -0.8844793316468778, 0.12007521909268593, 0.9035821895569645, 0.6776930457045598, 0.38795652325047403, 0.5373274658909377, 0.376026652413495, 0.8527920137966316, -0.9912261374225391, 0.6900250124386449, -0.7415914612649281, 0.7409893414351199, -0.15669373190956049, -0.4073600077314181, 0.33456198103471, -0.9036976642487022, 0.7955917579453529, -0.2536308528005682, -0.8069608948404163, -0.976962085867402, -0.08707687106674022, 0.6416417699211479, -0.7141112601275561, -0.5030099965414339, 0.25122692761300547, 0.8672963016556243, -0.24108559097533488, 0.8772465196230592, -0.2835812395146966, -0.8016877258250283, -0.8671828009382394, -0.9724354253298897, -0.5964742417392197, 0.23125118453806848, 0.6275218162712559, 0.5197859044879756, -0.42549499598413854, -0.8536760775577481, -0.052626854507112686, 0.05778230455092803, -0.6526698303731437,
                   0.9972695541500922, -0.3172179359570788, 0.7389347600210701, 0.4130563034533423, 0.5649946430767341, 0.13429023073038815, 0.06116657573839035, 0.6542161785034013, -0.7418110782939384, 0.32791286797029273, -0.31293375286775005, 0.40703996640773443, -0.19610235839743062, -0.733989135907994, 0.8926510483580905, 0.29163183152605443, -0.4567020627397289, 0.045832793128640636, 0.3568384814876586, -0.9716225163711605, -0.9673092594331327, 0.37825856823610193, 0.48870810182474345, 0.4654970683404236, 0.3469865997833925, 0.25216762428938044, 0.8431516438463558, -0.8382013920580704, 0.99272833236812, -0.984756082605506, 0.34152547553251544, -0.6996720350294034, 0.6599600960453813, -0.020078193637555364, -0.007565552390635055, 0.45275139126916697, -0.8229824684505238, 0.8135669905007901, 0.5987928010945316, -0.2290519767185033, 0.9795785439725129, 0.4348772949584534, -0.4591781612199468, -0.4485698360994994, 0.16508205764736217, -0.08612140653853495, -0.8593385750257452,
                   -0.2001512373550136, 0.775885013003772, -0.03921735871264653, 0.34278694347364325, -0.5830155986973558, -0.6607044606074441, 0.9893625420698455, -0.5145608901396439, 0.9342620561841317, 0.8898850358874271, -0.5466055721949568, 0.7917898385294038, 0.34597059674029484, -0.290752809791075, 0.18010201976947648, 0.8755831179359017, -0.48299256371899113, -0.1593610437371129, 0.7624338639234898, 0.034505042776222794, -0.11089010871689897, 0.07070200774293078, 0.7223918696333531, 0.3955532196899896, 0.7651755749260871, -0.48226915199068454, -0.8794791182867265, -0.9198090845472766, -0.6795862864170032, -0.774281537729727, -0.4379397629031516, -0.526912300711013, 0.6707118043444986, 0.2356754550142266, -0.8618517422079934, -0.5932371257662146, 0.8227403449081763, -0.5095141583860654, -0.2591390093228554, -0.3845352206759862, -0.8630388035983816, -0.373173663999425, 0.4037583248449763, -0.8348663955365285, 0.05384425948697613, -0.7825564233299249, -0.15500006511435416,
                   0.7323119250042129, 0.2652381572524276, 0.34934739322246466, -0.7720153649413144, -0.41661802166887374, -0.7430183526137382, 0.5582457135538257, 0.3048577475834606, -0.1034748623119337, 0.2054829051032745, 0.08681056733663062, -0.4288787865805488, 0.6723136916778734, 0.5801336570531954, -0.8058678140908144, 0.8349713893877317, 0.894861726002423, -0.5149324657692349, -0.18401831645527245, 0.692881897268179, -0.16602366445191885, -0.23208954817939986, -0.7397297528898326, -0.1966632763361964, -0.5802322147270143, -0.7701863394026327, 0.8193575442379664, -0.43648779365161783, -0.5253095948442648, 0.8901985264756289, -0.8093723586084163, 0.8190951824957751, -0.7459959923736266, -0.8087151560532184, -0.5809834934411349, 0.5753160758257923, 0.22416002136274193, -0.7283865009765556, 0.7099409779447519, 0.3276664870135284, -0.7003683355648855, -0.005584382194113058, 0.9083446417338634, -0.40633961174505884, 0.8622250014495452, 0.2858212856816138, -0.3006590086753631,
                   0.5023903915792294, -0.19707879372604564, 0.5195402179681892, 0.6582010109925887, -0.11348067595331357, 0.5548170558972816, 0.17878971032415003, 0.1621975710584247, 0.22173947511255787, 0.6267467674714693, -0.1671171442132624, 0.8740834247706191, -0.028378518686785537, -0.49931752188124, -0.41496646908328616, -0.9832526088085449, -0.5788340360411821, 0.9220279326406002, -0.4204270900264955, -0.019142331331424955, -0.07973826005898998, 0.5904340889563211, 0.9009148596560985, -0.05793430354283835, 0.6086445052710612, 0.049782275789519215, -0.7467535190981607, 0.11161312748630747, -0.8431590451212174, -0.8205051524126468, -0.8209380555456487, -0.9960544026524085, -0.16702584156756228, 0.10535052113475896, -0.7437022145828835, 0.6772065540086307, -0.7068044422010393, 0.5764952797079081, 0.7079684821957497, -0.7127526900603414, -0.10609867022519226, 0.896357165665767, -0.8351577615164509, 0.49305680955324704, 0.03677960077678177, -0.651312254417197, -0.05916364309882027,
                   0.26127915793410295, -0.9168732755952635, 0.7259689043211326, -0.8875459112753279, -0.5602468276765067, 0.7117619916978455, -0.9540601968592199, -0.06227642910321607, 0.10477362212849806, -0.7547415870780665, 0.8330134743082294, 0.706740856516819, -0.8659603915571152, 0.43210126132305904, -0.5551309520468217, -0.014939246140258877, -0.2829793726896921, -0.5522263452924139, -0.15947890448624813, 0.3928171883733613, 0.5867375868129254, 0.46566035840210374, 0.9146820876060422, 0.5565983605647593, -0.3582797061911087, 0.04905233552686772, -0.935748295317828, 0.6777406892034143, -0.6758303014392932, 0.8284237084562232, 0.006280196216895018, 0.27984652659382747, -0.36988180694135897, 0.4535439607684928, 0.032769787780565274, -0.8700278773346382, 0.9496470049444763, -0.35212014781467404, 0.8661940234840377, 0.07207122765240648, -0.9300147102913237, 0.3431306299319292, 0.0706293376745799, -0.43207735449296614, 0.621581912547694, 0.3346312285679538, -0.423090051276916,
                   0.10990819021397091, -0.9929951801302714, 0.6737495877140292, 0.29623218694663267, 0.35165024259546973, 0.4247902893913085, 0.10129389706846115, -0.9748497096341768, -0.20334247141617556, 0.985129316213762, -0.7736637873162373, 0.8859211253940154, -0.808231718152159, -0.8884564371876782, -0.13224967135811516, 0.39967060212604877, -0.6086782667041692, 0.43246956758097643, 0.0011514710304929565, 0.9558714543693512, 0.38962819185369524, -0.811064727302403, 0.09181667414286121, -0.07986437444248562, 0.4634491365668032, 0.8192784704899874, 0.2445550987058287, 0.9133423725999186, 0.9228395837314465, 0.8291228656244602, -0.4356121962581623, -0.10136303370411803, 0.837845524425672, -0.6854080374628142, -0.602711952293828, -0.37114849707842223, 0.29018911965469507, 0.5790083839331863, 0.7081224945304168, -0.2645850273193422, 0.19321657258513847, -0.8319101986340218, 0.7533259515543165, -0.7191653748839306, -0.4699780884726963, -0.6540800173276673, -0.5812689980633654,
                   0.9503472432990472, 0.9788033733415846, 0.5437384536977135, -0.8816335546273606, -0.7756464482416054, -0.6738531434490613, 0.6358569300642081, -0.012072443955215784, 0.8517659888279863, -0.1869264035395064, 0.031778843203837503, 0.2570127363561745, 0.7354138099316871, -0.38467088798128546, 0.5208605757744065, 0.1471108976090978, -0.05583277394075714, -0.9714693145563995, -0.4343786143761459, 0.41771606632301017, 0.9824557660708686, -0.38956794514818904, -0.6879262212800785, -0.22916587174280756, 0.6763739847803645, 0.526704094071031, -0.029789640445943233, 0.7245393578421753, -0.923419247515632, -0.34672898075813663, -0.5849485275851374, -0.7208723099728243, -0.36828377772570065, 0.60738232386471, 0.8684209204801205, -0.27519526968006214, -0.4479249569463519, 0.8112413814334158, 0.3438844572569393, 0.2777372106703082, -0.23821035898836795, 0.9361517822674592, -0.8279492978296068, -0.11831277775183491, -0.8712055563461132, 0.631783866154523, 0.37069938087579923,
                   -0.5246613261425481, -0.2346438211959334, -0.6856962264729649, 0.883968642450716, 0.3008173016875404, -0.8920285939776089, -0.03125117583407366, 0.7870970704521867, -0.5832060230754517, 0.21394876932444573, 0.41517220994584747, -0.5620165009817426, 0.4799548439159911, 0.25300771374015785, 0.294722847175209, -0.745908164597695, 0.07150878395239113, 0.5257425015120141, -0.8490356892178725, 0.3029945880070637, 0.37079669637913115, -0.49318506782477534, -0.586620426105519, -0.7828856635299783, -0.18404460591077143, -0.6107449709120483, -0.16585275100768726, 0.077987345073258, 0.5419805247742269, 0.3972652389195457, -0.014082646866572235, 0.8626404161038832, 0.8494234418822935, -0.26156171580592225, 0.797452759780684, 0.804436077799819, 0.7323476117474597, -0.5176848605075761, -0.214353473322896, -0.4069916479488753, 0.5820603299629334, -0.11927513207960572, 0.6365622986169861, 0.7226761962644856, -0.41647310029536033, -0.3136356578844739, -0.905211493898016,
                   0.4959870333537728, -0.9657356848404375, 0.8795902627635401, 0.7395358206705136, -0.8011907408538008, 0.8409026362907561, -0.12058718212684649, -0.8424507874354399, -0.2143421961989087, 0.7864925805167844, 0.7931949699387171, -0.7329543780680179, 0.04870206423117951, -0.18797141540922113, 0.5505779177314429, 0.10352954545431592, 0.6911720994944301, 0.5185470503821357, 0.8706676420813566, 0.24747892491927548, -0.01435656638156324, 0.3715507542757388, -0.07246547197331288, -0.7444648341766973, -0.230052329717781, -0.32260012129255333, -0.6924075053027752, -0.04902939546509377, -0.5817794963689014, -0.6982982203407253, -0.6065596107838094, -0.7641516650100433, 0.1218656756339378, 0.31262188926690726, 0.6622635123098639, -0.7935705163484639, -0.6750226200281879, -0.7425714120357048, -0.4470905448062801, -0.21497609948571506, 0.4810314000073175, 0.6184757340260678, -0.465732653143917, 0.8256201453363632, 0.4324184002622766, 0.1452168562939664, 0.798595522271093,
                   -0.2927273979242664, -0.33833919622819875, -0.7782633539674961, -0.39304160714612757, -0.20163724748243683, -0.5770813590122559, -0.7953260546898402, 0.7299487816105497, -0.14350670228745344, 0.3346556994460024, -0.24569644855981965, -0.34461261444951297, 0.42291379057907097, -0.6008957760890647, 0.5819331375613201, -0.7242925800986801, 0.5770176256806394, 0.6114296352017374, -0.25033781975813874, 0.6212468426623023, -0.964410059435173, 0.3312648652611052, -0.316058103356994, 0.8716875628149614, -0.4868714615219827, 0.41154764792140197, 0.4022636704785003, 0.35639575389663625, -0.4963026422711574, -0.285639030557028, 0.10269612576687592, -0.45456677590311245, 0.20141632860089498, 0.09074165483688135, 0.22888035020344333, -0.8634742787199561, -0.8541706769285693, 0.16118817250723771, -0.6015165153519173, -0.7407107878980517, 0.36003356767638417, -0.694135547678332, 0.1710021472668899, -0.4534688816039425, -0.7564052934892724, -0.8417385267853612, 0.5311976091763955,
                   0.3634592675790982, -0.7110878535957683, 0.593344761416633, 0.17733458104155297, 0.8391459930975185, 0.8088392407543776, 0.7109815738131975, 0.22878672491468666, -0.48938771979512086, 0.2950049247372435, 0.6377056479111909, -0.1945371260602735, 0.3684100582790544, 0.770422655468112, -0.8953539100987502, 0.08189785814017347, 0.8835987533626288, 0.5566139727388169, 0.7067088310661116, -0.11825723568632873, -0.8004738474518562, 0.24461407313744643, -0.49213203162613905, 0.6537803887917901, -0.9121600376642096, -0.6591316563191056, 0.22968778994500494, 0.7584817688279848, -0.5033279630790408, -0.9468321927516004, 0.6449633406915267, -0.6054894486295641, 0.5599527651737717, -0.011586059033954266, -0.004368518913843378, -0.7925271979007864, -0.20967143334527183, -0.8468574844735834, -0.7361280299673596, -0.3514001020014301, 0.8441719004511283, -0.4206225183361614, -0.05241828940781024, 0.4889432013367967, 0.7256870596393152, 0.1425261021791404, 0.6261723854581709,
                   0.5450500300404812, -0.41186689629245565, 0.6946676397858436, -0.9500934146831592, 0.6349806018446009, 0.39369836239576395, 0.5057283831090649, -0.2542461963324141, 0.05891612075138197, -0.39042140633267763, 0.1866755061998875, -0.7949878525468295, 0.4864996498902936, 0.8463256453429588, 0.763004936221829, 0.6027085557010405, 0.033441431048522485, 0.8855482451246972, 0.37343680755796616, -0.3039446824429277, -0.47718943333186337, -0.10243554343973349, -0.8121261919305196, -0.9409940580248752, 0.4518263094479593, 0.8156183721227122, 0.570318870504622, -0.8677799732070184, 0.02285317733341885, 0.8117911865388081, -0.7434122312024276, -0.29748872228932144, 0.7476927308707557, 0.6107511849922358, -0.6267057480004454, 0.3731194995228757, 0.5971584470142339, 0.9810036370619939, 0.4937251724434073, -0.6540878898775881, 0.4433079696703792, 0.15931671076840415, 0.6078638581084927, 0.4088870289614639, 0.6295997221685055, -0.8074846913560318, 0.8779609969672071,
                   0.4071141028301317, -0.09537021221157915, 0.2218503150568718, 0.3576919340321014, -0.3274823443243038, -0.3320029248023637, -0.4113333920717215, -0.5279186554190616, -0.10233093940035731, 0.5022347515359615, -0.6938918933945197, 0.9870154944072045, -0.2389768934473533, 0.30263595230688334, 0.9201150216090916, 0.44130361750650526, 0.24579781339053253, -0.03280634766804291, -0.9260175761194325, 0.5708907703651933, -0.9580366870152639, -0.42823583672623355, 0.1942667877250137, 0.12396456194326944, 0.6805704477200731, -0.7690449362493186, 0.6318405640720401, 0.6473388996551981, 0.12191580791390577, 0.43312824920183934, 0.32041296025402666, -0.681360665850234, 0.40380572515194024, -0.4616995411333624, 0.5558484322857875, -0.2945463997703379, -0.3465537242197809, -0.6593871652125434, -0.14686833229729812, -0.5482464719001632, -0.7237464978776753, -0.8775667237453604, 0.0038424054580146194, 0.32353330691963844, -0.5428572677917576, 0.46504473133376334, -0.5644801360504461,
                   -0.21024902410866209, 0.2446212905575651, -0.7687361761861227, 0.7653114016744542, 0.3713001390684154, 0.4352756090988801, -0.40177527351271247, 0.04381647634382113, -0.6401205161149366, 0.2360081923505879, -0.20604656304621294, 0.6010516894698137, -0.8097316275341111, -0.7463538525457949, -0.7207506813303193, 0.6349795787672854, -0.2563423713235722, 0.7463087031559525, -0.48322038360223574, 0.7240573665635768, 0.16632476855112932, 0.2681860143528989, -0.8178147721281397, -0.7012195069617808, -0.5929156860584481, -0.15085140960933585, -0.3026148401888038, 0.7732082230436454, -0.39141672968721863, -0.3271297367182264, 0.30415853845174534, 0.5159765671965557, 0.7316541675568085, -0.1480844057797044, -0.4535266745657711, -0.8686816196100586, 0.3485767009768117, -0.6544938480903613, -0.007078262263486623, -0.3758740934215927, -0.5897586407980142, -0.6745527988789377, -0.6939877047545113, 0.9391076448668692, 0.9738479004054368, 0.3679607858751772, -0.6470393300166548,
                   0.09260597013179539, 0.3719133532241996, 0.7339383431055178, -0.046989933133658734, 0.6566581173105726, 0.11000268197700347, 0.6973894619251555, -0.9307960308549934, -0.29160106464731017, 0.8997388310016035, -0.6083686830939992, 0.7779241593189983, 0.5239187279967878, -0.33471291042585927, -0.03497260414742387, 0.18739763074416804, 0.27420605721760793, 0.10565029920622693, -0.027149894348749815, 0.6267667403074093, 0.7718201045047683, -0.5482479919977947, 0.8791078559605057, 0.8902070627853911, -0.07954635863019655, 0.4264255735015099, -0.6980576340942131, -0.9040309018017443, -0.06443757363807157, 0.09706669749451091, -0.7915318194928704, 0.8081091418982327, -0.36724674073334795, -0.6951504667576256, 0.6496908602761358, 0.4690670647487831, 0.12163830185564994, -0.33895553046086535, 0.8234074300146681, 0.9041369573671039, 0.09637830400461911, -0.2798807808736934, -0.7372428336801902, -0.31572950780507925, 0.5801539615468283, -0.12905722496163485, 0.7703568995779615,
                   -0.5284605436830518, -0.540680172546194, 0.7315545783341291, -0.6472796788703625, 0.12197879307778359, -0.5659896878667134, 0.3896002812856596, -0.14307289205509033, -0.7165814912023951, -0.8845159892618062, -0.30032271164099544, -0.7287276766414743, -0.49869448768845404, -0.08349281009888831, 0.030473015056071695, 0.723892890835927, -0.566593484993249, -0.8904063724516995, 0.34872012797511887, 0.09813355438858795, 0.6746738057603372, -0.031067091244987965, 0.3336328784519216, -0.11932036587069117, -0.09355416340208511, -0.7346439241602143, -0.665369913496189, -0.6317494380496225, 0.6095461673561786, 0.5497014734902694, -0.5832096087177485, 0.6809130045365717, -0.5812661253319751, 0.6384607695383926, 0.48262629402060453, 0.16483835892531706, -0.6536498550848433, 0.5066003430589272, 0.5154040874493233, 0.7145947514586648, -0.43410461070462625, 0.3959677654501885, 0.3583769024340644, 0.3048299637591283, -0.7783370752521397, 0.36371098038233973, 0.12593992413798882,
                   -0.406533464684238, 0.8184448550423424, -0.2763492402758032, -0.6631440148212298, 0.843295916364643, -0.8542828042264732, -0.1254351798570177, -0.8336581949221631, -0.3938745561736978, 0.40360147547387926, 0.8628565525838037, 0.8329441235173347, -0.3766044428313593, -0.776709169492781, 0.6930968166018414, -0.9322226335645056, 0.21945221275482862, 0.6125191891700896, -0.12155625685500082, -0.14216579087024428, 0.09842353333799148, -0.4929571616670392, 0.013959813821892597, -0.7882799180964981, -0.09708375526202873, 0.5719941263711537, 0.25936285734596676, 0.5653059074676123, 0.1525132155316249, -0.8276419162713293, -0.4662387702082642, 0.6264308967699843, -0.74768184041394, -0.9802876297264316, 0.9534407155095637, 0.9519302370455853, 0.14655993787063792, 0.9027633887685929, 0.5847855501496304, -0.6430980772495025, -0.11046571740590294, 0.12209978176136094, 0.9379313766736559, -0.6247322461487006, -0.0840735253700422, -0.22502277055299658, 0.792764435926387,
                   -0.5776753863952762, 0.11482384896306308, 0.8738373823411933, 0.9273960340771781, 0.05412216857499619, 0.9464931746055594, 0.4326559580732767, 0.11025724065278508, -0.2248555980235607, 0.7882626901153482, -0.8818959906384716, -0.67952516031319, -0.7850878443168654, 0.4556154497827112, 0.8378986569739586, 0.5772043223952548, 0.17576539467115593, -0.5831350004308469, -0.27339108127909073, -0.012887608597787592, 0.048223971798973864, 0.604550845544154, -0.6104266379631593, 0.6602157821527235, -0.6030336531493612, 0.9239565120219764, 0.9784772263372521, -0.002199766646259871, -0.8603930877426629, 0.5689185231541418, 0.6975268954431282, -0.47784197092047576, -0.3279083833331624, 0.07736070820918983, -0.32294989637600735, -0.10360170335432706, -0.461325106151137, -0.3062469013648199, -0.19825861115059462, -0.9263992705385404, -0.19892148547048127, -0.9470387251203023, 0.04366260334381744, -0.3819022780119494, 0.1305758498942342, 0.4597163393256476, -0.8745009639890844,
                   0.1576864952064383, -0.17161544518495186, 0.11020307320460399, -0.07253829836192449, -0.34591183123811886, -0.5261081766676556, 0.4870277830666292, 0.9516186478461788, -0.2800139673104518, 0.7068762775303492, -0.14390615341691038, 0.37032652363177077, -0.4355084397775033, 0.7690323529337182, -0.6963215779439287, 0.7108258320565237, 0.3149012716267563, 0.8617424771872311, 0.07426959171367664, -0.9340460506401602, 0.4411248180428573, -0.8788165130198149, 0.10448835240581955, 0.871319193108828, -0.19982165454908785, 0.06849471339517232, 0.5956859282032068, 0.7521715823157733, 0.4216543256631209, 0.7618749152139945, -0.7724277218800937, -0.281404358706258, -0.4289059395486643, -0.4408831771593402, 0.1874977200294614, 0.0755407856299688, 0.8567045097355079, 0.3932277769147874, 0.5747742622679695, -0.18540304665205243, 0.1858773228193118, -0.22957349912373037, -0.06330998571335411, 0.9673122003125492, -0.7685170933982524, -0.6810543789935473, 0.7026350031245607,
                   -0.5747599077925538, -0.26607150969148075, -0.351288657060161, 0.10502214819831446, -0.9302035345308095, -0.556515554181158, -0.4738888792442315, 0.8104536986004767, 0.7053935326885501, 0.7983990862281953, -0.0752981969206925, -0.06815207408581414, 0.25221230160496044, 0.5378372729669534, -0.6181492045525636, 0.35200722539390994, 0.9303629500297643, 0.36469446994903243, -0.7270565848999033, 0.36137438662767885, 0.7550748706398, -0.49814726055298064, -0.891069434885174, -0.5025110504927095, -0.43705230147263596, -0.47353835168013125, 0.7696311657442783, 0.170921514187653, -0.9978137689760544, 0.22034183679842045, 0.7043444056122032, 0.17222987965645586, 0.44513127328508184, 0.3406251947041985, 0.5629680226916893, 0.4767452458284125, -0.380708647080112, -0.8067389838313124, -0.5350415020259189, -0.10123571349707605, 0.9034680614792221, -0.3169667634828546, -0.829889359097298, -0.9060134674771911, 0.69846131812407, -0.882750761668099, 0.7842738695043652,
                   -0.8311968898164841, 0.05722884600271838, 0.45627652267540975, 0.3147641457846744, 0.7273091513092105, -0.15273323263643435, -0.06738055179249236, 0.384369458604938, -0.4283237193096976, -0.8816323191861666, -0.9987572148394275, 0.719489359428497, -0.53121185707134, -0.9381962886207371, -0.5626335466133401, 0.6470883474666131, 0.40580797392969803, 0.9723724369402464, 0.0639688249629804, 0.7667837276473304, 0.5796117390357638, 0.9626344257040873, 0.0697541174957399, 0.20852979498667734, -2.1355044568238668E-4, 0.8290616533653892, -0.8306468226434753, 0.4429998654069134, -0.9678403195002423, -0.5672939031092523, 0.9610438032006872, 0.48090938190227983, 0.4695905711466277, 0.2323903046721596, 0.7532913546378672, -0.5208370067193002, 0.7890193456466557, 0.19630918013832677, 0.8599272290015361, -0.7341502634088879, -0.8614630950766569, -0.7816732528581558, 0.9047968044024859, -0.8972493390214422, -0.7044965308725661, -0.6218615510034231, 0.030225957742843867,
                   0.12316449625534576, 0.3465084097548372, -0.0901365895201065, 0.06966926856670752, -0.6855034478770652, 0.16223112358290215, 0.5815191590332465, -0.2765804365976767, 0.6591979272367876, -0.12497199627909072, -0.6617760411976157, -0.432726154879123, 0.10428151300147981, -0.4856655516108106, -0.4751393048043169, 0.32756596068206556, -0.6012172451289093, 0.7147396144214195, -0.22067100368351422, 0.27199281035185563, -0.479826355826489, -0.016145876142162008, -0.22459986034679713, -0.2058246073782981, -0.5951215372837906, -0.5981413875448511, -0.6054167351538153, 0.3109127435607819, 0.004680880781382957, 0.8218858516570204, -0.38070143801485234, 0.4446013372404687, -0.6900468966987461, 0.6501546376167449, 0.4650466321540081, -0.8706905260497184, 0.9122772298918866, -0.8929599499336724, 0.607152185379054, 0.757510200956562, 0.9231704480972602, 0.5050291493587842, -0.9919488821445834, -0.22450650398481353, -0.3777335069499115, -0.47322212580216916, 0.8139812456050186,
                   -0.7610070743197128, -0.922554615254422, 0.7462501299235293, 0.06532698603350817, -0.24342185040646336, 0.1610452235160318, 0.7081853376621914, 0.35275883054113466, -0.023119548164799886, -0.1195699402023529, -0.23409329617491892, -0.5482999231550989, -0.6832668039892398, 0.7478275989772891, 0.5747486309113599, -0.8194578555129455, -0.019173969906706168, 0.03349321306586872, -0.3518543300471668, -0.7465772345255004, 0.48948619078193456, 0.177894629760047, -0.2943069763157531, -0.2637309934188068, 0.9097163389795888, 0.38275503792656806, -0.32274697709868705, 0.8652013614799481, 0.7756342694038616, -0.5425695421624814, 0.17444744840910742, -0.550718285365756, 0.2879744756379865, -0.6337597926101499, -0.9993573464875882, -0.14501764821536334, -0.20167129800162154, -0.3959921462660265, -0.1572022072610466, 0.6576410985956824, -0.18768326778974131, 0.5212118622133428, -0.3266853057125201, -0.5544403105875491, -0.21310500274859057, -0.9732141694640377,
                   0.5872910427772184, -0.02470706033058656, 0.8505265082420985, 0.7157481130898609, -0.18965383840871497, -0.09793391743390134, -0.7590494909221741, 0.4004815566148252, -0.5655470077543954, -0.35285031639663433, -0.2661933989841492, 0.26862511978468717, -0.9260238036074615, -0.3855894095496679, -0.3736172116073795, -0.5528207670476759, 0.0813817932617873, -0.0688505993482309, -0.05572619463086781, 0.6215675094909627, -0.778151191217274, 0.16372472048267706, -0.5452495006919904, 0.5400382805862605, 0.7782802874963237, 0.04635499524072695, 0.7991571394047654, -0.2720701752501504, 0.9429750256865974, 0.16188027206614475, 0.5976697696079387, 0.5429636083760456, -0.9505112175711667, -0.27384421825140604, 0.7332627782850525, -0.17812879138340576, 0.7079774215118559, -0.8468594595787515, 0.18774846537186374, 0.8003001724946384, 0.22958197143201864, -0.0843656450740411, -0.21873172088548842, 0.7320856316111681, -0.13982042809866613, -0.632991978656327, -0.04236914273207404,
                   -0.3093855064613178, -0.7089087697300835, -0.34783929254602297, 0.36140812112165444, 0.40415015125667053, 0.674102143736494, -0.3886690394581782, -0.6312852339439587, 0.6586255661806824, -0.059584234778469414, 0.6682766572762984, -0.03788636411774826, 0.7955541494941041, 0.607929487758889, 0.0947310390543954, -0.6188954362081411, -0.8606523068554022, -0.5415155138338292, -0.24406756710479027, -0.16683069290881836, -0.19252851256625347, 0.47650786581313254, -0.5325851188710635, 0.2163617755453977, 0.2885077827368199, 0.2627317371149387, -0.5688610885990903, 0.2829481420962727, 0.41733841842359487, 0.7714386277350498, -0.13056816002573002, 0.964616017375387, -0.06670055200118252, -0.3330515255973776, -0.06190334293004107, -0.31671087711863577, 0.6762996012264633, 0.8548285576733266, 0.275440822273576, -0.6124012544699833, -0.34246338937380805, -0.1844276620511729, 0.13186565763318048, 0.783577993299359, 0.7355848903734354, 0.2532600939207197, 0.9250480257257403,
                   0.7691625973186074, -0.5647308883668032, 0.2689785783316332, -0.7213628704335362, 0.4355952992133194, -0.43412079921823254, -0.22145640266033917, 0.12777860382145856, -0.8351516931134115, -0.27307766844711345, 0.21633437611177575, 0.3126705891501773, -0.6693068679385157, 0.5337480790014622, 0.7018143873594782, -0.768255227128328, 0.9020621998557106, -0.5479756794702482, -0.20742075369246615, -0.41104147077388253, 0.8508638219520004, -0.5008186771179086, 0.7075370633567768, 0.04507249632629806, 0.6824262053583949, 0.7525819195543628, 0.47609381113858773, 0.670680840415256, 0.6352623139253049, -0.40168249445849336, -0.14602045272586106, -0.5008503660701336, -0.862136297647466, -0.03883120794840722, -0.9563802617528439, 0.1905120766844317, 0.9637456076004118, -0.6245885376940488, 0.2667191715735895, 0.8427575864438939, -0.8080171443764159, 0.6877881732752866, 0.05899293379957804, 0.3248436593217674, 0.5589907405232921, -0.5718925674095585, -0.20825605126543767,
                   -0.8438501847790736, -0.9216796737734134, -0.5604277714591726, 0.37225283238013485, 0.9199834784590284, 0.9087549785556637, -0.7699195034657833, 0.3347104374219838, -0.5258616714130693, 0.3670080399550819, 0.6873810214635596, -0.3943098735999817, 0.8093957759622401, 0.7098397383258668, 0.29510865563341815, 0.11292186307484742, 0.3072522146715353, -0.14857489117198863, 0.7933632663565848, -0.8966044465841803, 0.3610433664974366, 0.5353211749232427, 0.3466379140637328, -0.4534118684416857, -0.14997046383977541, 0.6463976338850517, 0.7135415530315035, 0.3208720095746431, -0.20610496411063184, 0.32660977942155367, -0.34309673872187063, -0.9840690385039585, 0.6655056980325069, 0.3495090356244446, -0.11051549872231803, 0.13127283109763344, 0.302660307690495, 0.8119019045117033, -0.7803324003519665, -0.7893043899870043, -0.865039695885264, 0.6170830294261547, 0.24312830974620137, -0.975171771103041, -0.47018544881145785, 0.45258504128865495, 0.9206415922388773,
                   -0.701366971546934, 0.46198586683134857, -0.9935980928044204, -0.31118979878245456, 0.6780232397786858, -0.5760237209567634, 0.01915978037396404, 0.36964628077327677, 0.6395136694819941, 0.519939745233263, 0.43572303233219256, -0.9234236667126794, -0.13091149122567725, -0.976300258109045, -0.6017589000173085, -0.441418178598203, 0.2712364394132234, 0.3318426309837219, 0.11365518107821293, 0.98082654233861, 0.21636763858251018, -0.7942551679793297, -0.2677990544593649, -0.990735915973564, 0.8945946281147987, -0.5282141634511672, 0.2894582482992971, -0.7516414654612638, -0.2838774750282429, 0.9368872395471595, 0.4930150861873539, -0.424807969336892, 0.583261669584987, -0.8230901440930996, -0.32417215355578066, 0.7198740854441366, 0.5563400713989917, -0.9316688138907603, -0.40812347430469376, 0.5613080805242576, -0.054440883102290005, 0.45901655633193283, 0.8957280624157895, -0.9609904211709446, 0.18310204442153855, 0.21178685995513358, -0.8738619789930955,
                   -0.6741853991747002, 0.9645692226694669, -0.9951221261032288, -0.1951655177547329, 0.9629533122113121, -0.9039188199647741, -0.2637243029934051, -0.13485379056277869, 0.028107911432154742, 0.2431550004871572, -0.8456967032545859, -0.9588500712613477, 0.08871245103381198, -0.6502743845593817, -0.5752743533349447, 0.07652985304067106, -0.3131064765011422, -0.1195503034875176, -0.254971631969942, 0.6533558415509793, -0.590908134740038, 0.23669423482943808, -0.1705810737985045, 0.7945046266835916, 0.21254422372993131, -0.29362111057095475, -0.886512777280676, -0.8405387170790051, 0.6124471789986945, 0.1301057934547165, -0.13464519517534979, 0.13175706262918063, -0.4222361845543341, -0.8992231947788831, -0.9594685055186836, 0.8873121855979622, 0.540275056678805, -0.038836500394100915, 0.6805899361704137, -0.26352990692212686, -0.2093806947232686, -0.8763395625207764, -0.06884230212284281, -0.05147608602225806, -0.28329752475697756, 0.30871882772328485,
                   -0.6573812047917844, 0.9960526395568383, -0.16599656230456383, 0.18322271699481352, 0.017306092620889224, -0.3026013328048789, 0.36393840821286516, 0.5773353034998672, 0.3798566339991052, -0.9108641156832173, 0.23559055992514333, 0.9681087236705039, -0.12720655193422048, -0.9736089812930386, -0.18225052262038943, -0.7595830866265367, -0.28271277082074864, -0.5126014316783967, 0.4323480919641991, 0.870535811979148, 0.029441775899764044, -0.650316297741578, -0.2429771361380224, 0.6249613875188844, 0.04533868496978655, 0.00878402549488988, 0.5936662117368492, 0.6268215088877647, -0.01653379083071993, 0.6264363576215455, 0.16147831740791552, 0.21450284483025994, -0.8389511953725235, 0.02367095508210948, 0.06716813441562741, 0.4018809644504022, -0.6188160981008644, -0.7247305198910914, -0.5884075784293552, 0.9468049281717756, -0.6794167803686584, 0.9155614613563352, -0.8887363891711486, -0.955743393008071, 0.6731068443055985, -0.668453608440243, 0.4447969089328596,
                   -0.5171326462349839, 0.14216846007196615, 0.3978473641485598, -0.9214919014964809, 0.738242543046036, -0.9745190701936486, 0.6990152234815858, -0.6179857116826282, -0.0761910449567107, 0.09850263399797576, -0.21178104981961954, 0.7138300663444677, -0.22033380994332474, -0.22166608449524317, 0.8473743707945152, -0.4614074963477084, 0.9521885802191923, 0.13770621211080702, -0.15772296543035358, 0.7316889014012433, 0.7018896075004137, 0.2390943590323349, 0.8073226170998804, 0.23370175231857915, 0.8743273560294771, -0.5281248465714246, -0.7520876912282364, -0.663773838154385, -0.5214067918459822, -0.8991552156254905, 0.8146575836354806, 0.31225558497941464, -0.45980559293962764, 0.07229921103687098, 0.3820132764609183, -0.15152778752249652, 0.029903853228754107, -0.7257291533608614, -0.5450157756204523, 0.7934190421901892, 0.6174183063210363, 0.9104077578554184, -0.5689338871360152, 0.10467138921360442, 0.7584305942410106, 0.4223693162592128, 0.6605964979252372,
                   -0.4402337968916945, 0.19598728524145903, -0.5164942519789901, 0.4772085090105991, -0.21021243503628773, 0.34176975241633256, -0.013022587451545276, -0.5889060167088114, -0.49885067954369644, 0.20020025524506724, 0.6643618408912668, -0.3486999831517019, 0.3702528014130251, 0.5080426431999656, -0.003814095230145753, -0.5107032061926298, 0.027342975039946094, -0.4155687332590037, -0.5862878473641773, -0.04542819394760289, 0.7900219872955527, 0.021027591967516912, -0.5583014744629688, -0.8559796374733342, 0.3635245448062667, 0.7892702474478295, 0.39474102166955416, -0.1495824710843665, 0.5939954473223414, 0.6014494143983724, 0.5152001296096238, -0.9887264125881352, -0.4573454739486533, -0.565398233274742, 0.028601999574711545, -0.7721110656958468, 0.6428719902159572, 0.010988678157738185, -0.5072039018008896, -0.854558075350323, 0.508818463698818, -0.2738149000408663, -0.969958398844013, 0.8384573161172577, 0.0834957658229325, 0.025572980576787474,
                   -0.44341423542703295, 0.5120185756147242, 0.24559344711357078, 0.5522977420077231, -0.037812271125648333, -0.19239519023348461, 0.7387580437499524, -0.4754867704709649, -0.6575681604670676, 0.7934457581664833, 0.3435015393192766, -0.25299526521619997, 0.8495657970420121, 0.32280675622210975, 0.5811176262221902, -0.5779532401389316, -0.49701134274123904, 0.8596291929009732, 0.9206125745435598, 0.49661653912567316, 0.38840499256496575, -0.9005314638055839, 0.4538017377664674, 0.6678506749071107, 0.07505200985478133, 0.7546877205343487, -0.6896412037324429, 0.3658980693515277, 0.79329690785396, -0.27362138446067785, 0.7037502550514592, -0.9569130376765209, -0.05970754768251174, 0.5515006893018821, -0.5693759163719154, -0.07940913930904991, 0.8192694148240096, 0.49224104054251216, 0.8903653132027194, 0.9785425502549996, 0.16843501017425155, -0.07438409228199938, 0.24816989562244274, 0.42281159893563314, -0.00166235692575456, 0.7948493774909686, 0.23412937544215895,
                   -0.8785733571690171, -0.46805236655262705, 0.8000803396448644, 0.2524607554698859, 0.575888699381061, -0.37040424926788296, -0.26553515655140814, 0.9691537576662719, 0.8773877112412787, 0.98130101626919, -0.0872438268976179, -0.6470482824958264, 0.7952003283139455, 0.5619103228735998, 0.9929608461616106, -0.7362796777162874, -0.3456875057088238, 0.3698622301105796, 0.6576845440752737, 0.11390005780593881, -0.018424520388589105, 0.3082796373463912, 0.6002968439234568, 0.872879122688059, -0.9655297361458233, -0.7825155229758178, 0.05206015769768957, 0.740763854466711, 0.2738212165846321, 0.5309199756344576, 0.6898837284236159, -0.8128361962844641, 0.6015637107436071, -0.19814421660627346, -0.8196219103068043, 0.7322503990479088, -0.24306142919856533, 0.43614201159685484, 0.8444031794741276, -0.15851205820511582, 0.6492499001371959, -0.6333310743103864, -0.032683706427307335, 0.6120759875395443, -0.3974109393688485, 0.11807990113182454, -0.8663560454025274,
                   0.5482843536565676, -0.5023064389535883, 0.735675185204723, -0.36908399456108376, -0.3454996546936575, 0.6183748427868836, 0.3903354648491566, -0.5540832571943468, -0.6810495768019489, -0.3552853758549148, -0.8783156724212715, 0.6676224245106708, 0.017698143296850377, 0.6824950407537484, 0.14459953545885162, 0.21461142472271155, 0.737661043238671, -0.5219063715452172, -0.5804265512116289, 0.7207481443375103, 0.19902968281150413, -0.9596598840111477, 0.7753138305345595, 0.14145739980501748, 0.4516127731037911, -0.22440375935080437, 0.7723669791038448, -0.8160780338622444, -0.7737265405750302, -0.06041232485224701, -0.2905650097738548, -0.9819485662408198, 0.5417607938706084, 0.7192804031668898, -0.30054925999151494, -0.10891414764737539, -0.5011164970774085, 0.5803614153103827, -0.4196361723445232, 0.24721411904558832, 0.9381116214653666, -0.3493519447507478, -0.4240180865517569, -0.25539831672610935, 0.6933332352707842, 0.8570162577344864, 0.08510210212204172,
                   0.16573816477105408, 0.5436341280198209, 0.29320412352911585, 0.752558101530195, -0.3669684863890046, -0.3533912118153104, 0.8681117446385762, 0.06369922210255119, -0.2299095417752457, 0.2308510419151708, 0.6689851882228948, 0.10420761915863253, 0.4901525299190397, -0.3680853302976552, 0.6711367644436175, -0.15994823068357156, 0.9466311888667482, 0.37517361103175006, -0.5643794337448065, -0.702236592033556, -0.1532295793452636, 0.9406530971021099, -0.7414527521582277, 0.3631175802890858, -0.19208830609879324, 0.0890421807779831, 0.8704876362234402, -0.43331643660348496, 0.7268927025084579, -0.8152589670379833, 0.2316804216620607, 0.18760803664410663, 0.23661412252853276, -0.17906547794929484, -0.7714748848930191, -0.7176198072613917, -0.7601748663658263, 0.2333260479826922, 0.14910692017455562, -0.16678034019369248, -0.7613049302531727, -0.516694728352036, -0.2997346781333572, -0.8888800016921965, -0.28024666142929955, -0.8020356231797392, 0.02598456432038798,
                   -0.4385312343450767, 0.20729596387250004, 0.7735181563736304, -0.3033506982480221, 0.04555843768040502, 0.764372402947028, -0.619676621664351, 0.9609748253871189, -0.8908090844411776, 0.698982284144668, 0.8518341994421605, 0.6871860997819328, 0.09997024422579237, 0.024126667073271824, -0.4035293212780704, -0.6327138736565385, 0.4177681593737752, -0.176125205085504, -0.9594822556643914, -0.6819801663012592, -0.0417614378185287, 0.09362252121349934, 0.7918131476892807, -0.2155952261645102, 0.8971459111418079, 0.8512343810458023, 0.7081234798200746, -0.9635340408310908, 0.36366564342257446, 0.35677650614778833, -0.6392525949760848, -0.2529224004474857, -0.7362972638052683, -0.8679359266405651, -0.9969734454065742, -0.051419756579353715, 0.5396672260626192, 0.13726129830234468, -0.9149150668052533, -0.3118924639015672, 0.49103499471000656, -0.05971058338593016, -0.3997419735016805, -0.8636779959956238, -0.569587009306193, -0.9941882808577363, 0.10531395526014431,
                   -0.6484744942778373, 0.9127190392757394, -0.6659185903941347, 0.12570894176642256, 0.1608749947552337, 0.05047618297914869, 0.5959543166812447, -0.16123901016422804, 0.6927006070021839, -0.9838865073417329, -0.3611175602772754, 0.42176488689650515, 0.9786876893592815, -0.26744581254290845, 0.7797214816111029, -0.7412859513392338, -0.2580486629111347, -0.033735890385971246, -0.34630009652839955, 0.31843348906843727, -0.0032039683220379533, -0.43767681402558267, -0.4450859815655781, 0.4485703438197661, 0.4082203263379254, 0.3989735552240772, 0.18222795021575466, 0.07402803389911172, 0.49399068635458243, 0.7894888197185372, 0.4954736854461539, -0.15492894869666451, -0.3006250197880078, 0.5220987675057112, -0.009638272283980509, -0.8505312325285115, 0.13439181415386603, 0.5040957647475326, 0.11652892359293432, 0.08468520820274694, 0.3356966691963843, -0.9358616589122921, -0.20833500115335046, -0.5197347250397959, 0.7999931820409494, -0.6328411894586745,
                   0.0056755711779823415, -0.13279427764465557, -0.7668792128997244, -0.8908273555928836, 0.3623546885159452, -0.08975827369764389, 0.4687389155729187, 0.6846772636875573, -0.9279443198677779, -0.24938020474612888, 0.8327904227974914, -0.6427812951172582, -0.6126353840297716, 0.9510679789055572, -0.6353053112918905, 0.6140960261847821, -0.2352033072445021, 0.2913161558906898, 0.7534858614699687, -0.1509373509709111, 0.13905064530800382, -0.4683414188383923, -0.8931237437250961, -0.20772460600603027, 0.6123732750846502, -0.5563081615160379, -0.2565072054814148, -0.305367508074889, -0.4099000286811165, -0.09849713952702532, 0.08329758614125038, -0.8259680432545511, -0.06554526365542435, 0.6332207080416155, 0.40847114127999706, 0.8439730646237371, 0.06220708639849426, -0.6480121435616324, 0.7768014516231585, -0.6296864765992047, 0.07780010712162788, -0.5829534289817089, -0.30811217634980825, 0.4398862464455515, -0.9879212566774063, -0.08022311942675575,
                   -0.7307654227393849, -0.7249298879499144, 0.027763641802703853, -0.8538369696823775, 0.5845749050150764, 0.7912987526785942, 0.05633068314902956, -0.2636935682129866, -0.34946993561127226, 0.26988406275447807, 0.746973250053047, 0.5432429665985168, -0.9600231630798433, 0.5064196164475367, 0.8368382195133537, 0.1874638764637402, 0.5592594131499604, 0.7954209986901499, 0.0927441609401165, 0.610179730631986, 0.5716484483970514, 0.8247506434543237, -0.1718962682912144, -0.45249225680038907, -0.41878296871196485, 0.578764828614005, 0.6647904287846833, 0.7794028541613929, -0.49043399423142375, -0.31298830476853134, 0.6882028746942754, 0.25129448645659536, -0.2674006473457249, 0.3056992460640835, 0.43479707608481943, 0.7230306769370425, 0.39181492990226263, 0.5820435878717789, -0.7451272478635431, -0.531127399389911, 0.7543409938309633, -0.9730025270975537, 0.5094144544138193, 0.7339222652443125, -0.5317844102367861, -0.29024928480456147, 0.7883041437513008,
                   -0.8534920166465443, 0.44417089136635735, 0.08778376059591309, 0.8782387958294551, 0.09269148153224815, 0.7924500588815857, -0.5008530865248815, -0.43780005473350325, 0.17820236871040906, -0.2470494052938057, 0.8690285607552015, -0.5016601348566503, -0.665254672857926, -0.12074742021106122, -0.9794345005116869, -0.5953367615262861, -0.128233014817708, -0.12814277481734626, 0.13023985014772066, 0.5668995514774195, 0.8314089558065536, 0.4032256544632755, 0.3310975205582112, 0.2904767473604215, 0.37351503251937523, 0.036973738081606644, -0.7575865693245991, 0.3335227117875228, 0.5992007167160278, -0.0841549130177055, 0.2923197592999105, 0.5863360994126179, 0.21563431307937053, -0.7221511657636526, 0.10030425739914017, -0.6962159697245405, 0.7513184774326489, -0.9427845059066791, -0.10062476466773895, -0.9493996978929589, -0.34969676126675253, -0.1634108567325243, -0.996917982567403, 0.1855721616659738, 0.5054006301521257, -0.760860255990101, -0.3395310761361199,
                   -0.8108792639028801, 0.9346126072639933, 0.48404615902757353, 0.2924324018393045, -0.5702054800614442, 0.1616242981932019, 0.27606400305627754, 0.0658408224660112, -0.9312879099875999, -0.7696447483686815, 0.45966035565525165, 0.03716769568718892, 0.7170049192638468, 0.9652021941620599, -0.05953858166390047, 0.6852792498535505, 0.18982735793957284, 0.2827465669955549, -0.7384749308946621, 0.5534082923072365, -0.7584835398410343, 0.2673423007007705, -0.08078309419193097, -0.06676353671657553, -0.48000073347761063, -0.4578617874111581, 0.013232498053976194, -0.9506017944325706, 0.1445622699429152, -0.10126872607598658, -0.8524247774212783, 0.040609612502092274, -0.7103889574898625, -0.10721276663236834, 0.6340646027605004, 0.1840168269586271, 0.9480667140238053, 0.1696050997032592, -0.002785013489834842, 0.27224883372929765, -0.34844151463938844, -0.34725616269223836, -0.46090890812791385, 0.8280494499605504, 0.3405893443723407, 0.4104690800943718,
                   -0.7049885969017997, -0.0707442152680966, -0.1936335207553359, -0.9910808356947101, -0.9042627689504439, -0.18272108720733127, 0.16700421661447185, -0.5324703183099813, 0.6490497815734078, -0.8248655010192583, -0.30567804623823847, -0.44860310650804514, 0.5633838571154353, -0.013538566638884308, -0.026522934384922925, 0.06690096394278733, -0.9415475041275203, 0.22485715958973396, 0.8561857571613736, 0.07105001002729261, 0.7425619590736086, 0.27324840636114844, -0.040697043915586706, -0.5383523724979329, -0.7164177974107218, -0.06409212197572733, -0.4359535502425813, 0.2845259374263396, -0.9396952985600089, 0.181117042631173, 0.6871665255501695, -0.7242721374534054, -0.046816937417927695, -0.6596340691112919, -0.1761218911984297, 0.6506624740720464, 0.31821926613472695, -0.6096276491252413, 0.403387838558217, -0.041102216164012884, -0.11497655110131855, 0.15709194211128108, 0.3513644819503854, 0.9423965329772719, 0.8298403899975266, -0.1937170604846934,
                   -0.6802841050609163, -0.23335965065890996, -0.4033981868098213, 0.8230788661676987, 0.12255263857450993, 0.738228972811672, 0.1738293003800051, -0.16658690590368996, 0.7082326470904705, 0.6791813985426578, -0.5175215070796972, -0.9169691724719795, -0.39910200589107503, -0.5026928532706769, -0.710420536059367, -0.834786555920896, -0.8390702073697418, 0.1736548578920043, 0.6347963385352857, 0.6762315378575996, -0.7577366350458339, 0.38218856132052226, 0.14179203153229825, 0.8779333450032913, 0.8655208248412927, 0.2634799432382373, 0.7835519147211267, 0.15976574848810032, -0.6205652215063282, 0.493347436136377, 0.6805839291276934, 0.11791647029787788, -0.7509488115923184, -0.2809872207552573, 0.3331804082327603, 0.4071551765672716, 0.08771647221977075, 0.6794867067434385, -0.6763099081354662, -0.8286566503440864, -0.1591314688744272, -0.11646495791788491, -0.027983482037214458, -0.17852738427675985, 0.13316716578353782, 0.5877189455882075, -0.31267659789656643,
                   -0.7945178454544226, 0.7065158543846626, 0.9986417622981394, -0.16880635982965586, 0.8019956547508409, 0.6050407105132791, -0.882322970004771, 0.7922541415048254, -0.5294890537701429, -0.6958897465420195, 0.453091356088015, 0.6695442354854226, -0.49653713421043877, -0.6660067252773192, 0.9436186127238888, 0.4716715792445998, -0.7490260465335308, -0.885230858574632, 0.8691797685456888, 0.0610636161033542, -0.5730045200751384, 0.6976195584057754, 0.2685860141691754, 0.20739059415961525, 0.35593373925449834, -0.10135289029580719, 0.44070172261367935, -0.938500451953513, -0.1182539145208934, -0.08511919341263008, -0.525315615362538, 0.12096528239050142, 0.4368824049446851, -0.07185448727219068, 0.43656130003897053, -0.6693693927376392, 0.8563378586621864, -0.5690134662191992, 0.833540217718566, -0.6537277752045116, -0.07752590658880054, -0.9501426513999736, -0.7159053754650644, -0.8132045986867009, 0.3391558171445779, -0.5057066900330547, -0.0589964152742104,
                   -0.48987375512880504, -0.14085890976436266, -0.07318971677469066, -0.8538334096651923, -0.4297650155208468, 0.626553591321767, -0.4732175539594452, 0.0401554410682925, -0.45250933909814317, 0.4466564551533323, -0.6874304596232115, -0.5938952509366322, -0.608188155373198, 0.6580373379432312, -0.6103209798330018, 0.8966836874518909, 0.5778449673810524, 0.5577052591953009, -0.29123582424593963, -0.025467239951639797, 0.24366664227873458, -0.0825098730378706, 0.14577558801276713, 0.3906892026621809, 0.0692719345262498, -0.9694727743345366, 0.25759823397542925, -0.16783466019342863, 0.9483767209842495, -0.060501678029230455, 0.4200219901127098, -0.5110142754588247, -0.9831725665407403, -0.9763069558984445, 0.22091011284148543, 0.722628024796921, 0.923177345908182, -0.04015649140457622, 0.9599443460175978, -0.2380450699781942, 0.36175427551834316, -0.7779229191464359, 0.897342065948338, 0.40936351926272074, -0.16419056357904993, -0.2507783746739949, 0.49745594488602474,
                   -0.8909800797207783, -0.6218958667657204, 0.18308891934087912, 0.32918264644675843, 0.4106878081450025, -0.4714445202382702, -0.8307476107818168, 0.51362191517018, -0.14320074837521735, -0.746671630499854, 0.6122259320813477, -0.3495555746430925, -0.37294214840744444, 0.6870266017970104, 0.48984098386783836, -0.9920594232254076, -0.1059958629436093, -0.4105982916433355, 0.8570099583033599, 0.4277397723337364, -0.22538606325755461, -0.7697368773037332, 0.006954938182655246, -0.12629594130803645, 0.007141421027055506, 0.05104024447511857, -0.5916298448763015, 0.8455166299613852, -0.9609412731859297, -0.10128921621454112, -1.2122570163408675E-4, 0.98100026593132, -0.9189154201836947, -0.11242306500184296, 0.6985344255500259, -0.9127824685923358, 0.8639359192777556, -0.9833641176290724, 0.5670080567634221, 0.5689191512716352, -0.6714245158232053, 0.5550566519657694, 0.9844925194708705, 0.7639235391972046, -0.7126292531439591, 0.9055411580676909, -0.6207532524570523,
                   0.04852536075498515, 0.21271916173955274, -0.834886552751569, -0.3939072510440209, -0.25795339174589915, 0.7413348903919477, 0.9251450862794204, -0.10993096297024119, -0.1818395978325107, 0.2265346087746143, 0.9871927787669599, -0.5921653459293166, 0.22163167844360276, 0.7274982608598028, 0.6184872419279708, -0.9586911447412751, -0.807364086316221, 0.1936232480405431, -0.26931941003814, 0.6134212741606841, 0.41683204446716293, 0.28373014504709837, 0.9377223443762561, 0.003739798134180461, 0.04788131603369261, 0.25401845484861996, 0.8335026618439447, 0.9421317663866491, 0.6901199192880074, -0.3446874254856809, 0.027164737671513617, -0.008493189778481236, 0.28260806611687794, -0.800722028407377, -0.516244630370202, -0.743921571203616, -0.5082051402269758, -0.9150721985863965, 0.5401470048375185, -0.3105262254770167, -0.3014471433491601, -0.6791175906474451, -0.399067946944591, 0.5274099873010623, 0.9563526114780665, -0.8645173039303098, -0.03374608100017973,
                   -0.6674520217402391, -0.599137349669632, 0.4360011352826032, 0.5033060876680522, 0.41452316238418785, 0.5072766091287964, -0.5401673795437316, -0.05450443581697795, 0.9760449941327864, 0.3201835801595032, -0.7453101678235461, 0.4880336591956955, 0.23634635676421545, -0.6781848185375214, -0.5685325714415908, -0.9540193055462833, 0.9961559742644801, 0.9580166193351298, 0.7525802987633488, -0.949460779258523, -0.3662438165615207, -0.9926937947554646, 0.5502215070042691, -0.01834582085073344, -0.6404201020839675, 0.9569022162561329, 0.7131265834129077, -0.03025495952171431, 0.7071911372653528, 0.3406123912357224, -0.9629290013049427, 0.746660034999713, 0.701383619548642, -0.4030032764686009, -0.8965311904363469, -0.014291480054434702, -0.08047652776100422, 0.42222937905740654, -0.46791968346844315, -0.3571068372738968, 0.7501515579687454, 0.8206026228455026, 0.44797802559403377, 0.29622891414079167, 0.5532526488404574, -0.9033710852400212, -0.6904394911790963,
                   0.6773018700678108, 0.13948839466629126, 0.8862157172820655, -0.13645604100847075, 0.5989863735485221, -0.9889174985957405, -0.009599564568705699, 0.9245024328281013, 0.30550419451400534, 0.1449363985686707, -0.7463487517398502, -0.0793314245950425, -0.094224092125053, 0.7489446703203593, 0.019752455481657494, -0.29565381953111425, 0.9982797053604973, -0.6146294781980528, -0.19818862147322958, -0.005572185826116627, 0.8553981336483947, 0.13053085134197295, -0.28106305516820296, -0.9998494678277141, -0.3924997993015351, -0.9720665649750417, -0.019342705238089675, 0.2693638197749679, 0.7517701654129432, 0.9923963025110294, -0.7588406695632703, 0.6112239574152967, -0.9919386783170026, 0.5355732147942551, -0.0010525079293810524, -0.9376294674825258, -0.9461815320278955, -0.7913027943247841, -0.6164638360174586, 0.2972802376034349, -0.007622588096806382, 0.4962460154382311, 0.6493052107965112, 0.6276349132242831, -0.4562832239100052, -0.9553757513974166,
                   -0.512337106526108, 0.7921807811425636, 0.8940025531725826, -0.4271539048283066, 0.8721777353206579, -0.8923037967053566, -0.09996935421278352, -0.4124049306288451, 0.567354928938683, -0.7666733383277347, 0.9169400608581237, -0.8870738303208372, 0.6602495269023885, 0.7946858674140342, -0.09766752620423524, 0.04526782473137492, 0.5151237992177629, 0.42076219598641185, -0.35279709186944475, -0.5427297904215871, -0.5951279954658757, -0.6341890369659253, -0.006152002032780546, -0.8005611629567677, 0.7389512175370827, -0.17994668387136992, -0.27681235462359255, 0.4863811632542163, 0.28607258661669177, -0.7428571164667181, 0.4286982493089846, 0.8048861791545381, 0.08092346377777981, 0.7230025968917955, 0.14142426925329388, -0.5560536973809025, -0.19229486312501676, 0.8397741744781266, -0.2697019570513566, -0.7777776246603896, 0.36928867762418904, 0.4802636418662851, 0.1742646821346019, -0.4236572522031401, 0.07930456143086562, -0.43759348271539467, -0.3125252150139013,
                   0.0738182117752999, 0.4554994854064196, -0.7800822229491076, 0.3760718536984844, -0.37275474266191644, -0.05379186741312303, -0.2755687167012537, 0.6148690252602671, -0.7881396868670363, 0.08780351011401177, 0.9660746971886898, -0.3455271530520807, 0.7263991826900675, -0.430494235303029, 0.043566214532271275, -0.8958198395390273, -0.21239494009139048, -0.5270560063177825, -0.4284397328385203, -0.11676252230996953, -0.2883789837428823, -0.44261804821170037, 0.6821099171938918, 0.9928190305371936, -0.28910863104063256, 0.8775337813819954, 0.3787537643550565, 0.03334563565098647, 0.4230889996399827, 0.24941187543235577, -0.10412270788121525, 0.7765304105009492, 0.2889746883510902, -0.8793359221486021, -0.977363085037275, 0.298805587597633, -0.9910138829177908, -0.7446261588134939, 0.9698779905680819, -0.8074523676836909, -0.6662553157182398, 0.3093115077215016, -0.7098827067386337, -0.8062143799100765, -0.21844177616571891, -0.149858628331905, 0.604800617169563,
                   -0.8001579143469357, 0.5263894927261437, 0.6681051138919825, 0.44968898817045844, -0.9023781660263765, 0.6626497685737665, -0.6737838837956449, 0.41836484297719556, -0.9169701827066166, -0.6714316271835636, -0.18498324809632538, 0.9654921971575099, 0.22426428432457812, -0.24222069175062821, 0.9054656054748795, -0.27439544724793974, -0.09350218594177617, 0.22915609654034363, -0.6654637100542002, 0.009835054131665633, -0.5072759753657203, -0.902181905949009, -0.6883727653926908, 0.262144025529534, -0.9546872596152225, 0.6156206286492738, -1.050909675686107E-4, 0.6882398431272627, -0.7249160506974008, 0.08733295829817456, 0.605141236544289, 0.014529572220693066, 0.019474044869475904, -0.9409023451244973, -0.5041011781432849, -0.6089240045517337, -0.5013647071271539, -0.3013426081032857, -0.6181242274217957, -0.17832043191403546, -0.6419253311561739, 0.30222334843821974, 0.9939357064796095, 0.18584257341831933, -0.9212972454191775, 0.9507522954755061,
                   0.9373053372498317, 0.9115151995208581, -0.28006874360356004, -0.26121123097545995, -0.5692834541093048, -0.3358372870868507, -0.8412794910156123, -0.28808055551279677, 0.4977066624898525, 0.03683404342794061, 0.9972466369725379, 0.674278133591576, 0.8423153633129172, 0.15755241489568061, -0.8658782416791806, 0.36542979213475046, 0.11993288606569319, -0.6255102924142786, -0.3711329686927545, 0.15635287796070108, 0.6050691419574505, -0.2639737077589599, 0.9366043108758999, -0.6228703942961713, 0.36132991589290175, 0.8519393671665454, 0.849696528306551, -0.3356548656419136, 0.9349008512342345, -0.0963972932247057, -0.44466547191236816, 0.30408633664588347, 0.3213566642443515, -0.9347778871102232, -0.8971687610712407, -0.5531779776680001, 0.49662764460694353, 0.4047707985069766, 0.2716522885812833, 0.28333726469524834, 0.1564719062247546, -0.5475697411132432, 0.34097562997682407, -0.2037159645257527, -0.5251831022218003, -0.9440180239143099, -0.416160464623915,
                   0.5398029802778135, 0.6698500474913789, 0.8867366896765929, 0.44370381638342304, -0.9499406464900875, -0.04224726646091437, -0.9837717338412801, -0.701485496882684, -0.047571299208737594, -0.9225109526732265, 0.3530398429862929, 0.47712770395155224, 0.606457555897874, -0.013924759967239186, 0.4107702247703884, -0.2347852272806139, -0.7823847385030875, 0.5522717240619528, -0.12771001104040725, 0.022535282673145973, -0.2547561636506399, 0.11765525228365958, -0.2779234746469723, 0.7307311757678472, -0.3509971488910131, -0.35245572989887397, 0.33755808956360944, 0.300560649664108, -0.4585022542500199, -0.1397333197753421, 0.8620380903670737, 0.7897549957113634, -0.8473301158356823, 0.20177348666849015, -0.5740894141867263, -0.6825506550396705, -0.3380822731947335, -0.17638052340889665, 0.33951925366818525, 0.9084014567670506, 0.41680908188050836, 0.8258457161619825, -0.7498475155925526, -0.2400195846996962, 0.3326345887266584, -0.05768466777777648, -0.5626919393003262,
                   -0.002828603772748073, 0.8549439433470989, -0.42543001938768943, -0.08549914539969827, 0.8656084762145271, 0.2675963922720159, 0.6212886705981655, 0.8125287747334331, -0.9900858740076797, 0.3417476638565986, -0.0960522472108063, -0.36174843116214594, -0.005612141900449341, -0.15341307511788282, 0.15529564010079255, -0.5654214378527969, 0.5453537217441828, -0.7194732267509003, 0.5157714661217778, 0.611501740709951, 0.12924155525390968, -0.6680593316390944, 0.2740132436928129, 0.27465117164161756, -0.5722587759553868, 0.0414329070830044, 0.19111515855063943, 0.2469904404278518, 0.0311374658851451, 0.4130163047304847, 0.16431161742775413, -0.33908977886519986, 0.24319524309273266, -0.6905360148722235, -0.7857212842715928, -0.00788032055829535, 0.27573420993146924, -0.13250823298324277, -0.1787385962297372, -0.28662603153519517, 0.30829388355933585, -0.6186814770843607, -0.5112509062147461, 0.36521918318122637, 0.898059266027033, 0.6285429253001931, 0.07943897781949971,
                   -0.22742503060079122, -0.9525202787266578, 0.9501644253539332, -0.05325952118538746, -0.6198837568207007, 0.8717818187624735, 0.046968572060910097, -0.8239077080210202, -0.7639437244352689, -0.7441590283046116, 0.4173857348153167, 0.35387603313846316, 0.023073884439733128, 0.14964277444763563, -0.3626756813068037, 0.1977292776017583, 0.7167673522787747, 0.2238311630991483, -0.7296307659495127, -0.8894262631494865, 0.946160054919222, -0.5083953207451446, -0.90287242451966, 0.2616917032647643, 0.9374215328085349, -0.7995102757615029, -0.43460635381629253, -0.8483010998091853, -0.958890521439506, -0.19271646398629683, -0.11180348997304401, 0.7842769795441493, -0.7251215874430146, 0.17172605274084551, 0.05468373446716557, 0.5460458105466455, -0.9422584507010612, -0.760774195121467, 0.16918517134083788, 0.6869701393931935, 0.36861262717236243, -0.5697704514894295, -0.0033077622304495957, 0.897210955522763, -0.06900401801448486, 0.7135922116107138, -0.6316224893512503,
                   0.8923487671167036, -0.2926676862276738, -0.08643180671597528, 0.1886715123411966, -0.8525800568508584, -0.01935023843900252, -0.81518851521589, -0.8141974095536848, 0.9612234179678312, -0.3921904513543917, 0.7870378476956685, -0.7394020500583691, 0.8858719464772897, -0.35222347472092275, 0.3007663162079435, 0.3137335489536488, -0.7774244884003021, -0.5000149377554766, -0.7521744228462937, -0.23048867400606543, -0.23683780079098193, 0.9320121428307968, 0.7150739059163829, 0.6646331330175224, 0.24785597155465888, 0.9324097483485743, 0.027048328518580078, -0.8666683438865628, 0.11334701035868155, -0.806311700279916, 0.773024026768923, -0.6043923635251829, 0.021206519517781075, 0.21208596856710327, 0.5662304625977996, 0.9432161766111558, 0.3852120481101866, -0.5151035370651271, 0.7977365080749057, -0.4815677194645267, -0.3428336548255919, 0.09277046870664152, 0.34478219460736126, 0.6935189528136809, -0.515692101066213, 0.07354571809479782, 0.7031010583850794,
                   0.38421057355425825, -0.9825733882586507, 0.42580171445581794, -0.6649586103692131, 0.46841517945696576, 0.6273144418855048, 0.32447532230034026, -0.8399381217934812, -0.16827028182590542, 0.07819544350575769, 0.18751425267501642, -0.9385585344193852, -0.37574653324273655, -0.18970334165379277, 0.6016863410329247, -0.9571905089545245, 0.28857482672768264, 0.3997354233075654, -0.2026974071807368, 0.2403678357922101, -0.9933022557170101, 0.4924150711520727, 0.35575328783665294, 0.38144213922635806, -0.3859354655791991, -0.45036296393892306, -0.8596034855730432, -0.5227349969952038, 0.7417221209342428, 0.5966520773678012, -0.939520803641984, 0.5476596050250879, -0.026929631147332156, -0.16861070350994845, -0.7905884381983816, -0.0024424137742360408, 0.13952031825259503, 0.4291178901949493, -0.8371435409154153, -0.09994839015007462, 0.5297823312862977, -0.8301888687176762, -0.221264366558519, -0.11205306970283901, 0.750393179280664, 0.6184981738306108,
                   0.7103954765510307, -0.38445636289313767, 0.25048345055558485, 0.5049056330068027, 0.025815621059189775, -0.17535367365932153, 0.11167564667665486, -0.3227948950856203, -0.18011558007104833, 0.9335622867791848, -0.39560475651859184, 0.5456993701909101, -0.48659432370131794, 0.7651174912778385, 0.22476675700686988, -0.7142041929760288, -0.3483581621555272, -0.4605172161591138, 0.4516576026427135, -0.7643700711162591, -0.5531602364095005, 0.5798698891890626, -0.6986756123519167, -0.09778051047391667, -0.9466750515959006, -0.6860784021002777, -0.0924957443866834, -0.2189566216353116, 0.6203915822698789, -0.8991132800999402, -0.1823122779225128, -0.20254382073957267, 0.5373562860459435, 0.17443582761411225, 0.47169745062084, -0.40563496996208825, 0.9667833392071428, -0.4367610659582053, 0.41658957231346383, -0.8808902378284955, -0.24396915615776926, -0.7638072605866495, -0.7877160186859968, -0.9964948029264493, 0.12390688720840726, 0.9164503102109332,
                   -0.30477860982556293, -0.9154827154070986, -0.8616335744263106, -0.9818289008358299, -0.7072653888295699, -0.9209369592860002, 0.9328190531628364, -0.017823839528916974, 0.41871321042718734, 0.9049378422462868, 0.8527596611261117, 0.9214576118073234, -0.36846981257707045, 0.4308906429398265, 0.2810738516306004, 0.9347579100601817, -0.80190877661857, -0.4576174552761094, 0.7595102952215667, 0.7062003013223064, 0.9832563113213144, -0.6744201003246793, 0.2054543466209131, 0.08121607284086485, -0.8282245564805382, -0.09935101217522702, -0.4827241992968052, -0.8062180774735161, -0.7697304881189873, -0.023565163835739922, 0.6681660972950905, 0.4172443600974898, -0.2703204094276117, -0.44787787615498487, -0.49248037325935323, -0.07922474803210222, -0.8260032249656317, 0.3574919535970402, -0.07944467712696146, 0.9357671312922622, -0.18102353080439748, 0.6019042618375685, 0.7410147606460624, -0.02875666548314504, 0.6149318968765312, -0.6068495811417918, 0.8684138267862318,
                   0.1646229392852976, -0.12940943867632626, -0.20916327130818013, -0.5895343694544091, 0.713946905444609, -0.9896096211886154, -0.12502435475129303, 0.35293323421031175, -0.0900635468631239, -0.11451928497403263, 0.4389755803219051, -0.7466894508630086, -0.3289784867487353, -0.5009145160767825, -0.04276202825626152, 0.7613715217935209, -0.3182585933114084, -0.28022447688833596, 0.331083878912668, 0.3887674128539249, -0.7553131240443909, -0.4184117100333198, 0.5827492286813325, -0.11850666426834877, 0.7784170199544282, 0.1078234855439526, 0.14570320863435637, 0.7931850205802589, 0.5285510706148004, 0.24573386656241492, 0.6905595679947845, -0.5861519842211143, -0.49349959259198695, -0.5377789593496085, -0.4706333671494307, 0.07776259212453507, 0.9300716566529446, -0.17396342409633014, 0.5570773011805625, -0.9440536703931128, -0.4875625947040829, -0.010119723932856983, 0.41751583973635475, -0.267300045093944, 0.830736943714101, -0.6138238671770313, 0.12217401880672152,
                   -0.4276005374275167, -0.29716547422164985, -0.8368005210901144, -0.29631515675625786, 0.5018407853476443, -0.9894709275149285, 0.9136177662508527, -0.348162458961671, 0.7650823252887518, 0.7297740676957678, -0.9146014701757972, -0.14148702856955198, 0.05187168635091299, -0.9367678360930012, 0.11542893747963312, 0.5889351965023542, 0.7573839939029494, -0.48224438669661973, 0.7890819576815453, 0.6001490471530446, 0.6001392638316876, -0.47085254145913336, 0.10435415422366745, 0.9436502024859494, -0.4933928184366805, -0.19117419333837327, -0.9929081467113361, -0.7823898661097408, 0.665148058197218, 0.9717782395699324, 0.7642117216970834, 0.07515279442438305, -0.04142007149274818, -0.7975410871322908, -0.5452862878036546, -0.3513434420174033, -0.3961986105653015, -0.27495286824356, -0.9163260153983668, 0.21607005486428643, 0.7585415731656009, -0.33324873918969167, -0.8377032357745786, 0.9137462671295735, 0.09944807600085914, 0.5376078694810635, 0.2474938134608291,
                   0.5913151808876456, 0.9323656105703508, 0.9974912788710255, 0.8973522621272323, 0.007630430145845857, 0.2647972500949196, 0.840814588118429, -0.22575984420825557, 0.4828405723158715, -0.5478177693826694, -0.7983316098325475, -0.78480819146284, 0.850153680544228, -0.4762361728266822, -0.13308955124489819, -0.5103955928271662, 0.09187531576660635, -0.8995377664996631, 0.28198243205792917, 0.17716780783041863, -0.6021795231867861, 0.9363071056082453, 0.3439989932300944, 0.6243412124607, 0.5502111935628617, 0.5466260536415211, 0.3989168814040458, -0.7244204325486452, -0.4823092117581036, -0.8696748413434723, 0.99435975638222, -0.17409512893112922, 0.9320987313402986, -0.9497610215132095, -0.11306189834607738, 0.6043062263546106, 0.7251673391191225, -0.49533284808582256, 0.5626713685715721, 0.3728450608696452, 0.6183162659628354, 0.7048563977992084, -0.8986173137517381, 0.8142306162746915, -0.6127630205901566, 0.8673266209521846, 0.6007043459117092, -0.6189763940249631,
                   -0.13602575205303102, -0.3634193980471838, 0.5725962319992948, -0.10748543708996361, -0.5072714511540461, 0.1728777081984234, 0.7350460711229292, 0.2907150641847154, -0.07838264232635783, 0.07139211549367719, 0.4438852742179118, -0.6952095669072944, -0.40884991434336126, 0.5232171404415018, 0.6444204455652724, 0.4236535597115527, 0.5711870817217193, -0.3146821853725412, 0.6698453447100701, 0.5393290326984388, -0.9983326554558329, 0.09567422914348844, -0.5268350611209451, -0.10091146733545475, -0.5915927459039059, -0.848501170198479, -0.04837259404159444, -0.407751413137615, 0.8937865496359647, 0.16255192825709264, -0.09257288553977805, 0.6979118159985593, -0.25684546186775914, -0.7017802183137551, -0.11429454130262817, -0.5671745712040968, 0.3413039281641841, 0.8661604419726832, 0.6406437595296746, -0.8360818995369399, 0.4593772330187431, -0.20803697772417395, -0.6972927367265493, 0.987970428415158, -0.04891607757651495, -0.0865928527523796, -0.5451357341098171,
                   -0.2687377437582843, -0.31187590240245466, -0.24045064509253766, 0.2728715829463604, -0.31506772118477144, 0.1682293261128207, -0.14599581056716104, -0.3555183863525426, 0.16717502047207544, 0.6452462326567574, -0.7365716166214864, -0.24530681083176686, -0.9087085517169637, -0.1407943550567088, 0.1063569347038662, -0.9910671128889439, -0.8809998531918186, 0.48533414650943185, -0.9558192760802093, -0.636680322253379, -0.30869383079744694, -0.7255184581901255, 0.3951956955936624, -0.8590678761912658, -0.7453845648138542, -0.47654720824659824, 0.44429467886158136, 0.235260933596946, -0.6623261307857609, -0.0077510669922078446, 0.7650129149566625, -0.7632021718387354, -0.21470026042224588, 0.6762369811276621, -0.07272031974362103, 0.46597518868212684, 0.4177698183135252, 0.944784372273904, -0.1666314624142049, -0.8445622429637585, 0.6601091318513461, -0.5260905511802576, -0.89382067418766, -0.9031067589981021, -0.7265739752787364, 0.12901455897699243,
                   -0.551873753449823, 0.9525533236290593, -0.7791760232016718, -0.505215453715161, -0.3044538485113384, -0.8715562191482511, -0.08668747426001522, 0.10284866521977443, 0.4126826229230016, 0.616626361000024, 0.6065303199619874, 0.5738196154137352, -0.6990588729803209, 0.5356421521732275, 0.5569839425161154, -0.13116056099201145, 0.44325221889171007, 0.23325060661695463, -0.30310311860903916, -0.36322331461642743, 0.46001955678287865, 0.7628050630818539, 0.049653221404577064, 0.2467361507784287, 0.022270409204864228, -0.6956297530991196, 0.8832115784946781, 0.5115570873714672, -0.15073310618173652, 0.6739537969118086, -0.31324422641737293, 0.33649162356333395, 0.3119773238721961, 0.4489884046648309, 0.4562385581987394, -0.4317285006454221, -0.9295537462775982, 0.4354683633578387, 0.48736589090587823, -0.8267143160738959, -0.20775525802658468, -0.7880923150740953, 0.8999597658856773, 0.5396077618568842, 0.9308202570560811, 0.3075553049663142, 0.45552833672897197,
                   -0.8238721449089563, 0.7867086981492832, 0.2070656470794583, 0.8802623144411315, 0.37305881295350884, -0.36879001463330296, -0.17890874417082614, 0.8685675410268086, 0.49160188272183647, -0.4587298862357885, 0.32202718480514725, -0.7823653174869218, 0.4665740817706383, -0.22404690168098162, -0.6835950769707919, -0.8685634912227349, -0.16568638162956484, 0.01902633842434276, 0.21019693501339898, -0.36706534066696084, 0.13919554140246837, -0.3811757304355192, 0.32056046685892814, 0.23616875010981286, -0.07439343531539633, 0.07569560091598881, -0.20268619848981806, -0.4447097378335909, -0.31175789679006716, 0.4589794673020138, -0.26029613742329594, -0.6345051505573518, 0.132105519611029, -0.8204026106411684, 0.028790193148005905, 0.7811977805372021, 0.8846040451609634, 0.6009877885035146, -0.6569645269578415, 0.6796867287930068, -0.5943474806258877, -0.6597491956577526, -0.4155415355276275, 0.27387800938158513, -0.7422893653636031, 0.7930120598325918,
                   0.5149711451513974, -0.17430804735982508, -0.8738035437594895, -0.5903464378875465, -0.2127128264672311, -0.10804581547443215, -0.9782673169352398, 0.557201988735494, 0.7520542398502021, 0.20762057818290125, -0.05080039772593703, -0.5796957272758785, -0.5673705488476679, 0.33651332339114526, -0.00541670288182039, -0.6736388477613462, 0.8811809235469221, -0.44235293771650075, -0.7408769441588958, 0.12403035729977296, -0.23543379936659892, -0.8943857987957955, 0.3612788469547046, 0.8009512940073829, 0.4757018608600905, -0.15361049457707576, 0.33055376092550826, -0.6043159346021711, 0.32590643709424927, 0.7453039487305511, -0.3667771649059903, -0.5595040824452857, -0.554038034029747, 0.07182935490672104, 0.7985357567634006, -0.33423073044298635, 0.9084075003153724, 0.1862058312611128, 0.65497490992918, 0.5034457073817802, 0.2808414568191542, -0.03416679310379833, 0.5396679949324985, -0.9539387553017573, -0.670222803029114, 0.4547411038435123, 0.01006810979289563,
                   -0.4504315337067042, -0.08313996666145029, -0.8426601853964184, -0.43446875270443885, -0.6877176968332082, -0.7359789319405943, 0.5480469809404216, 0.6774360600400817, -0.1810740036385925, 0.7114075282599153, 0.831274821393946, -0.9699935431659246, 0.407895507997041, 0.8516965244855741, -0.2383466974124968, -0.7099866568242112, 0.4015002392524478, 0.4966625089606109, -0.5777447616519882, 0.9520475465099745, -0.3669209777418012, -0.6833630278991907, -0.7410194240187118, 0.8470710774147521, 0.6115314759188841, 0.19076216399661683, 0.3821465575575793, 0.755470457046233, 0.18294371939465837, -0.6842920477235053, 0.17172207887791968, 0.2318502233468238, 0.4806308864698434, -0.05721437667035456, 0.5605351018529325, -0.7726784669277393, 0.19905621008803953, -0.7518417743969059, 0.02017167555298971, 0.3860555034609119, -0.21626995329491971, -0.7118867736651164, 0.23717750385627712, 0.9963873414493234, -0.0016695083001527777, 0.2511278098872469, -0.25743896991880066,
                   -0.5072279481297342, 0.5119297664339022, -0.7092758302807651, 0.3454907644266656, -0.866591152493178, -0.3566502947304442, -0.892123936046896, 0.12246755476168558, 0.019367276660395172, 0.206310980174367, -0.851004548232337, 0.9261442916521805, -0.025438130908121526, -0.4818184511462491, 0.79529172411183, 0.9033566145349707, 0.9719408121616773, 0.06676981223220291, 0.24156372548364513, -0.6501826444463399, 0.7697666596572592, -0.9262509443014406, 0.8972478506950774, 0.3793197871533731, -0.787115269356442, 0.24993998869076783, -0.910186113987834, -0.6518160591989726, -0.7426103421228283, -0.5931803845212553, 0.6103637852314614, -0.24892811961302708, 0.19311648840104945, -0.729735952240758, -0.7864446656409927, 0.3921877322519296, 0.7668983923708166, -0.16332532274740497, -0.431262760425392, -0.963730585002198, 0.950547505887325, 0.9182770425510078, 0.35644981730057523, -0.5433305310665122, 0.9510338158196099, -0.8564773899653682, 0.21420809602116453,
                   -0.1767873659855792, 0.7263490969038973, 0.6101643218498711, 0.057752605061878626, -0.28959055401964906, 0.7490613699823403, 0.6327172341410698, -0.061282736755237766, 0.6569941195685822, -0.9494205637362798, -0.41734361741850545, -0.9530706877673443, -0.8293523167280861, 0.6826635508769006, 0.688844484308897, -0.27467719216574515, -0.6528440860866302, -0.35126718099729315, -0.15267703930701582, 0.4377116881211889, 0.8611438399273175, 0.08603692966174581, 0.9837398220516325, 0.5144354694658384, 0.9859181513787063, -0.9792255911833192, -0.6816707203478012, -0.8018459289318034, 0.512839944744055, -0.628244570089757, -0.997501454671754, -0.9781631923224987, 0.9660412617963945, -0.9557768853927529, -0.9722834512360881, 0.8732533113796968, 0.3461879770353211, -0.06603713848545234, 0.0515423431451969, -0.764958194846652, 0.7873846478424535, -0.5471556525990349, 0.8422693473923359, -0.9633051144186444, -0.025795004577103597, 0.7904571228597774, 0.4128032745925949,
                   0.3145687731548932, 0.9412241501486556, 0.5425160266822022, 0.5960324492231346, -0.5517126042943927, -0.03468142080240133, -0.43477165865177425, -0.19191110753652452, 0.12667899495447577, -0.3386094381189029, 0.9558298715261218, 0.052539826962562985, -0.6063522558196168, 0.1050643708332697, -0.08698623375426462, -0.1623095490835904, 0.13414863998350546, 0.831292296399248, -0.10300980181723474, -0.38918239815468514, -0.8172995851728844, 0.895699065411607, -0.012568390575835364, -0.4427031704543267, -0.12836468995510875, 0.8358839779549958, -0.3714264441002897, 0.16922425949400588, -0.7164479372479127, -0.6672094091596712, -0.945743337126765, -0.1427237227680267, -0.26779814538951774, 0.6476441633532581, 0.6017109606982645, -0.9388099097830314, -0.9316949328437909, 0.4978486011045653, -0.14860015614779587, 0.8083892488878823, -0.7084400493366694, -0.06437599076283762, 0.07125547774539154, -0.523081256846391, -0.35262819880973906, -0.9939425414323639,
                   0.7971545710604582, 0.6379943341628507, 0.8965958307858606, 0.48663010782052596, -0.4525023578067693, -0.7148870433378656, -0.23114751688086077, 0.8481739781949227, 0.6482718395754101, -0.49919059616745876, 0.7447407890642068, 0.7276031418178643, 0.9359913946225931, -0.33121263073994145, 0.25953894406742606, 0.6656532273040809, 0.7094274492797572, -0.2917083432314347, 0.18074901242258057, -0.3617441056809525, -0.28294735156672624, 0.48923069702971533, 0.9841069232900868, -0.0019363148601727609, 0.19564285219513367, 0.5806976054285988, 0.43744484541727946, 0.84303204084141, 0.01649335903265481, -0.20782019072984426, 0.7990593305487397, 0.5418558158357656, 0.8492228201244707, 0.5858868466757323, 0.5470026470518148, 0.17144897618437382, 0.919153465485177, -0.6297318179772471, -0.5220170773754698, 0.9595447779161994, -0.58722775261371, -0.07106903120858465, 0.6493210347183849, 0.4758870122902241, -0.1232070662757716, 0.7958114251646684, -0.9153188535275603,
                   0.2783712195922696, -0.9401455593229457, -0.7434089906428192, -0.575945370876318, -0.004112977455639433, -0.4380929382494321, 0.8379920199707918, 0.7707451043097391, -0.5824979696052273, -0.6151465060331403, 0.6407253662124659, 0.8728654380309715, -0.941980162705857, -0.9809732456218105, -0.3214939030512407, -0.5258740257112924, 0.882833552985367, -0.9463964972040471, -0.49530903478389, 0.17845626251953473, 0.4003889742813338, 0.4650667640389745, -0.16026544523956288, -0.5047296496919942, -0.8315402037592279, -0.28365339507025755, 0.6567999893502017, 0.8669492321688852, 0.9108300534707399, -0.40971536661284125, 0.23950124572634723, 0.42738211773488355, 0.903868317364444, 0.2845826464432726, 0.09489501759733332, 1.7124192124629012E-4, -0.9321312369078794, -0.28858825820928624, 0.2864803052090019, -0.5826275372969323, 0.28592606080059135, -0.8266267202433002, 0.7461044459962625, 0.6244090543029912, 0.8734999962210632, 0.6285204282470491, -0.569684366506312,
                   0.23406075258999626, -0.1652822229984705, 0.610662055896352, 0.8058414416383923, -0.7978028340381216, -0.4331486788830825, -0.4628113624863823, 0.08051474157311933, 0.1201894201578464, -0.9877006166709548, -0.38482568411928253, -0.15989177249573583, 0.2536996673811811, 0.9077920188355022, -0.15247508539858123, 0.3081794181115327, -0.09518588625293178, -0.2565710055511006, 0.7444581603838802, 0.3151689936059565, 0.9342498950821192, 0.3195513728753121, 0.26161637161371787, -0.755364079267036, -0.6473669642315898, 0.37733793979565133, 0.21462972143045778, 0.7186528896078748, 0.7746094777047259, 0.6794082639388093, 0.9609616921187523, 0.5910269849965093, 0.22681665737634726, 0.1296043315541633, -0.5066149267429918, 0.9361505583046044, 0.3105098153193311, 0.04651427935256369, 0.2554525868002888, 0.1529282796975009, 0.04190129943622001, 0.5431915156361307, 0.9678411269254501, -0.9526336047147932, 0.8280382341087273, -0.8840866411966906, -0.4545350241867421,
                   -0.18594811502053465, -0.016691531977429896, -0.9922856285706554, 0.7887682779571346, -0.7941113618430411, -0.04856884458137389, 0.6878199028116168, 0.9486234456987819, 0.8630990493667627, 0.9700528832180213, -0.12400457973099388, 0.49277695785802056, 0.7666404094310779, 0.2998374302873019, -0.058781142661437524, 0.44280065861717155, 0.08782484690625947, 0.3372011937583159, -0.7830265449109441, -0.49597906669060277, -0.6749261984577337, 0.9396133886904863, 0.8008486131623394, -0.1230848127525932, 0.5503773332546398, 0.11464445040330773, -0.2559725292332924, -0.8830202725879768, 0.8321696661686222, -0.14128693246320245, 0.18663102318172342, -0.2989108598127126, -0.5420264016177079, 0.3787464646899714, 0.9577776986473128, -0.9381174253755491, 0.4493055092481355, 0.18973541996950827, 0.43460977861469385, 0.7677288574876733, 0.04297527002689128, -0.3465695446050232, 0.042427135710354724, 0.25433635828921175, 0.30705410915483267, -0.5918088197516105, 0.6784603296620364,
                   0.05509109913745003, 0.8664257612594817, 0.8372669885313986, -0.4452946430374596, -0.1692795658740489, 0.8319753339365943, -0.7674477850524375, 0.4806331727495117, -0.33510314125677887, 0.45911221862753937, -0.3668748487288893, 0.4444773309665322, -0.929089613746165, 0.8054634621533248, 0.452645364190837, -0.49780593755255387, 0.9455556275517194, -0.28578596789585986, 0.9569498315069884, -0.5539200341355568, -0.6901067236043879, -0.18128720385811636, -0.5870462920314026, -0.49957767056860125, 0.09256261358796647, 0.9456010868251303, -0.6977725542221835, 0.3794485544616766, -0.5601628072151288, -0.7121359340384492, -0.6226126344341536, -0.5567689437021444, 0.07763092149330753, -0.9741562778484438, -0.46329747466596904, -0.5999042883802226, -0.8958225342862176, 0.6933936826895395, 0.5020865694303689, -0.9705292454653982, 0.25812228562111805, 0.5179971757078958, -0.8920919455898411, 0.7370056728784584, 0.18528150687667866, 0.20780886086097117, 0.6428197026559148,
                   0.4041792066749572, -0.4816626960081347, -0.7847769707236858, -0.8659285510585624, -0.04249442102001244, -0.31274406415221634, -0.6242891375620803, -0.2764618377591508, 0.3578405325458376, -0.7057707127940394, -0.9976902363547688, 0.356959233703253, 0.298177285768352, -0.38454274950613043, 0.010288979770585849, 0.8926513184424596, -0.20128494414593812, -0.6668970767641704, -0.5468443305506796, 0.4332227680468781, 0.10087157745613151, -0.045730608264233474, -0.029692534768269496, 0.19358323867227223, -0.21269659745010538, -0.46013885796126686, 0.07827587837252548, -0.4591930095746195, -0.10711276505327927, -0.6552896911743074, 0.981648486923421, 0.9761749379908822, -0.07799351828194778, -0.23863800814062097, -0.3636038193940805, -0.22301411211115107, 0.1503779845942521, 0.5634300857927861, 0.167472856201222, -0.22450714392782123, -0.6349554728845779, -0.13955113353962734, -0.5758032212600119, 0.32458572196965507, -0.7159509510680409, 0.4046633461714151,
                   -0.5489383786839521, 0.8846079038407513, -0.7023026581559586, 0.9312153438065798, -0.04173265241786006, 0.8629595830264469, -0.11379330271964272, 0.06669803472864966, -0.5045991757382289, -0.6520938776209328, 0.09392834313329801, -0.9929312347970023, 0.9633369524103725, 0.20027548165932596, -0.22177752693749486, 0.6444234572182468, -0.6564280362141592, 0.6014946267559549, -0.9985451075533911, -0.8868465273145043, -0.9191593727124237, -0.39631980674076295, 0.8974752722203001, 0.8295198705926274, -0.5759895830524786, -0.5127358097026353, 0.6193274035784095, 0.014697133309448995, 0.211885495395149, 0.8039016752516273, -0.4371670057374446, 0.4198521619734865, 0.08469479431467919, -0.13282787295021814, 0.9610797586601927, 0.5128424241412923, -0.09132130505359992, -0.8491472947689169, 0.15773737247249264, -0.26036992727470754, 0.3203292278639891, 0.564306872660749, 0.39251342043330273, 0.6395267168364762, -0.2715200520650669, 0.8967973525529433, 0.7699532428693916,
                   -0.4281022407479462, -0.1434185078303194, -0.1427022126835813, -0.7668219759730361, -0.39702824751320964, -0.03585936705286219, 0.08364422797961857, -0.17295421605376937, 0.7049146849256132, 0.25543921126543756, -0.7453682849289536, 0.3106107322911187, 0.8353464671611739, -0.33470718975841685, -0.9890723379883795, -0.733352491681299, 0.6939906496670112, -0.7942532909621367, -0.22456031090210615, -0.7436035028273724, 0.9941651823404021, -0.6357503718092294, 0.7610399965418113, 0.28057795716214007, -0.05966231825392776, 0.6629849790535398, 0.46331554071386805, -0.1668873116238645, 0.268983322291964, 0.420680514584556, 0.7582774070615752, 0.17187831070544246, -0.31149940005704013, 0.002357674404807275, -0.6402705410193366, 0.08121965065287218, -0.15504612903667891, -0.6008768965686573, 0.8382616502958802, -0.6361761790449898, -0.9560703588108934, -0.15075540490151362, -0.336401398538565, -0.4250399003842711, 0.8748065069224462, 0.39339474941338515, -0.1956995921843241,
                   0.09997578817948183, 0.6122568277208138, 0.75108657195047, 0.3035776092206768, 0.5763298263947099, -0.18918804162842218, 0.7119655657311486, 0.9472483969871714, 0.8743365088689421, 0.8618391559385419, -0.6452370586116509, 0.6496311713498188, 0.3761822783043929, -0.5517054550567904, -0.1616435813616499, -0.49456797588542867, -0.8693308299809044, -0.18886931666346096, -0.7464750710202921, -0.40954861220191696, 0.6115293219247242, -0.46774534145729185, 0.08062347167591688, -0.458799727323544, -0.7299548511213758, -0.8575535682169437, 0.16754731327998917, -0.29671758148217453, -0.1518991399250853, -0.7213354477220666, -0.17600786823879555, 0.2106636031345812, -0.49431860663520943, 0.3436803772308905, -0.37783948937417033, -0.2572850558135171, 0.21722553457595084, 0.6619990258274773, 0.8007031971347618, -0.09681574243013702, -0.39586952360404193, 0.9795397139816067, -0.21217402599158897, -0.8249875950262906, -0.603904292177869, -0.17943572802022945, -0.7088669986614529,
                   -0.7728201798698731, 0.05188550760776822, -0.30614130823669705, 0.6459171807631725, 0.98453248138898, -0.27952189430904495, 0.4311870240376352, -0.32974903582451254, 0.3188026963874886, -0.3135372807809327, -0.9554461587837604, -0.8556311213069832, -0.5211943504322081, 0.7189337171431274, -0.5951736103479786, -0.7865238200884894, 0.4500583907700668, -0.6689759563752307, -0.2272169335176717, 0.5072567908555485, 0.3188333064562645, -0.1423911454173572, -0.2453287888659137, -0.42400415727766205, 0.4129242169857763, -0.5283978134426337, 0.6078718063007986, 0.3623708509628256, 0.5011681764207805, 0.7846597931369845, -0.011714677091811732, 0.8596087042680904, 0.65603219977141, 0.5159908263186563, 0.2095678929398439, 0.850843677885849, 0.5183397480622065, 0.8633740999214559, -0.12463715419962185, 0.6811545603574887, 0.6504771919371484, 0.6733622492083031, 0.34918829546540575, 0.382897554927363, -0.5758136648257508, 0.2830504306977466, 0.9426568275981215,
                   -0.8434059804328509, -0.5386859008498421, 0.9936750665519622, -0.06878028059639152, -0.7962281839297127, 0.03065750900328368, -0.46330031852666287, 0.5476426096166043, -0.7852005535303739, 0.0013057281657331554, -0.3940653338177076, -0.043063944417097266, -0.5661063782567084, -0.03735160855891917, 0.5546300026087383, 0.8824797204456816, 0.1551208703997049, -0.05651590703279519, -0.3056317757882774, 0.031594665113592546, 0.7473543094328974, 0.7974285579299731, -0.7170218989568233, 0.7598336021070147, 0.8627884527671723, 0.7690880207264086, 0.25157783892112584, -0.4832254024010636, -0.6472463092639964, 0.015228663938453568, -0.0037507959047329287, 0.5321203588478449, -0.40378539650900724, 0.9274388936681401, -0.8522668195304042, 0.9294185358203972, 0.4404754785600282, 0.9875463069903674, 0.13292857238528422, 0.9957370705973305, -0.2397421090939167, -0.4156824673368351, 0.5548670543367045, -0.3108583913052845, 0.7604554263030159, 0.7297342232045327, 0.9432263541902866,
                   -0.36816046489145093, -0.4204886858230301, -0.06459556883776463, -0.7346640870476335, 0.8153692680767366, 0.49450537762206004, -0.35556646267799286, -0.2619807500760112, 0.0352306811666665, 0.0856776041101619, -0.43865381861708475, 0.9435076554432473, -0.32413344835820346, 0.08725547917465248, -0.017586220630161575, 0.028050674744512172, 0.882921800008438, 0.3330940170914012, -0.1989897731441479, -0.1502601026597512, 0.38822192579939885, 0.19977693537743701, 0.24738284887452133, 0.342452187627329, -0.7060710716372889, -0.7435312931056681, 0.619490559724883, 0.07571037962375748, -0.024152937703489474, -0.550052129508634, -0.7286707194760715, -0.9843512161226349, -0.06813256077207508, 0.08489186924364778, -0.052573070958708, 0.8301627138523511, 0.8800563236074934, -0.8963746821723315, 0.1869591428247448, 0.7317106257024824, -0.2980935265582949, -0.2738163902311861, 0.9682001337431139, 0.8141294754783528, 0.9297875640887419, 0.38068460498260603, -0.9762098826307266,
                   -0.8847868627299531, 0.39364681950150837, 0.03564129809774341, -0.670434346364577, -0.7753045900377393, 0.37052696401259766, 0.5399901532732236, -0.8455346266192976, 0.1576689478396427, 0.1214836511739994, 0.3832488206625213, 0.756288355975274, -0.5736368934838587, 0.05400071045266808, 0.40735900498555355, -0.6821503489597458, -0.23140694194932876, 0.9140173280207633, -0.566589847760147, 0.040340575680714474, -0.17091599160385318, -0.6098043986148105, 0.5105800684273616, -0.5545299062405062, -0.7597351231293332, 0.30407463695351145, 0.5512700611296781, 0.8852262024594111, -0.9879836155826491, -0.8367288663411798, -0.5144483390044088, -0.9821357421976158, 0.9152484893819093, -0.7290973184314138, -0.25031603433672367, 0.5809933261960478, 0.6320740097795627, 0.8129261495753037, 0.9175996398960207, -0.8226020860738277, 0.8457345793947482, -0.6139017406900824, -0.40910001953827635, -0.013502149623381587, 0.1563268083482945, 0.2827815447509461, -0.7179613737805179,
                   0.3241149651375579, -0.2662339325802088, 0.5148349481114012, 0.9570150658278977, -0.20451962843096405, -0.042682989235558466, -0.962657545271949, 0.14803901434350997, -0.5788177110516675, -0.05859014019274866, 0.9239508766782967, -0.896731696208271, 0.5630194052904312, -0.18964766615341166, -0.6508982342670355, 0.3098621084509954, -0.26350851449976953, 0.8599376785564405, 0.148458818913334, 0.9082337059302157, 0.46892267513527597, 0.3318398697699583, -0.5426520054791528, 0.009525944024261879, 0.24698083168149654, -0.5729220049253312, -0.5319577726029432, 0.7349745483017776, -0.7053802309878743, -0.6022782118454408, 0.7002396658258845, -0.08503593839966705, 0.4322388414596272, -0.918598305338165, -0.6549014068916534, 0.8172223985756861, -0.4098820759334043, 0.08107664385467683, -0.7782913487882348, 0.017740341988773167, 0.6839532242911115, 0.9429042251568636, -0.3241065070250764, 0.584643113818212, 0.5075840206209501, -0.8520521444817377, -0.6452447018042804,
                   0.11438164869723089, -0.9422336114871159, -0.49115357561140205, -0.7936125276010184, -0.15874802436157842, 0.2781150682485585, -0.34953740363459573, 0.2826975899235278, 0.03539134206257599, -0.40589858431323056, -0.34608344365010946, 0.9642979769583446, -0.7889589392808649, -0.191740495645619, -0.3436973526467326, 0.07993691094904465, 0.286005813164121, -0.06548051111114073, 0.22377636168510562, -0.9410404317869296, -0.23511929948840837, 0.3728519311064409, -0.34640893487570157, -0.26619714688924456, 0.2627568486970735, 0.46017262640929046, 0.011053130869134842, 0.7005441460795663, 0.24275820262513403, 0.8807241326300888, -0.32974442584484787, 0.24903977116935327, -0.2900980996230986, -0.3210865063840873, 0.40284073977859225, -0.20979578101294982, 0.7854156429869845, 0.6898543575600862, -0.353067270781771, 0.43649732848375855, -0.42021452761588374, -0.9123753903722902, -0.511227971437682, 0.9327014360167385, 0.5894497969952885, 0.44851754494954044,
                   -0.8975260487116228, -0.5124124811278452, 0.9611930894725411, -0.7636696244196446, 0.9187246439479928, -0.776917069499723, -0.6000634299165353, -0.061000999828642266, 0.6311172594892318, 0.936071224499325, -0.7602386278250417, -0.7793466678410645, -0.7007763055325325, -0.3410317295769183, -0.7453581514703729, 0.40043433292169905, 0.27038593632219077, -0.8912641130339671, -0.8454938861371251, 0.3291778794651212, 0.41760266818536107, 0.5300088051498832, -0.6061175130652168, -0.6926624469865017, 0.2787776358481209, -0.3334726634594327, 0.9145309473211483, -0.9277164327515981, -0.917014082289958, -0.36391721503656616, -0.24990143655796615, 0.08038024037765101, 0.3623059320240123, 0.33769883192315575, 0.03997949397200751, -0.6809890945534014, -0.12684504816690745, 0.6857882366487595, 0.9312004552077049, -0.29401576369709814, -0.1110432113154518, 0.444538254419639, 0.09129267443838085, -0.23147848005679506, 0.5763608998032177, -0.27320411617084783, -0.8882473152394987,
                   -0.25870962999931546, 0.13012273813235975, 0.32523765630451806, -0.04916540487828969, -0.5172841729958049, 0.6171458495166895, -0.0403744593010793, 0.7460570913818114, 0.6810654142062036, 0.06807693690254113, 0.2524664797788012, -0.17905105719610415, 0.7065702748192626, -0.7447646190731885, -0.2135929689372833, 0.5171980977071249, 0.1631103650472958, -0.19145009943364366, 0.9502117969365729, 0.6025086313963044, 0.7027569232787747, 0.8465650108556515, -0.553810238318972, 0.43611237236194356, -0.011268222043633092, 0.1374712500524904, 0.3104470808608053, -0.18197917343942072, 0.8271076041265075, 0.9555028854116872, 0.08836216574443023, 0.5047450776868663, -0.5651603130093863, -0.7390611905852233, -0.5888069182202438, -0.2566493861351762, -0.4285145226770748, -0.39266887672776685, -0.3642559125062539, -0.14264381944303794, 0.8342331389812503, 0.2674781474839709, -0.5624674282757898, -0.0878750853407424, 0.8006262406602329, 0.4700486377726074, 0.288914620209723,
                   -0.20596210223697886, -0.09936234036150382, -0.6483458864290341, -0.3928583475515999, 0.6568797216198214, 0.9095114359349332, -0.7454422632472197, -0.9766623509744832, 0.6205072158957954, -0.22494448087973384, -0.6619019889980216, 0.7019004182349851, 0.8510129883007274, -0.8721632505309493, -0.877703133832531, 0.930847983314619, 0.7082349893344926, 0.26999205620887956, -0.4549882946666375, -0.6928551214146195, -0.48439418748924457, -0.30186265204559914, -0.6106178012582324, -0.2429399828168668, 0.6889744386935477, 0.288120910458165, -0.47378761446761253, 0.36012318573555757, -0.5448259160584765, -0.8458670165017061, 0.31632068699801974, 0.48626199653413016, 0.1977517007183467, -0.10805026237593629, 0.7307792898727941, 0.10921853393460079, -0.41358680884095755, 0.4258551621647919, -0.7276434184185514, -0.6746418856449474, -0.9216191801497506, -0.9989503995321842, 0.6193658317094064, 0.08550089625763535, -0.8760010730377348, -0.0018803983717741168,
                   0.06351452594856766, 0.9384206037250671, -0.5344345996820838, 0.19341034040173, 0.27623919683983345, -0.8118292443480462, -0.7124940495764363, 0.8489051248436854, -0.28659869714830544, -0.7165845341963577, -0.3423499906701415, -0.41251539947401783, -0.9438900315509724, -0.5145437043666239, 0.6935071681322538, 0.3942028262987882, -0.5141711182524062, 0.9026625270822992, -0.8059167626991675, 0.2814822839857962, 0.3048141344667945, -0.16747191349981394, -0.9030835192591247, -0.8341026418939745, -0.5616786955004303, 0.018594252922353283, -0.9483448370397076, -0.6735208238200987, 0.6929104058630544, -0.09200796787289711, 0.6456367821203901, 0.4492428164943183, 0.29556780518129244, -0.713625586008056, -0.6264278723466845, -0.2900989769886997, 0.20125494739120886, -0.22947558050847094, 0.3125763575785738, -0.39350395546544137, 0.9318846864505175, 0.5549396766195065, -0.3545696999164787, -0.6740536980270779, 0.2976159575660906, 0.5372109237793066, -0.33090393981390953,
                   0.3132694293264189, 0.2517486314230464, 0.4880266590896163, 0.7321516957676151, -0.5904899041537204, 0.34339119776465354, 0.28515214462352123, 0.6639494430890476, -0.5658339756176354, 0.7574636903930247, 0.3967467351368157, 0.6899442963638482, -0.5898289905953105, 0.7694902068813032, -0.22210156470558462, 0.5717784503699346, 0.7642596863837632, -0.406975512780525, 0.13527132037845102, 0.767349262591777, -0.21719534351374215, 0.060444167315574004, -0.8817239509490122, -0.2247721870887769, 0.3629817904463202, -0.6467615937342002, 0.4111897888316789, 0.5094373478416063, 0.6713850468805818, -0.9141941111190475, 0.9291676475944035, 0.5447393064108961, 0.7069783902614739, -0.5528310082122274, 0.3621029191684406, -0.6207824789083396, 0.9625373556908903, 0.8179022952672654, 0.1638093299442427, -0.5199099836180165, -0.6115009256734099, -0.6950370816746039, -0.24998572882669268, 0.8103198019241422, 0.6854040751678212, -0.024998284267339965, -0.8528332352582855,
                   0.7777355735702753, 0.5858807736829836, -0.7828762674303791, -0.3749184871105191, -0.8949851158759545, 0.23438685448483088, -0.37104084358086165, 0.9499528078182828, -0.01197807652299554, 0.032688919764016555, -0.9047061539091987, -0.18970875369273288, 0.09163363843614913, 0.39132735365477456, 0.31640136222935245, 0.6968492923419565, 0.4263277706124038, 0.7120779709759222, -0.0563918268850323, 0.05861841330715234, 0.9554479533778706, -0.3763113727038876, -0.30748582689363113, 0.910495686304081, 0.9021999647659071, -0.9970655632294956, -0.6703185559222364, 0.23423423204314764, -0.842923563183422, -0.36469991867378293, -0.7335045689332529, -0.5071936937189221, -0.0694552701218829, -0.49563535865022357, 0.644291858985224, -0.6306758819924336, 0.8524767501335277, 0.6462286358773262, -0.7422648479329055, -0.554820338597517, 0.08893695809682756, -0.7349104041975298, 0.4388233348286621, -0.05192136900793587, -0.2714123344506494, 0.37833793051856013, -0.8882143918931653,
                   -0.07083836863910808, 0.6851103652940966, 0.64430959334151, 0.629529413414196, 0.26941936412210366, -0.33131742627023386, -0.7769009157519418, -0.2848157593585918, 0.7691251434243276, -0.6013095524954073, 0.530601060355153, 0.8576804729065675, 0.24002348673192997, -0.8385021846314811, 0.5682143546003333, -0.15180861984360994, 0.9985293486876639, 0.061233436893561644, 0.1454083226443763, 0.7946578011861114, 0.44920779927476895, -0.13180212320940665, -0.2403828959752281, -0.08549438152906585, 0.551319904541177, 0.23863336062741292, 0.1576921490069576, 0.6810558613941362, 0.8772361461457896, 0.26123964608097117, -0.32458976255575656, 0.48701924075470604, 0.7000338255288534, 0.7799595220867623, 0.3362985809070165, 0.8888563226667383, 0.08861195358337515, -0.2508009040146668, -0.6116945686350213, -0.29132392791160444, -0.6578295053266359, -0.0032124938272737324, -0.5333906111501754, -0.20426438143401393, 0.26191882955540735, -0.7983779016503509, 0.89312613801055,
                   0.08749457768564217, -0.2636078366701129, -0.3629144337896901, -0.241961993072642, -0.23943112394840416, -0.09222875388414553, 0.287961976385839, -0.6240638366895346, -0.2751818605272076, -0.2098841852322535, 0.3177193004845906, 0.46893173481143346, -0.5108074115503534, 0.62643831564417, 0.544666915140759, -0.9986429444640448, -0.484144249524326, -0.49994074239394504, 0.5796369378495327, -0.024555330392487473, -0.4205112290466757, 0.7567026665516305, -0.41552110231745454, 0.6946364461401682, 0.3615751838202559, 0.6473036670290351, 0.2297979467300404, 0.010941992933243938, 0.5348525399785715, -0.7324823764177821, 0.3620073618873698, 0.8950295216129092, 0.9310108794185803, -0.6515646345985979, -0.8660929961649766, 0.3024997031057959, -0.9648937129566939, -0.25597600206240245, -0.09036059074658387, 0.7570570941104879, -0.9869501580567857, 0.5643292241222515, -0.8759835607665651, 0.33106099894733565, 0.753922582208179, 0.8194905417199849, 0.4578498053286164,
                   -0.22379860831164056, -0.0032428177860022167, 0.1681360164018444, -0.8149408207901092, -0.5581872356572222, 0.5118183369976077, -0.09526079199032078, 0.3539726180801146, 0.3156034680308015, 0.5193043422805943, 0.6740365051310806, 0.5843747522088714, -0.968655879981623, -0.9462174444458553, -0.433381518690487, 0.6934975102745786, 0.010323903848276972, -0.17943748450686492, -0.7716753597684634, -0.8917086521404107, -0.9618501788769629, 0.24060090429949854, 0.4070869478531207, 0.07795260670313753, -0.474270924389844, -0.1565766369286954, 0.9643821579175433, 0.06982558541969008, -0.5369594584132855, 0.9429384006751442, 0.794469848968631, 0.4134371758374782, -0.35343335436212, -0.381090178413779, -0.6768924327942158, 0.25156317534496875, 0.14602346458771942, 0.4272439429741568, 0.29845089130281455, -0.8361341171618917, 0.17329131242986806, -0.02486037559389631, -0.5596384968204939, -0.5522171991244795, 0.5262145197487205, 0.5225383543662365, 0.18036195278916423,
                   -0.02701228060912908, -0.39779545025094687, -0.8907963390321758, -0.7970085825583368, 0.769885434770031, -0.3886542092986205, 0.3571486945520832, 0.790333032670747, 0.06924171001941803, -0.9340299859057228, 0.061021960838108624, -0.9250024296370765, -0.6208079113691232, 0.8834297763661334, -0.045548400066146355, 0.2228653675833634, -0.6324880509932289, 0.5110676487318584, -0.37176032025377204, 0.2954033402623426, -0.18882789435638192, -0.6328562833373386, -0.5233124138260064, -0.9176106383447193, 0.6818447679282507, -0.41125146356240894, 0.7612256808368476, -0.45060781334433675, 0.5551644726329059, 0.5874958597126547, -0.5940905832441132, 0.7103709529836524, 0.9631901855975464, 0.23605339501503853, 0.9772361330283554, 0.7142733737382128, -0.9463449777823887, -0.35666328266235037, -0.5488734838779041, 0.7255278409264672, -0.023769753286878803, 0.42345699125779546, -0.16719441438510851, 0.7818397896151603, 0.7851717476474582, -0.8767226203713632, 0.4707086055094438,
                   0.2287190326573747, -0.6706431072549666, -0.9542988560958325, 0.9934282539217822, -0.4413205242922196, 0.5948607881795793, -0.9094225861504299, 0.19910986950445642, -0.4759307622696478, -0.7281384011796552, 0.43237836773452276, 0.9395915430189283, -0.7539961408782194, 0.9368052628737957, 0.8635654466316058, 0.06697780001389253, -0.2921084756019201, -0.8276004457423503, -0.5451704404833382, 0.8482249764355221, 0.5005551517725295, 0.19215038983355015, 0.8181540755516854, 0.9037076829471742, -0.023029831237656895, 0.09279069501469484, 0.7059169583855969, 0.6886263333632265, 0.07815535771027626, 0.7270962492074284, 0.9390119855348775, -0.18849394525842111, -0.9922337686681899, 0.3712201123798893, -0.2707637268985956, -0.9280719132146111, -0.3385407220656085, 0.8548213438561341, -0.6540438768318517, -0.9999743413510678, -0.7418912393590071, -0.3710994812992978, 0.7743292236727035, 0.893166229141477, -0.026525732935892332, -0.6346005664130556, 0.5221259549987838,
                   -0.12936618769565333, -0.4570663928372325, -0.8857411376539281, 0.0544600172603964, -0.6949858839716789, 0.7784089076373697, -0.21222799290103356, -0.5294979182401467, 0.13937747645496845, -0.264367267732899, 0.7987102346318231, -0.016482621621751292, -0.9377559745050452, -0.12858662869356308, 0.04783876323309433, 0.5253327694155361, 0.28770668304953717, -0.05342366181492597, 0.13808081817950768, 0.3686054297875325, 0.7759784226827044, 0.6587740052308735, -0.3132711021963399, 0.045029610986254465, 0.13001968707252498, -0.9696297582051583, -0.08625696566117402, -0.9651147621825942, 0.1167943277219039, 0.43012427218663896, -0.696684357888514, 0.7130957459359915, 0.7870914854225008, -0.22658874967769727, -0.9324751969431817, 0.8592301843977932, 0.5057420166776552, -0.07576492556976122, -0.6149638807415669, -0.32635138895905125, -0.1455510072873203, 0.6133027680580203, 0.9444561473297897, 0.6892568156236776, 0.33234182598504525, -0.710053603878456, 0.1910797839270162,
                   -0.7718998422163919, 0.054754800415363736, 0.24169349649129468, -0.14258525730322846, 0.12429470166659295, 0.23862078190456049, 0.043803279443084486, 0.8520072882277661, -0.6804435643144517, 0.16326222808294966, -0.05789900497306566, -0.6122662815837108, 0.6348570323580331, 0.5082516184440495, -0.874041766867804, 0.3498625177372652, -0.46850015084789565, -0.5045957661828568, -0.9857163647621243, 0.33583362370962666, -0.002825485314907894, -0.9001427983209997, -0.34409719455823806, -0.4571596593158853, 0.8273157041201629, -0.2544213904030135, -0.8632773288842799, 0.73594170179958, -0.4806444368782512, -0.6364578658440587, -0.5104090582134255, -0.10142743891398243, -0.24277366281898138, 0.4661918945268937, 0.018776989696857704, 0.8489599529942191, 0.0357347082861843, 0.9797146020142131, -0.820483990275136, -0.8999056047221821, 0.35919466698700364, 0.9760246849350405, -0.8059264092704084, 0.5475458000969036, -0.8540277106466532, 0.1592607302492408, -0.5606951583110764,
                   0.18855433516271414, -0.5196396436943158, -0.013019389529475678, -0.69330445980393, -0.992533326520703, 0.675467311009671, -0.3558660723927314, 0.7143907824225051, -0.5932706251002273, -0.25442985859172396, 0.7382816402284877, 0.741730520860242, -0.8807084308605948, -0.16635372294057382, -0.4525782601455761, -0.03967193664121127, 0.4074087474820287, 0.8995659492897483, -0.07282760961703127, -0.04825778360603272, -0.46551756809901335, 0.046690939452045166, 0.5719529164274937, 0.23031072769298433, 0.9907080436101694, 0.7493925531223595, -0.7748047818118411, 0.3307268624909492, -0.240053058138598, -0.05873364077040688, -0.3534931931220733, -0.8265422123665649, -0.9656233940036563, -0.5664128801586032, 0.6690997126009945, 0.9871179316775223, 0.21511586916863523, -0.5441869622724047, 0.5186878964245718, -0.9545723561298152, 0.7811605334390963, -0.7552859159039238, -0.2845520061432736, -0.732599117101389, -0.3343924561899678, 0.5559647082434802, 0.3435279397533739,
                   -0.6670759105051893, -0.852715827097196, 0.73180676369122, 0.24797905972136425, -0.6180990425109494, 0.9475416071156879, 0.5306546077326966, -0.19312204896342533, -0.4063435931301569, 0.21611149516945938, 0.30927889743292725, 0.32679435071186447, -0.01569373446702671, 0.39656448735090066, 0.3454728469268904, 0.82265300104796, 0.6133161628036348, 0.5810516777088159, -0.9011315705253122, -0.031572958148942565, 0.056488712087775284, 0.8956495856644908, -0.9701627336994807, 0.3429917664223108, 0.4811327485930046, 0.7017695005731925, -0.7688241580913535, 0.7458577547888239, 0.9501852903974319, 0.9557342422383694, -0.4254599464464146, -0.6796062329655566, 0.6855711196557073, -0.06223582139161832, 0.3333601792623304, -0.5024610554632216, 0.9371022170658685, 0.7432561261784509, 0.3162331507867968, 0.3608966984819939, -0.1682669786441675, -0.5766208419632508, -0.5704459913368967, 0.93449066954913, -0.8809954891790392, 0.46865155112596746, 0.2854143545078074,
                   0.10735624430436141, -0.9179747399476876, -0.5516954215458256, -0.01943854229153863, 0.43452998449581415, 0.04935314711316652, -0.9501218335858155, -0.5055229676632824, 0.8918907600208104, -0.07834317248571687, 0.5076357180584654, 0.28940032048468867, 0.8126968283733438, 0.9180528819176801, -0.5349754593492646, -0.3576141189098758, 0.6965538245677549, -0.6494722030164486, -0.31445038511078205, -7.481179663253457E-4, -0.6430613079970455, -0.3491271677193728, -0.2919306293273629, -0.7874590204403451, 0.9612965594799019, 0.10307211195229105, 0.5366620941119804, -0.08961987773733959, -0.0818836733861188, 0.7751023556352294, 0.7618523410158884, -0.5596070072020565, 0.2782827262964409, -0.0023827991626159672, -0.8708717863094353, 0.8425831002305277, -0.6566066412933684, 0.40300465815516295, -0.49768058844980767, 0.12737164711753457, 0.5116184280311495, -0.9831774762816268, -0.5864516862231199, -0.23026028537499443, 0.042072594434733546, 0.18463113610703674,
                   0.9481053183799266, 0.7776905561330907, 0.38155991692592384, -0.7792401497900143, -0.4182578691808365, -0.04698737029235822, 0.972940373682541, 0.5815580311030264, 0.7695151268887501, -0.9654100342905745, 0.052261140611505, -0.9058024614510314, -0.42054628693904994, 0.29927197815122986, -0.15781721474230248, 0.3131091333724898, -0.5290321842559378, 0.8925810494887025, 0.7621236795287232, 0.42965142125110245, 0.43704940471656073, -0.32869941465694463, -0.1990646135020948, -0.6733744969940094, 0.160964460895604, -0.33774672184416343, 0.035812123344346025, 0.3477649399471614, 0.58227147158268, 0.37634408978227407, 0.4315295383560245, -0.8566536543780936, 0.3236431192159046, -0.835792057227525, -0.48300342936062646, -0.4434193569846321, 0.7991100079008246, -0.44043137360542484, -0.688503015638013, 0.9314620869148544, 0.5059831149273148, -0.871319854899862, 0.9608855231212827, 0.4512659717411729, -0.0031445262871345747, -0.2710122662410883, 0.9443963656708352,
                   0.15790332060011747, -0.6652785605890579, -0.7225883055070323, 0.4256109033415514, -0.8441445066746074, 0.45339430880930265, 0.13842182262170022, 0.3465228386593764, -0.8295728042460604, 0.777043237385324, -0.9356625223996022, -0.7984711901033312, 0.10857252137269335, -0.30318980397012285, -0.7449478799869891, -0.9214353190041431, -0.4384356034767636, 0.05910970099776924, 0.5516480533845858, 0.5174967705184861, 0.2922919990593964, 0.6893537329406023, -0.9618676779679698, 0.011268186755596155, 0.6659889507568431, -0.8435151171197941, 0.3977903393383424, -0.43680780518311857, 0.7117076005137501, 0.477842414094545, 0.25541922886637414, -0.3858260830568785, -0.9571267096250573, -0.7294753545956665, -0.5660306350741882, 0.8665324443538849, -0.7812344634447446, -0.02544591476820668, 0.5638863550717632, -0.9435129485936218, 0.2947911725578318, 0.20479007675178806, 0.805713956509446, -0.4347395977516273, -0.7120725608190979, 0.03267593553871162, -0.4187183156314913,
                   -0.0908564176997897, -0.9252687429003048, -0.9444523477390396, 0.15606764595430778, -0.38824256479456665, -0.15717538277019494, -0.5196285110326542, -0.32490701916397, 0.7371357954392614, -0.36721719838086675, -0.8695809471885378, -0.17303127484431235, 0.45014663179171777, -0.2641332420313178, -0.7224713408871213, -0.3045171162556999, -0.6288415630292163, -0.24594225481292775, 0.5092426057834976, -0.44461924641776296, 0.2047480757404201, 0.9823670541052663, -0.5407494611115453, -0.6212441282999057, 0.14763929482957017, -0.5605903563139985, 0.16120939446358395, 0.6407370685515674, -0.08105035094518298, -0.6492537501731859, 0.03015209380340056, 0.647920262120093, -0.09378436289102265, 0.1805136231530411, 0.032336075939683795, -0.9397392009543846, -0.11644474219554302, -0.5896211778328158, -0.5264749291659006, 0.5369072870052272, -0.9662804635443198, -0.8191311275967745, 0.017337259642099756, -0.4912764993509495, -0.40632720180880444, -0.3613574454967383,
                   -0.9605666172772971, 0.9650101577085832, 0.3437346402092156, -0.6257133658085481, -0.7700402005024221, 0.8392050598074328, -0.6267948584128398, -0.3112646705681843, -0.5241983477780277, -0.9118952128577793, -0.9192134041699112, 0.5924259165640555, -0.5141585802807052, -0.11858343062190979, 0.20600572596494016, 0.41982071075195226, -0.6334401831286296, 0.4532159314814097, -0.16177026395901395, -0.2670602802543143, -0.5880438684916947, -0.8858504163703613, 0.8127620473232477, -0.2089348565735536, 0.7409052122239823, -0.31700992291506624, 0.6788925759515723, 0.27979383629967347, 0.09670959084602271, 0.1684136483782106, -0.4873311677565604, 0.22112936942723804, 0.11774603248719684, -0.08336177625764307, -0.7524553181120652, 0.02732151164588248, -0.3052544119453924, -0.36217509653129043, 0.908954021267375, -0.48532372260922996, -0.35061260721837373, 0.7339108883446996, 0.16424730028437873, 0.3595752329126918, 0.9005275494617355, 0.17280811522514972, -0.06268028323897412,
                   -0.04254187328484149, -0.47147654015247786, 0.7886677499325225, -0.5123498575541392, 0.9365396335616778, -0.29524311740344933, 0.33517471256098186, 0.3163606736113538, 0.6815449914958425, -0.6774026030561922, -0.12139524804381385, 0.5372972350794112, 0.13876658804655007, 0.34769033161973906, -0.8103607709230511, 0.9390227099382855, 0.9022576778560889, 0.3995923662515657, -0.3222993494313928, -0.13599631894013542, 0.12199623062516429, -0.1165801417542578, 0.10883676954085342, 0.2441638396867205, 0.12836700477248564, 0.10931520041665199, 0.857945899291086, 0.6573665234503123, -0.16667471448944515, 0.2691763058892145, 0.7684039775825899, -0.892719659632371, -0.05287778978652202, 0.8842363590723035, 0.8269163804131701, -0.62052815963534, -0.15951384064225005, 0.3937205262065149, 0.8086861378999828, -0.7105396060996003, 0.48567122205726676, -0.07605259055261926, 0.807754113507904, -0.08613946646419635, -0.6742012057800857, 0.9588835088864005, -0.8603677211388274,
                   0.9836277969475615, -0.849565472572996, -0.37270036800004713, -0.5118820270359259, 0.018815370310486745, 0.09635764418582071, -0.3716812680752819, -0.6345583610440044, -0.5028619938349952, -0.25400380052652816, -0.5637144286728484, 0.964666381004027, -0.6259679749435936, 0.9620168318146387, 0.35752867556860757, 0.3055236759064788, -0.39021043389820775, 0.575073561923434, 0.2557263137271344, -0.7903651260217845, -0.9902637226287145, 0.3327353856440429, -0.6807715957593659, -0.9116662847098058, 0.6463127828413335, 0.6211287374994658, -0.8871730773883013, -0.21392905946127172, 0.40730985975973844, -0.6131544894204959, 0.3381085321573909, 0.7946474158644001, -0.19760138338779232, -0.5848157689182933, 0.22066261943028342, 0.03828829932605826, -0.9395121576754599, 0.6618061947721143, -0.750939895576044, -0.8731641644796795, -0.23667000081288725, 0.6884051827697244, -0.8903422014228033, -0.515896076257802, -0.21381020618160784, -0.5809955359571197, -0.9070761267684748,
                   -0.9192955610317315, -0.8457217338238343, 0.4120103076044934, 0.9911213431113184, 0.6354191022758076, -0.42595303414713914, -0.36583280159155507, 0.5638263966314692, -0.9035423030274865, 0.8613270143411613, 0.32607521509111814, -0.9831006984200101, 0.9737084230978283, 0.2636885293721247, 0.964945926074299, 0.6221658418444929, 0.665352305312211, -0.23771373224789327, 0.5587890374975322, 0.5992089739216271, -0.6930219541009037, -0.16115102392224379, -0.45276351872968035, 0.6873093096305372, -0.13139865616044988, -0.09364122882567827, -0.562271835622062, 0.4547156008492228, 0.2347561215082039, -0.37264056920988686, -0.02820300072285531, 0.32406496400944906, -0.1749079091129473, -0.3182782733589038, -0.9052911383375946, -0.4630848466536932, 0.9476785551039917, -0.8039149082593096, 0.2927261201170561, 0.8369906795499371, -0.47552046972688444, 0.010085776699033655, -0.03421836601176387, -0.4791150617818265, -0.9117687996772641, -0.770128499922474, 0.4193638162754394,
                   -0.3133974945820066, -0.9777815401271748, -0.892705822247958, -0.21207048905413517, -0.9955350648153425, 0.8992123208243479, -0.8164378016952809, 0.7431779527228677, -0.858455296894211, -0.5269033586655103, -0.1509754518636126, -0.8059669585909, 0.08993263943971463, -0.6135229228300991, 0.9961501064091249, 0.3511984134585928, -0.9192155366828738, 0.9798314677752422, 0.280655463800519, 0.5076644001398931, 0.5102330287630608, 0.7225268175923552, 0.6561252200987651, 0.6775616793280772, 0.3598932526455445, -0.2667076642835797, 0.3103607183557531, 0.8392730481107291, 0.9638271592660268, -0.38249451814423185, -0.09291899539049542, 0.9447283022829509, -0.3445220895776653, 0.4209117196327159, 0.12449377369126857, -0.914674425747753, -0.17777891663730983, 0.884165852081368, 0.8826420736033731, 0.2459355498884246, -0.7981766281352543, 0.3798472396839314, 0.3643243983556326, 0.9319620557610546, -0.12576177848059422, 0.9233679746369434, 0.9597760183533977, -0.37955184301513945,
                   -0.666190055813457, 0.47145716966037154, 0.6031906010435959, 0.46473146036721835, 0.2710971346935427, -0.6433277365494905, -0.38044911104643964, -0.02683067580191989, 0.2501675929464915, 0.15365166080954307, -0.8368718008829124, -0.5140310805652895, -0.6636423050959035, -0.4674099590216225, -0.395597957885063, 0.1901743786644694, -0.08887074367391001, 0.7276830077437042, -0.18996004769259134, 0.14137672474204122, 0.40479770515311, -0.8669465837590229, -0.8390300384052443, 0.30173017543861835, -0.5154534166147164, -0.7640636470725735, -0.6335322212807073, 0.9573722205430661, -0.6457171083652302, 0.606667747894881, 0.6575719164532432, -0.9556381050096701, 0.2545297712607715, -0.012715978165741326, 0.7810298491733823, 0.8907647864819042, 0.9946923343522713, -0.34469124433284626, -0.6346299086348748, -0.32646931221223285, 0.4066743162103559, -0.8134339673389321, 0.6935848104090843, 0.4923240845841559, -0.022037666414876478, 0.5006267383589269, 0.6922981452098096,
                   0.37848362765848154, 0.6039373220434219, 0.6447589147067005, -0.9648131743681716, -0.2141970867122327, 0.3933106442165386, -0.8429795167150296, 0.6588095027193757, 0.15083955903553536, -0.23277434069964675, -0.953604575992973, 0.7018923865256672, 0.9639762903927691, 0.7117040902647374, 0.43764378729215636, 0.9045089628793694, -0.5415277151206945, 0.9747584846105264, -0.2492020689170611, -0.15851737008127742, 0.5032596757595595, -0.7563681656454779, 0.9200061162411381, 0.014731092774297405, -0.4237453643213054, -0.10573327348393136, 0.6036876074573685, -0.20783694800442398, -0.20153086717330582, -0.2257534156839478, 0.5977310532510436, -0.5438730243283383, 0.9252248216188561, 0.6620920024875403, -0.2642677694986182, 0.21230332506398475, -0.08071495662317263, -0.46006319859128597, 0.4450339582833742, 0.91032243930972, -0.7282613469773589, 0.6232474095059553, -0.1754050432171681, -0.5597205475032745, 0.5899828749763467, 0.21039176022334782, -0.3795175851408954,
                   0.29471021235132233, -0.8618306559227438, -0.1534972725568955, -0.2029536380630126, 0.623481190373369, -0.0060071907675596226, -0.9597773709311488, 0.5366008715840118, 0.25943801738191974, 0.2767174865755464, -0.17144341807727392, -0.5522973216221005, 0.6555470161462695, 0.50485117895017, -0.14579877281867426, 0.5218828307588108, -0.5503441916266809, -0.4421055564970111, 0.30135800158635573, 0.9509526629803824, -0.539111157301327, -0.05489428395160867, 0.030179791474560513, -0.6982273675009418, -0.6779158539098846, -0.8747594875965619, -0.44336015805940154, -0.2573651511403663, -0.33443033290993673, 0.47473763282289827, 0.14375402011762772, 0.02517960549611331, -0.19503381079652127, 0.16784121266283103, 0.2017266448201036, -0.25197112002293376, -0.8064040201723817, -0.28933717846230933, -0.5612686005449632, 0.06308679675689222, -0.4095336772306504, 0.1622039854170516, 0.7606795695557589, 0.4327225783912567, -0.17231221876781144, -0.8211255948287928,
                   0.787467930921411, 0.3058168577386058, -0.6514225899026, -0.06494251696857045, 0.03484032413245397, -0.3470825256928989, -0.7264112689077549, -0.2992102859400385, -0.6948643928071585, 0.1855179399661535, 0.6515992277318166, -0.6013634137757427, -0.36316229899859565, 0.781735298108269, 0.7669786374242815, -0.1846364810649319, -0.718017422964853, 0.1504316560077188, -0.10174934928444856, -0.6899851659565153, 0.6121176176512679, -0.911911939782077, 0.7620512724691086, -0.161469181388479, 0.5976498531084746, -0.18249438383047112, -0.9735213007979404, 0.9060476282863885, 0.562060907352973, 0.844673964653051, -0.2591805911818539, 0.6615406700989999, 0.2816051461828035, 0.5299134759538415, -0.6959745212334354, -0.2345120875274571, 0.1827501327115213, -0.8583379943652039, -0.3177898616950454, 0.8897446425161288, -0.6162418282378976, -0.4539554572948832, 0.985817123858675, 0.7498376124540793, 0.633448222027049, 0.8826272398003945, 0.2300126198758483, 0.4234015166874163,
                   -0.18068128707371844, -0.4134744066336209, 0.12723617907974272, -0.6255035655345693, -0.35114817467502113, -0.060700222905803125, 0.09874085713884351, -0.9179388289451338, 0.3519605485923132, -0.034162437281664904, 0.342340380337373, -0.5382284985883508, 0.8091812214820695, -0.0583102637968711, 0.756474173345417, -0.6259097258387081, -0.6206467388394465, 0.03508923248260931, -0.2039097310617195, -0.5726905320990254, 0.8821701054418691, 0.6801505072594867, -0.0762625966409558, -0.9870982144358318, 0.6851250774511284, 0.16251092338837392, 0.2196479301196439, 0.05151419248938627, 0.6510166846150278, 0.5959431190193869, 0.9274839323009942, -0.718548771836244, 0.821318622785965, -0.021769873131366246, -0.7621903383908137, -0.9124876920026037, -0.8576400650013833, 0.3368641277455784, 0.44924367493314965, -0.1534379830735979, -0.7145975708030556, 0.10989209338661143, -0.20236497237309492, 0.5572288295936627, -0.33165810523305495, 0.8544994524738982, -0.2935014854792759,
                   -0.8753576251469268, -0.8440994008436009, 0.13568753757590502, 0.9873342843548134, -0.912863979325816, 0.5968202183970539, 0.019666409979308908, 0.6063479359845756, -0.1996436255062357, 0.35070303367675937, -0.01864992513648822, -0.9655222799276149, 0.1352373135049536, 0.26379206279646983, 0.23260728309163503, 0.9105269399523133, -0.4001883492409941, 0.17211908811156618, 0.48768595209074905, -0.22741186397515878, -0.5391507486629081, -0.05790732086978734, 0.004495826310744189, 0.9508393748492492, -0.08017401939760149, 0.3027365131616888, -0.6248016261232276, 0.36518017513474743, -0.3112564381132812, -0.719569219807561, 0.9868200274701306, 0.9340684905163297, -0.5851642031339566, -0.08385105324967634, -0.6676308786498242, 0.5763079779437978, 0.31921252260368327, -0.873176613532537, 0.4252034085671572, 0.37542756737403415, 0.143008489136091, -0.9333862602352017, -0.5458398922799146, 0.3766844407615799, 0.03561908687147697, -0.8495074663500806, -0.485773082089787,
                   -0.3003867260144366, 0.3731215738014335, -0.9782967825840092, -0.24270337146092946, -0.8255657144349033, 0.6609953636961265, 0.17628652986765037, 0.6298149036950982, 0.5485239662883796, 0.6537706904708966, 0.7020114125331256, 0.9988960091117307, 0.4595430096317823, 0.6152511137414869, -0.3543121079426579, 0.915746461201671, -0.1652946719723296, 0.24646955164543605, 0.8919790894226911, 0.5250029449652867, 0.1510651016415201, 0.22455539233273125, -0.8815290075368047, 0.20085562882191188, -0.9204135849730049, -0.47579648032136745, 0.3858991989173757, -0.6244052539022868, -0.5638466014846917, -0.26554671459247725, -0.7709287384134, -0.27651754473266643, -0.9244321677225704, 0.4370003402533462, 0.7331659557727885, 0.37261901771611994, -0.6100279988493986, -0.6217194736900393, 0.011729231876833834, 0.6397374743937372, 0.41102550881181577, -0.3539204873072772, 0.7468290680902594, -0.5823743082201027, -0.8344441994872946, -0.6475612410293554, -0.9915290146313087,
                   -0.20785768965417528, 0.897291307607121, -0.5718565163932914, 0.4395165036385762, 0.7255564636019722, 0.04267865971323026, 0.0500932071878315, 0.0739886322837755, 0.7152086663375969, -0.5633257217917857, -0.04862709584851066, 0.5507876058750181, -0.7787457816250551, 0.12479284159897586, -0.5951440225109721, 0.29148201127232465, 0.35451819625363545, 0.3705840724381282, -0.975311503346926, 0.32997975082344766, -0.17474597333950337, 0.973102129813608, 0.6960172768286288, 0.9135255214343709, 0.4758660637301615, 0.8468196305335398, -0.0781579162031727, 0.7386103427086572, -0.14458032048317415, -0.03316821580153251, 0.6668866252115357, 0.7420211724889731, -0.7589696548257421, 0.24508520941350298, 0.3818989238028174, 0.7363970861849958, -0.7290132984474247, 0.6787563330822306, 0.3078074024808126, 0.8613015695152277, 0.9267298578627354, -0.8002034227019006, 0.387674634945683, 0.8149589142037541, -0.014424410935413023, 0.0550496695291105, -0.8445936525766868,
                   0.13844464879069807, 0.26333306131734546, 0.2592399083210435, 0.8236883312982433, 0.4432602262352743, -0.024262760271930084, -0.31148182902674404, 0.4596410653092733, -0.5951373258430954, 0.3698805683943365, 0.720145202975103, -0.38480503006450695, 0.5888605625705212, 0.41159260736298453, 0.9742805195474933, -0.9297920482186504, 0.7332284280246137, -0.23801675626719487, 0.6767377962144954, 0.4498592914374262, 0.9790325035285314, 0.5361303810880489, 0.4369561793304466, 0.39387276964523044, 0.7238005253032753, 0.4091590667497682, -0.17667035505732787, -0.316365899392409, 0.2658102918235088, -0.4802763355962705, 0.01842517964006185, -0.7257130727727703, -0.7857891709291682, -0.6818288310399183, -0.9504868926565397, -0.46759543110875645, -0.24096831620408388, -0.9835706507925037, -0.3847783476641158, 0.14585114350802852, -0.6732237312027198, 0.6189286283848858, -0.6329800064312558, 0.15096233359012357, -0.10856472555840324, 0.6486320096488853, -0.3453762724453626,
                   0.31318986162236095, -0.7326023921954676, -0.37048585553975966, -0.7926921597185703, -0.299129237566661, -0.9201238553300211, -0.8534050471547896, -0.001497779059709936, 0.707798200241931, 0.7901072372585405, 0.8768995056573727, -0.37106352831105616, 0.008784368839833467, -0.8532779689780854, 0.3694813391502332, -0.036009596131231225, 0.9734246703209137, -0.25867030923435785, 0.5499123622785969, -0.5262028577072844, -0.08321510617866101, -0.03450714862295423, -0.06559289997420459, 0.7500379602807645, 0.23175119340158523, 0.5979156306657463, 0.3610163342274142, -0.9887437339935536, 0.9262965481229695, 0.4948058315847288, 0.6800888493307926, 0.05891081067517767, -0.6200659487092175, -0.5869865081673638, 0.29346732546692267, -0.8907226469280389, 0.23714279643990488, -0.13531148479454647, -0.3083509921841925, -0.7982999888152222, -0.7284328308057078, 0.0838400854169723, 0.2104078812499499, -0.5284174813244802, 0.783307162406158, 0.6793228698722575, 0.11314042766695098,
                   -0.5230980999750574, -0.3673688613975503, -0.6445866202332076, 0.8024629515733561, -0.02008342625422843, 0.36696068951361926, -0.7997175404653094, -0.5256463839442602, -0.8888682097337786, 0.06398978434539893, 0.47430955072263914, 0.8503989416434072, -0.3195236119604272, 0.49231870057014304, -0.42567510228296834, 0.7584684422072046, 0.551034379599386, 0.9819974287915554, -0.09349303282085408, 0.2794088316593748, 0.45356747801441166, 8.888203026111707E-4, -0.23019712274334436, 0.9371129352047411, 0.39545247568913533, 0.8487785831019155, 0.0333796958485304, 0.9815854702196845, 0.9055510013319785, -0.8903267833618669, -0.6142440530492292, 0.19939220443330963, 0.7569174313025415, 0.54374353143814, -0.7600972410158475, -0.11685368460391454, 0.522142946578849, -0.04611233398075942, -0.0022551446375336237, -0.39855072968437466, -0.6584769184765069, 0.22790110590338264, -0.6908681070876419, -0.7670286447488643, 0.6221284467290833, 0.23575942971076103, -0.7955077733614611,
                   -0.636879267065178, 0.03624617490787263, 0.5525324608404942, 0.5370203242528311, -0.4109830098938452, -0.8494460860944621, -0.9190331406213827, 0.03939110546797231, 0.5382496175411946, 0.2927752846058218, 0.5637871591677384, -0.5183653886789048, -0.520831512496551, 0.8504587765752802, 0.7042830616249376, -0.08905598155354832, -0.7147819287832411, 0.09691147442789161, -0.8452349335477352, 0.7608329169595791, 0.21309350904789914, 0.8522100978811962, -0.014013653955366134, -0.26380440600693245, -0.5357379142175343, -0.18739766664759916, -0.5833543514764656, 0.8124672983548875, 0.821658798531073, 0.21950196055329996, 0.5605993245518335, 0.4312202704975019, -0.21186885526077615, 0.6896724001310635, -0.25957181677189367, 0.031172858547048987, 0.6004731712728848, 0.4910827033867451, 0.5195201763028641, 0.4227289842100357, 0.2526549842287291, 0.32431267927497043, 0.020844169203598106, 0.1958068502056689, -0.7658810910148766, 0.7572719394335332, -0.6708923073555613,
                   0.3448555469888075, -0.926219887514486, -0.1386090859276785, -0.1859944236163229, 0.6015989521521814, 0.9776338805280085, -0.6896699018317944, 0.06578285645452042, -0.4148128134380482, -0.1423614987163644, 0.22764622689082437, -0.19964928714978902, 0.8110765989646764, -0.1281925087581055, 0.40774342015231646, -0.4302902525037562, -0.15590026409837732, -0.7697888614471022, -0.8151705503105531, 0.40445062258754394, 0.4378268076619123, -0.5311475913037478, -0.8125142393500167, -0.38138208279523256,};
    weights[1] = init_matrix_from_array(89, 74, a1);
    double b1[] = {0.4617563814065817, -0.17983837701559668, -0.5845703173805659, -0.33456588808097765, 0.9355118188482414, -0.9877656354684774, 0.9274095940464153, 0.8797307775638197, 0.8943898353263877, 0.8741642977919393, -0.2056513156305888, -0.3049639415937795, -0.41188593599192647, 0.012967254652470173, -0.7680658239346845, 0.5410717601583555, 0.31978541738683997, -0.6865062188603075, -0.24359590935788944, -0.7204746341924977, 0.38989595920498377, 0.6104555429474274, -0.9899496480150949, 0.04627031157666606, 0.4879689724746332, -0.7159545935681477, -0.03654339684880403, 0.08910961778734738, 0.154200522748553, -0.5901729084828768, 0.2467276212633316, -0.6305858194441027, -0.9786311780355423, -0.6779133532565955, -0.6438903067256505, 0.08079412956759291, 0.9476680198374003, -0.5091468928374088, -0.21095818741155647, -0.5647957501846304, -0.13598624021292172, -0.5336884104973332, 0.7798175033107542, -0.9233462133614492, 0.18475848772587744, 0.31034720625883283, -0.7603219153916208,
                   0.3049534137919063, 0.9686456013385898, -0.5865249852789853, -0.2507009994120597, -0.07330042486104849, -0.3327795776506606, -0.11357509936546184, 0.008271133409147868, 0.9979616295358575, 0.2608097278457078, 0.8191680711780485, 0.015319832139896183, -0.017097588876774816, -0.14243164250759865, -0.3838141695830404, 0.43464480324445387, 0.9248291854674928, -0.5810145330728713, -0.6547023581719895, 0.09790503386131122, 0.11093676417592313, 0.17563165780266687, 0.5783813156701916, 0.39801324486896106, -0.590728895588629, -0.49018109740036864, 0.5558327190390557, -0.5499552279832491, 0.9662063696674061, 0.607237212030854, 0.6726298955126573, -0.6737929452233706, 0.27499485971881277, -0.9824791653625367, 0.2623797766230498, -0.584044041914755, 0.7607145698140847, 0.41408845210793044, 0.44674329243160504, 0.018985839340738497, 0.9742042761925997, -0.6935075343674779,};
    biases[1] = init_matrix_from_array(89, 1, b1);
    double a2[] = {0.4617563814065817, -0.17983837701559668, -0.5845703173805659, -0.33456588808097765, 0.9355118188482414, -0.9877656354684774, 0.9274095940464153, 0.8797307775638197, 0.8943898353263877, 0.8741642977919393, -0.2056513156305888, -0.3049639415937795, -0.41188593599192647, 0.012967254652470173, -0.7680658239346845, 0.5410717601583555, 0.31978541738683997, -0.6865062188603075, -0.24359590935788944, -0.7204746341924977, 0.38989595920498377, 0.6104555429474274, -0.9899496480150949, 0.04627031157666606, 0.4879689724746332, -0.7159545935681477, -0.03654339684880403, 0.08910961778734738, 0.154200522748553, -0.5901729084828768, 0.2467276212633316, -0.6305858194441027, -0.9786311780355423, -0.6779133532565955, -0.6438903067256505, 0.08079412956759291, 0.9476680198374003, -0.5091468928374088, -0.21095818741155647, -0.5647957501846304, -0.13598624021292172, -0.5336884104973332, 0.7798175033107542, -0.9233462133614492, 0.18475848772587744, 0.31034720625883283, -0.7603219153916208,
                   0.3049534137919063, 0.9686456013385898, -0.5865249852789853, -0.2507009994120597, -0.07330042486104849, -0.3327795776506606, -0.11357509936546184, 0.008271133409147868, 0.9979616295358575, 0.2608097278457078, 0.8191680711780485, 0.015319832139896183, -0.017097588876774816, -0.14243164250759865, -0.3838141695830404, 0.43464480324445387, 0.9248291854674928, -0.5810145330728713, -0.6547023581719895, 0.09790503386131122, 0.11093676417592313, 0.17563165780266687, 0.5783813156701916, 0.39801324486896106, -0.590728895588629, -0.49018109740036864, 0.5558327190390557, -0.5499552279832491, 0.9662063696674061, 0.607237212030854, 0.6726298955126573, -0.6737929452233706, 0.27499485971881277, -0.9824791653625367, 0.2623797766230498, -0.584044041914755, 0.7607145698140847, 0.41408845210793044, 0.44674329243160504, 0.018985839340738497, 0.9742042761925997, -0.6935075343674779, 0.4211495502966689, 0.6479617509063196, -0.7526575429941516, 0.23390277349257982, -0.03627528947880365,
                   -0.8004994373485674, 0.23429177972286364, -0.9437592540976656, 0.18287821190927622, -0.8539479595723121, -0.8519041454533085, 0.3598708403820501, -0.43999676843466795, 0.2079673753671345, 0.6992816140285527, -0.2923534344590508, 0.19726497654931419, -0.41581317728749156, 0.7528853066489665, 0.48849924027210045, -0.15353454226742125, -0.24838358251388248, -0.45741334965056835, 0.459647857870199, 0.7734888447426316, -0.9125946034063506, -0.0845996338529802, 0.724899509523101, 0.8038479114846022, -0.8750230801744465, 0.5276409763493284, -0.1445998371634829, 0.9983668615811874, -0.8761963954427474, 0.17264126398586965, 0.13458614180741013, 0.3240519226107552, 0.8145623802363859, -0.5070287373953819, -0.5030224658253748, 0.41371666950135744, -0.3245581454720592, 0.21467015792151267, -0.3691544270850342, -0.951055740329511, -0.09798299476910088, -0.26173136899734595, -0.0723244806805814, -0.6455145763740981, 0.3522463090507362, 0.43114496998271745, 0.48673924553053105,
                   0.8582052208550688, 0.17978705894087765, 0.28802457476710486, -0.7790202931650292, 0.5492082189178475, -0.3870344237893202, 0.7611973395786324, 0.26080288156998255, 0.857787426649109, -0.6066211981784508, 0.5092924905221334, -0.6847387723847258, -0.9156426320965521, -0.24275424927878264, 0.5732184915636163, 0.42550433553218703, -0.7794527769815014, 0.5242661394785624, -0.9398623282422005, -0.3667370420546159, 0.33414670639259225, -0.1471139051811421, -0.19888567737380747, 0.48752309522700377, -0.2693885244307648, 0.527509798143645, -0.3546616179089326, 0.025437646764806088, -0.04264825558681129, -0.7376623737501162, -0.9364070460294214, -0.11108677780932652, 0.38304242336296523, 0.43815137069360843, 0.5559005447604162, -0.42379856790026804, 0.5627094672588622, 0.9325079013644189, 0.9930469030238771, -0.5217871273450587, -0.10963747653244882, -0.7395561439767482, -0.5310518249898872, 0.3121648779767989, 0.14445918501177846, 0.11193326222583289, -0.4994074888102318,
                   0.840562385096266, 0.4475024168644697, 0.553981931358702, -0.5154477144827585, -0.6485960909997945, -0.11041198807748454, 0.6593824466983986, 0.20382927212011515, 0.2758937921094, -0.39326571048193704, -0.8692059577422617, 0.8760610024651603, -0.46716658534520294, -0.5416572832644959, -0.39204712487533855, -0.25611897302693687, 0.7142540853584027, 0.6141527310483261, -0.22254144849415747, -0.46950707793809965, -0.9352542931981471, 0.3395705269641569, -0.8512772474037791, 0.2182821628673659, -0.6426752811796594, -0.9407335772053, 0.4163076558810881, -0.44444345058469303, -0.7684231144477478, 0.3824518650005857, -0.3019785521812086, -0.5437289954810616, -0.6648000080008807, 0.1892622659742289, 0.6197796579677197, 0.02985828169711513, -0.3020372245568883, -0.8938130714882238, 0.2123804084067331, 0.3630529243217582, 0.1674866344119328, -0.2604950014005061, 0.8919840619507564, 0.3728611665601458, -0.5508522595776915, -0.11080113051030072, 0.41522708850908274,
                   -0.9404045646350769, 0.3177872479269317, -0.3681649705532324, 0.6315404829028237, -0.6888427297301514, -0.09319152829934141, -0.36058366622658444, 0.09938505757310678, 0.04451849300256239, -0.3693699674372739, 0.39222576450162183, -0.8102451896845759, 0.9176161100317948, -0.7406448221923483, -0.952802763757856, -0.8305583514396848, -0.8953584308861684, -0.060092800991414785, 0.6393647783990011, 0.9381944945543601, -0.3000286259712659, -0.4542271181418751, -0.977542944962007, 0.5880466869499925, -0.7992492604291472, 0.6378848064003326, 0.49571961454667535, -0.24027382652589435, -0.7845912736244673, 0.5320655384878354, -0.5824367631047411, 0.8902041803858236, -0.9448522188144981, 0.4938463541049951, 0.2726517766875447, 0.8770772247487733, -0.8122829453839511, 0.9193644982735831, 0.6375592204524125, -0.5703801304492622, 0.6675442057285212, -0.4885674170728609, 0.28114096218840445, 0.4329845654626039, 0.6035657632551754, -0.07723458209616907, -0.760720375079807,
                   0.021166654236003835, 0.485862421006954, -0.8862599023879809, 0.11968219083981557, -0.26400725354637866, 0.7644703964442963, 0.759954413758579, -0.8790802317447803, -0.8564689514905641, -0.5414934855402953, -0.554758540226437, 0.2909182221813966, -0.4140360075988738, 0.7710093572261563, 0.08868715483308742, 0.40362185197409084, -0.6338976041733178, -0.3053344801905238, -0.9204112478252877, -0.5014300699616621, 0.593362607679703, 0.41585420017249497, -0.0678168917254871, 0.6066398889726063, -0.8358319871083821, -0.04078885930975695, 0.9223636956218182, -0.5237730174200228, -0.09362361593075197, -0.03782443553414039, -0.8685032291790877, -0.7823532007301461, 0.7864765742550983, 0.7287528662943221, 0.5978363044994466, -0.3004387865358291, -0.740062733569075, 0.7839879246135268, -0.1360703359664024, -0.7402549205104585, 0.33923187827797374, 0.03541276588258202, 0.10879520691077893, 0.8335257327328773, -0.2392958342417062, -0.6550048188585313, -0.7334556545329904,
                   -0.043255722716780465, 0.7505799471975545, -0.17511741572253237, 0.058511332327899845, 0.26610137326114214, -0.6779300180015784, -0.5075003126890016, 0.7038436325218125, -0.9017322953121019, 0.7796517360338735, -0.9446747387303911, 0.22389816722296274, -0.4198460762437439, -0.8367444080582236, 0.5429373153513242, -0.28132271049682855, 0.20625050403561307, -0.26577579064015633, 0.263684137807622, 0.7689307137181995, 0.31496408683598087, 0.2865532371047357, -0.8643989460574832, 0.20543900210088784, 0.10439199214576789, 0.06106168814262758, 0.8259616291751632, 0.8873857705804806, 0.8689926148048137, -0.6532178585906796, -0.6688846916915596, 0.8446155768889705, 0.9921245603305735, 0.47300071741863303, 0.6380300766818798, 0.16396586531348323, -0.10455643996500963, -0.1729822345961347, -0.06877578578197374, 0.3377512645671197, -0.3753384059915801, 0.7768403719106152, 0.4435373478338622, 0.4428696503882732, 0.04218769023983593, 0.9346589722418666, 0.49645519848127484,
                   0.4818898150939612, 0.7006870619845313, 0.06610090762259224, -0.46453011839092007, 0.4928710774641969, 0.3876087522264322, -0.868106217994175, 0.03822173798830253, 0.622556343870637, 0.05924937330418878, 0.4427129596431163, 0.05793522726276468, 0.2579697305072117, 0.4239353347697836, -0.285372502144996, 0.8619265945813017, -0.47565821896263705, -0.21257848901206233, 0.9316998514658994, -0.36527315220597223, -0.017325085220385228, 0.9396202840385481, 0.35393508757797476, 0.5376287774531514, -0.8018474656154688, -0.1597096167192229, 0.7529158270059466, 0.5512336090277055, 9.864339512639653E-4, -0.3476695330819941, -0.9858995135582167, -0.5251636700735824, -0.27581265182058545, -0.7251620171299673, -0.3860980191413501, 0.21625679994852565, -0.521866405676193, -0.055890195764195516, -0.8386945947268156, 0.07798773037047702, 0.609704658962309, 0.6992764155353963, -0.812076482456136, 0.13029646449166, 0.894135919760972, -0.4497228973625478, -0.32297012246989687,
                   -0.040815831976717565, 0.720642058084145, 0.039399522796289776, 0.44124938512551726, -0.913129915858155, -0.44175268996846273, 0.5536220392507769, -0.6097726471838123, 0.1847354089696971, -0.6738490912803419, 0.22351377096764224, -0.026261683719869255, 0.8334262707136533, -0.056653037754744284, 0.535461587845538, 0.6423367941140314, 0.03208146833942638, -0.7048910956801153, -0.11248933074675826, -0.9632074244754596, 0.9184153016710372, -0.8706614056267428, -0.7648809961689669, 0.7991938560330296, 0.2336529082846095, -0.016328097395461816, 0.7016432551773015, -0.6072576421085734, -0.048920470112241876, 0.1446212756320453, -0.4306116130452171, -0.8760956088910088, -0.8157679476940964, -0.14707094114558816, -0.5794309261038282, 0.6797237917918408, -0.9994739186478907, 0.4946937454493163, 0.3635469837486227, 0.9247420187247666, -0.9116223178326195, -0.6442774520991257, -0.7417773483731642, 0.4461120161383725, -0.639355738230065, -0.3330444891104416, 0.645477965463219,
                   -0.607867100619121, -0.47552435792047776, -0.8683051296075128, -0.31092101923671067, 0.518572955020401, -0.5925698995294806, -0.09923385767834181, -0.14553411175917796, 0.03878964260051876, 0.4019781697103719, -0.9503736708631689, 0.27737171727201826, -0.1322048145007273, -0.22471932391323746, -0.9881736596714608, 0.8162276457730837, -0.8429426429125628, 0.7883938571634896, 0.13612503034764178, 0.6358275930365096, 0.6359656857276144, 0.5788206673153771, 0.5599425363677661, 0.8357412255018737, 0.20753402474083948, 0.5825913148025064, -0.07026813851511182, -0.8845201354264969, 0.4900506056408893, 0.9068237285250427, -0.8507622193807312, 0.7594347060808979, 0.5530278048144603, 0.5109787241829884, 0.299502934628346, -0.671876164389478, 0.9649444748963754, -0.685209149603812, -0.8501284115000383, 0.35829841810365926, -0.5480863862633054, 0.42312192083773237, -0.732867710200992, 0.1029759189417172, -0.03944995170988741, 0.5427486778893642, -0.36966902225923626,
                   0.8834819756586672, 0.724951899803354, 0.3633634166936208, 0.8716769746398525, 0.9790996773829199, -0.9431484672252473, 0.18482999042936932, 0.7893550843837316, 0.14593968186272144, 0.009631615533212345, -0.6049950120350034, -0.3840529882517494, -0.16483916921641995, 0.8110429045705321, -0.4628012347408834, -0.3234434120273826, -0.028314028770306532, -0.9838998570148614, -0.3463839554134107, 0.5336234876586188, -0.24192615064738665, -0.29922503053686356, -0.6491171813880459, -0.5091986342809189, 0.3842236022640888, -0.3012573036323791, -0.39396939925394836, 0.4161658591572903, 0.7521027312034498, 0.8775829790888101, -0.2839243739074371, -0.7545902003670362, -0.6342924404621659, -0.24252641528954855, 0.9915953158988744, 0.23857250068975144, 0.38997815224162835, -0.6054429891121651, -0.19552901253261035, -0.9107583518628244, -0.45159332171499855, -0.6171982155764295, 0.42223078507581024, -0.6769441581380431, 0.00759419724783994, -0.9276815108513372,
                   -0.30380796705337954, -0.9272725855827195, -0.3119747495816658, -0.34126621434187476, -0.8821308709891407, 0.2796953588366484, -0.4561538698771319, 0.6673646226901524, -0.9821226430098648, -0.3520915640492801, -0.7295519295850796, -0.4624245380598375, 0.7538295413593057, 0.2631303795740023, 0.6271725760600964, -0.49315638493645064, -0.3815126586586235, 0.6915480463204711, -0.8965294905773702, 0.8006959120908563, -0.10330448591388719, 0.5843555047815325, 0.010364393175903208, -0.7461770556376985, 0.00939227612983995, 0.34037685332194423, -0.754923624319918, 0.9193159924791634, 0.005254846951199266, -0.7833189037391082, -0.04479766023508902, 0.9671134046085805, -0.20469470395736655, -0.5724658880611122, 0.11982629699117098, 0.22938346044068103, 0.4730661472707469, -0.7650786486055623, 0.6684865150649821, -0.8802888388988859, 0.8226227983882577, -0.19073198732031327, -0.8250811083914014, 0.3345976704260267, 0.6892134772240224, 0.7268721025348197, 0.3910200522279623,
                   0.7399207352106172, 0.2717342424659679, -0.6290429424350787, -0.07251264104068822, -0.848353794351494, -0.40618784546971076, 0.2161417094066418, -0.7588908973920898, 0.33922070907143187, -0.5801356304071028, -0.9984846408412742, 0.7546936820604886, 0.5117158056491005,};
    weights[2] = init_matrix_from_array(7, 89, a2);
    double b2[] = {0.4617563814065817, -0.17983837701559668, -0.5845703173805659, -0.33456588808097765, 0.9355118188482414, -0.9877656354684774, 0.9274095940464153,};
    biases[2] = init_matrix_from_array(7, 1, b2);
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