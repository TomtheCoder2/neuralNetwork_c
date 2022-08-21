#ifndef NN_NN_H
#define NN_NN_H

class Matrix {
public:
    int rows;
    int cols;
    double *data;
};

double l_rate = 0.000056;

// just call the train function for each learn_set each epoch
void fit(size_t train_count, Matrix *[], Matrix *[],
         int epochs, int layerCount, Matrix *[], Matrix *[]);



#endif //NN_NN_H