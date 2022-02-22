#include <iostream>
#include <cmath>

int main(int argc, char *argv[]){

    double tol = atof(argv[1]); 
    int size = strtol(argv[2], NULL, 10); 
    int iter_max = strtol(argv[3], NULL, 10); 
    
    double** A = (double**)malloc((size + 2) * sizeof(double*));
    for (int i = 0; i < size + 2; ++i) {
        A[i] = (double*)calloc(size + 2, sizeof(double));
    }

    double** Anew = (double**)malloc((size + 2) * sizeof(double*));
    for (int i = 0; i < size + 2; ++i) {
        Anew[i] = (double*)calloc(size + 2, sizeof(double));
    }

    A[0][0] = 10;
    A[size+1][0] = 20;
    A[0][size+1] = 20;
    A[size+1][size+1] = 30;
    Anew[0][0] = 10;
    Anew[size+1][0] = 20;
    Anew[0][size+1] = 20;
    Anew[size+1][size+1] = 30;

    double step = 10.0/(size+1);
    #pragma acc kernels
    {
    #pragma acc loop independent
    for (int i = 1; i < size + 1; i++){
        A[i][0] = 10 + step*i;
        A[0][i] = 10 + step*i;
        A[size+1][i] = 20 + step*i;
        A[i][size+1] = 20 + step*i;
        Anew[i][0] = 10 + step*i;
        Anew[0][i] = 10 + step*i;
        Anew[size+1][i] = 20 + step*i;
        Anew[i][size+1] = 20 + step*i;
    }
    for (int i = 1; i < size + 1; i++){
        for (int j = 1; j < size + 1; j++){
            A[i][j] = 0;
        }
    }
    }

    double err = 1;
    int iter = 0;
    double** p;
    
    #pragma acc data copyin(A[0:size+2][0:size+2],Anew[0:size+2][0:size+2]) create(err)
    {
    while (err > tol && iter < iter_max){
        iter++;

        #pragma acc data present(A, Anew)
        if (iter % 100 == 0 || iter == 1){
            #pragma acc kernels async(1)
            {
                err = 0;
                #pragma acc loop independent collapse(2) reduction(max:err)
                for (int j = 1; j < size + 1; j++){
                    for (int i = 1; i < size + 1; i++){
                        Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                        err = std::max(err, Anew[i][j] - A[i][j]);
                }
                }
            }
        } else {
            #pragma acc kernels async(1)
            {
            #pragma acc loop independent collapse(2)
            for (int j = 1; j < size + 1; j++){
                for (int i = 1; i < size + 1; i++){
                    Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
            }
            }
        }
        }

        p = A;
        A = Anew;
        Anew = p;

        if (iter%100 == 0 || iter==1)
            #pragma acc wait(1)
            #pragma acc update self(err)
            std::cout << iter << " " << err << std::endl;
    }
    }
    return 0;
}