#include <iostream>
#include <cmath>

int main(){
    int size = 2048;
    double A[size+2][size+2];

    A[0][0] = 10;
    A[size+1][0] = 20;
    A[0][size+1] = 20;
    A[size+1][size+1] = 30;

    double step = 10.0/(size+1);
    #pragma acc kernels
    {
    for (int i = 1; i < size + 1; i++){
        A[i][0] = 10 + step*i;
        A[0][i] = 10 + step*i;
        A[size+1][i] = 20 + step*i;
        A[i][size+1] = 20 + step*i;
    }
    for (int i = 1; i < size + 1; i++){
        for (int j = 1; j < size + 1; j++){
            //A[i][j] = (double)(size+1-i)/(size+1)*A[0][j] + (double)i/(size+1)*A[size+1][j];
            A[i][j] = 0;
        }
    }
    }

//    std::cout << step << std::endl;
//    for (int i = 0; i < size + 2; i++){
//        for (int j = 0; j < size + 2; j++){
//            std::cout << " " << A[i][j];
//        }
//        std::cout << std::endl;
//    }

    double Anew[size+2][size+2];
    #pragma acc data copy(A) create(Anew)
    {
    int iter_max = 1000000;
    double tol = 0.000001;

    double err = 1;
    int iter = 0;
    while (err > tol && iter < iter_max){
        iter++;
        err = 0;

        #pragma acc kernels
        {
            #pragma acc loop independent collapse(2) reduction(max:err)
        for (int j = 1; j < size + 1; j++){
            for (int i = 1; i < size + 1; i++){
                Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                err = std::max(err, Anew[i][j] - A[i][j]);
            }
        }
        
        for (int j = 1; j < size + 1; j++) {
            for (int i = 1; i < size + 1; i++) {
                A[i][j] = Anew[i][j];
            }
        }
        }
        if (iter%100 == 0 || iter==1)
            std::cout << iter << " " << err << std::endl;
    }
    }
//    for (int i = 0; i < size + 2; i++){
//        for (int j = 0; j < size + 2; j++){
//            std::cout << " " << A[i][j];
//        }
//        std::cout << std::endl;
//    }
    //std::cout << iter << " " << err << std::endl;
    return 0;
}
