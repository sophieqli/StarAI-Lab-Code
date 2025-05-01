#include <iostream>

using namespace std;

const int rows = 3, cols = 3;

void normalize(double p[][cols], int rows, int cols){
    double tot = 0;

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            tot += p[i][j];
        }
    }

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            p[i][j] /= tot;
        }
    }
}

void calc_margs(double p[][cols],double cur_rm[], double cur_cm[]){
    for (int i = 0; i < rows; i++){
        double tmp = 0;
        for (int j = 0; j<cols; j++) tmp += p[i][j];
        //get row sum
        cur_rm[i] = tmp;
    }

    for (int j = 0; j < cols; j++){
        double tmp = 0;
        for (int i = 0; i<rows; i++) tmp += p[i][j];
        //col sum
        cur_cm[j] = tmp;
    }

}
int main(){



    double p[rows][cols] = { {0.02, 0.1, 0.43}, {0.07, 0.06, 0.04}, {0.07, 0.07, 0.14}};

    normalize(p, rows, cols); //so they add to 1

    //specify marginals (uniform if extracting copula)
    double tar_rm[rows] = {0.3333, 0.3333, 0.3333};
    double tar_cm[rows] = {0.3333, 0.3333, 0.3333};




    //current row/col marginals that are updated as we go
    double cur_rm[rows];
    double cur_cm[cols];

    cout<<"Initial Distribution: "<<endl;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            cout<<p[i][j]<<" ";
        }
        cout<<endl;
    }

    //calculate and store marginals
    calc_margs(p, cur_rm, cur_cm);


    for (int it = 0; it < 300; it++){
        //update the target marginals *EXPERIMENT*
        /*
        tar_rm[0] = 0.3333 - 1.0/(it+3.5);
        tar_rm[1] = 0.3333 + 1.0/(2.0*(it+3.5));
        tar_rm[2] = 0.3333 + 1.0/(2.0*(it+3.5));
        tar_cm[0] = 0.3333 - 1.0/(it+3.5);
        tar_cm[1] = 0.3333 + 1.0/(2.0*(it+3.5));
        tar_cm[2] = 0.3333 + 1.0/(2.0*(it+3.5));
        */
        cout<<"tar margs update "<<tar_rm[0]<<" "<<tar_rm[1]<<" "<<tar_rm[2]<<endl;

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                p[i][j] *=  (tar_rm[i]/cur_rm[i]) * (tar_cm[j]/cur_cm[j]);

            }
        }
        normalize(p, rows, cols);
        calc_margs(p, cur_rm, cur_cm);

        cout<<"iteration : "<<it+1<<endl;
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                cout<<p[i][j]<<" ";
            }
            cout<<endl;
        }

    }
    cout<<"--FINISHED ITERATIONS--"<<endl;
    cout<<endl;

    cout<<"final row marginals "<<endl;
    for (int i = 0; i< rows; i++){
        cout<<cur_rm[i]<<" ";
    }
    cout<<endl;
    cout<<"final col marginals "<<endl;
    for (int i = 0; i< cols; i++){
        cout<<cur_cm[i]<<" ";
    }
    cout<<endl;


}

