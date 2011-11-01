#include "octrf.h"
#include "benchmark.h"
#include "cmdline.h"

using namespace std;
using namespace octrf;

int main(int argc, char *argv[])
{
    cmdline::parser a;
    a.add<string>("data", 'd', "data file's name", true);
    a.add<string>("model", 'm', "trained model file's name", true);
    a.parse_check(argc, argv);

    typedef ExampleSet<int, dSV> SExampleSet;
    SExampleSet data;
    int dim = io::read_libsvmformat(a.get<string>("data"), data);

    cout << "initialize model" << endl;
    typedef Forest<int, dSV, leafs::Avg<int, double>, testfuncs::BinaryStamp<double>, double > myforest;
    myforest model(dim, testfuncs::BinaryStamp<double>(dim));
    cout << "load model from the file" << endl;
    model.load(a.get<string>("model"));

    cout << "predict" << endl;
    int tp=0, tn=0, fp=0, fn=0;
    benchmark("predict"){
        for(int i=0; i < data.size(); i++){
            bool p = model.predict(data.X_[i]) > 0;
            if(p)
                if(data.Y_[i] > 0) tp++;
                else fp++;
            else
                if(data.Y_[i] <= 0) tn++;
                else fn++;
        }
    }
    float pr = tp / (float)(tp + fp);
    float rc = tp / (float)(tp + fn);
    float f = (2*pr*rc) / (pr + rc);
    printf("Positive: %d/%d\nNegative: %d/%d\nPrecision = %f\nRecall = %f\nF = %f\n",
           tp, fp, tn, fn, pr, rc, f);
    
    return 0;
}


