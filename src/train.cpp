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
    a.add<int>("ntrees", 'n', "#trees", false, 5);
    a.parse_check(argc, argv);

    typedef ExampleSet<int, dSV> SExampleSet;
    SExampleSet data;
    int dim = io::read_libsvmformat(a.get<string>("data"), data);

    dBinaryDecisionForest model(dim, testfuncs::BinaryStamp<double>(dim));
    benchmark("train"){
        model.train(data, objfuncs::entropy,
                    ForestTrainingParameters(a.get<int>("ntrees"), TreeTrainingParameters(0, 1, 500)));
    }
    model.save(a.get<string>("model"));
    
    return 0;
}


