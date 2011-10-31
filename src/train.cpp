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

    typedef Forest<int, dSV, leafs::Avg<int, double>, testfuncs::BinaryStamp<double>, double > myforest;
    myforest model(a.get<int>("ntrees"), dim, testfuncs::BinaryStamp<double>(dim), 0, 1, 500);
    benchmark("train"){
        model.train(data, objfuncs::entropy);
    }
    model.save(a.get<string>("model"));
    
    return 0;
}


