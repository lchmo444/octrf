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
    a.add<bool>("online", 'o', "online or batch", false, false);
    a.add<bool>("chatty", 'c', "chatty or not", false, false);
    a.parse_check(argc, argv);

    typedef ExampleSet<int, dSV> SExampleSet;
    SExampleSet data;
    int dim = io::read_libsvmformat(a.get<string>("data"), data);

    TreeTrainingParameters trp(0, 0.1, 1, 500, 500, a.get<bool>("chatty"));
    ForestTrainingParameters ftrp(a.get<int>("ntrees"), trp);

    dBinaryDecisionForest model(dim, testfuncs::BinaryStamp<double>(dim));
    if(!a.get<bool>("online")){
        benchmark("train"){
            model.train(data, objfuncs::entropy, ftrp);
        }
        model.save(a.get<string>("model"));
    }
    else {
        benchmark("online train"){
            std::vector<int> idxs;
            for(int i = 0; i < data.size(); ++i) idxs.push_back(i);
            std::random_shuffle(idxs.begin(), idxs.end());
            for(int i=0; i < data.size(); ++i){
                model.train1(data[idxs[i]], objfuncs::entropy, ftrp);
            }
        }
        model.save(a.get<string>("model"));
    }
    
    
    return 0;
}


