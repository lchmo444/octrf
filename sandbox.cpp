#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include "src/octrf.h"
#include <cassert>
#include <cstdio>
#include "src/benchmark.hpp"

using namespace std;
using namespace octrf;

int main(){
    typedef ExampleSet<int, dSV> SExampleSet;
    // io
    SExampleSet data;
    int dim = io::read_libsvmformat("data/a1a", data);
    assert(dim != 0);
    assert(data.size() == 1605);
    assert(data.Y_[1605-1] == -1);
    assert(data.Y_[1598-1] == +1);
    assert(data.X_[0][1].first == 10);
    assert(data.X_[0][1].second == +1);

    SExampleSet small_data(10);
    copy(data.Y_.begin(), data.Y_.begin() + 10, small_data.Y_.begin());
    copy(data.X_.begin(), data.X_.begin() + 10, small_data.X_.begin());
    io::save_libsvmformat("data/a1a.small", small_data);

    // train & predict tree
    {
        Tree<int, dSV, testfuncs::BinaryStamp> rt(dim, testfuncs::BinaryStamp(dim));
        benchmark("train"){
            rt.train(data, objfuncs::entropy);
        }
//         benchmark("save"){
//             rt.save("tmp/tree.txt");
//         }
//         Tree rt_l(dim, new bfs::BinaryStamp(dim));
//         benchmark("load"){
//             rt_l.load("tmp/tree.txt");
//         }

        SExampleSet testdata;
        io::read_libsvmformat("data/a1a.t", testdata);
        int tp=0, tn=0, fp=0, fn=0;
        for(int i=0; i < testdata.size(); i++){
            bool p = rt.predict(testdata.X_[i]) > 0;
            if(p)
                if(testdata.Y_[i] > 0) tp++;
                else fp++;
            else
                if(testdata.Y_[i] <= 0) tn++;
                else fn++;
        }
        float pr = tp / (float)(tp + fp);
        float rc = tp / (float)(tp + fn);
        float f = (2*pr*rc) / (pr + rc);
        printf("Positive: %d/%d\nNegative: %d/%d\nPrecision = %f\nRecall = %f\nF = %f\n",
               tp, fp, tn, fn, pr, rc, f);
    }

    cout << "All Tests Passed" << endl;
}


