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

    // tree
    {
        Tree<int, dSV, testfuncs::BinaryStamp<double> > rt(dim, testfuncs::BinaryStamp<double>(dim));
        benchmark("train"){
            rt.train(data, objfuncs::entropy);
        }
        benchmark("save"){
            rt.save("tmp/tree.txt");
        }
        Tree<int, dSV, testfuncs::BinaryStamp<double> > rt_l(dim, testfuncs::BinaryStamp<double>(dim));
        benchmark("load"){
            rt_l.load("tmp/tree.txt");
        }

        SExampleSet testdata;
        io::read_libsvmformat("data/a1a.t", testdata);
        int tp=0, tn=0, fp=0, fn=0;
        benchmark("predict"){
            for(int i=0; i < testdata.size(); i++){
                bool p = rt.predict(testdata.X_[i]) > 0;
                bool p2 = rt_l.predict(testdata.X_[i]) > 0;
                assert(p == p2);
                if(p)
                    if(testdata.Y_[i] > 0) tp++;
                    else fp++;
                else
                    if(testdata.Y_[i] <= 0) tn++;
                    else fn++;
            }
        }
        float pr = tp / (float)(tp + fp);
        float rc = tp / (float)(tp + fn);
        float f = (2*pr*rc) / (pr + rc);
        printf("Positive: %d/%d\nNegative: %d/%d\nPrecision = %f\nRecall = %f\nF = %f\n",
               tp, fp, tn, fn, pr, rc, f);
    }

    // forest
    {
        typedef Forest<int, dSV, testfuncs::BinaryStamp<double>, double > myforest;
        myforest rf(15, dim, testfuncs::BinaryStamp<double>(dim));
        benchmark("train"){
            rf.train(data, objfuncs::entropy);
        }
        benchmark("save"){
            rf.save("tmp/forest.txt");
        }
        myforest rf_l(15, dim, testfuncs::BinaryStamp<double>(dim));
        benchmark("load"){
            rf_l.load("tmp/forest.txt");
        }

        SExampleSet testdata;
        io::read_libsvmformat("data/a1a.t", testdata);
        int tp=0, tn=0, fp=0, fn=0;
        benchmark("predict"){
            for(int i=0; i < testdata.size(); i++){
                bool p = rf.predict(testdata.X_[i], predictors::average<int, double>) > 0;
                bool p2 = rf_l.predict(testdata.X_[i], predictors::average<int, double>) > 0;
                assert(p == p2);
                if(p)
                    if(testdata.Y_[i] > 0) tp++;
                    else fp++;
                else
                    if(testdata.Y_[i] <= 0) tn++;
                    else fn++;
            }
        }
        float pr = tp / (float)(tp + fp);
        float rc = tp / (float)(tp + fn);
        float f = (2*pr*rc) / (pr + rc);
        printf("Positive: %d/%d\nNegative: %d/%d\nPrecision = %f\nRecall = %f\nF = %f\n",
               tp, fp, tn, fn, pr, rc, f);
    }

    cout << "All Tests Passed" << endl;
}


