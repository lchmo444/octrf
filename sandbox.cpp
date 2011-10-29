#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <octrf.h>
#include <cassert>
#include <cstdio>
#include <benchmark.hpp>

using namespace std;
using namespace octrf;

int main(){
    // io
    SExampleSet data;
    int dim = io::read_libsvmformat("data/a1a", data);
    assert(dim != 0);
    assert(data.size() == 1605);
    assert(data[1605-1].first == -1);
    assert(data[1598-1].first == +1);
    assert(data[0].second[1].first == 10);
    assert(data[0].second[1].second == +1);

    SExampleSet small_data(10);
    copy(data.begin(), data.begin() + 10, small_data.begin());
    io::save_libsvmformat("data/a1a.small", small_data);

    // entropy
    SExampleSet biases_data(data);
    for(int i=0; i < biases_data.size()-100; i++){
        biases_data[i].first = +1;
    }
    assert(entropy(data) > entropy(biases_data));

    // train & predict tree
    {
        Tree rt(dim, new bfs::BinaryStamp(dim));
        benchmark("train"){
            rt.train(data);
        }
        benchmark("save"){
            rt.save("tmp/tree.txt");
        }
        Tree rt_l(dim, new bfs::BinaryStamp(dim));
        benchmark("load"){
            rt_l.load("tmp/tree.txt");
        }
        
        data.clear();
        io::read_libsvmformat("data/a1a.t", data);
        int tp=0, tn=0, fp=0, fn=0;
        for(int i=0; i < data.size(); i++){
            assert(abs(rt.predict(data[i].second) - rt_l.predict(data[i].second)) < 0.01);
            bool p = rt.predict(data[i].second) > 0;
            if(p)
                if(data[i].first > 0) tp++;
                else fp++;
            else
                if(data[i].first <= 0) tn++;
                else fn++;
        }
        float pr = tp / (float)(tp + fp);
        float rc = tp / (float)(tp + fn);
        float f = (2*pr*rc) / (pr + rc);
        printf("Positive: %d/%d\nNegative: %d/%d\nPrecision = %f\nRecall = %f\nF = %f\n",
               tp, fp, tn, fn, pr, rc, f);
    }

    // train & predict forest
    if(1){
        const int ntrees = 3;
        Forest rf(ntrees, dim, new bfs::BinaryStamp(dim));
        rf.nsamplings_ = 100;
        data.clear();
        io::read_libsvmformat("data/a1a", data);
        benchmark("train forest"){
            rf.train(data);
        }
        rf.save("tmp/forest.txt");
        Forest rf_l(ntrees, dim, new bfs::BinaryStamp(dim));
        rf_l.load("tmp/forest.txt");

        data.clear();
        io::read_libsvmformat("data/a1a.t", data);
        int tp=0, tn=0, fp=0, fn=0;
        for(int i=0; i < data.size(); i++){
            bool p = rf.predict(data[i].second) > 0;
            bool p_l = rf_l.predict(data[i].second) > 0;
            assert(p == p_l);
            if(p) {
                if(data[i].first > 0) tp++;
                else fp++;
            }
            else {
                if(data[i].first <= 0) tn++;
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


