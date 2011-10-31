#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <cassert>
#include <stdexcept>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <deque>
#include <memory>

namespace octrf {
    typedef std::vector< std::pair<int, double> > dSV;
    typedef std::vector<double> dDV;
    typedef std::vector< std::pair<int, float> > SV;
    typedef std::vector<float> DV;

    template <class YType, class XType>
    class ExampleSet {
    public:
        std::vector<YType> Y_;
        std::vector<XType> X_;
        ExampleSet(){}
        ExampleSet(const int n) : Y_(n), X_(n){}
        void push_back(const std::pair<YType, XType>& example){
            Y_.push_back(example.first);
            X_.push_back(example.second);
        }

        void push_back(const YType& y, const XType& x){
            Y_.push_back(y);
            X_.push_back(x);
        }

        void push_to(ExampleSet<YType, XType>& es, const int i) const {
            es.push_back(Y_[i], X_[i]);
        }

        size_t size() const { return Y_.size(); }
    };
}

