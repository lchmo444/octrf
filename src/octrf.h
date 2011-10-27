#pragma once

#include <pficommon/lang/shared_ptr.h>
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

namespace octrf {
    typedef double valtype; // type of feature values
    typedef std::vector< std::pair<int, valtype> > SV; // sparse vector
    typedef std::pair<valtype, SV> SExample;
    typedef std::vector<SExample> SExampleSet;
    //typedef std::vector<valtype> dv; // dense vector

    namespace io {
        void read_libsvmformat(const std::string& filename, SExampleSet& data);
        void save_libsvmformat(const std::string& filename, const SExampleSet& data);
    };

    double entropy(const SExampleSet& data);

    class Tree {
        bool is_leaf_;
        valtype leaf_value_;
        std::pair<int, valtype> branchfunc_;
        pficommon::lang::shared_ptr<Tree> tr;
        pficommon::lang::shared_ptr<Tree> tl;
    public:
        Tree() : is_leaf_(false), leaf_value_(0){};
        valtype predict(const SV& x) const {
            if(is_leaf_) return leaf_value_;
            return branch(x) ? tr->predict(x) : tr->predict(x);
        }
        bool branch(const SV& x) const;
        void train(const SExampleSet& data);
    };

    /*
    class Forest {
    };
    */
};
