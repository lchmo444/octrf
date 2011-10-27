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
        int read_libsvmformat(const std::string& filename, SExampleSet& data);
        void save_libsvmformat(const std::string& filename, const SExampleSet& data);
    };

    double entropy(const SExampleSet& data);

    class Tree {
        int dim_;
        bool is_leaf_;
        valtype leaf_value_;
        std::pair<int, valtype> branchfunc_;
        pfi::lang::shared_ptr<Tree> tr_;
        pfi::lang::shared_ptr<Tree> tl_;
    public:
        Tree(const int dim, const bool is_leaf = false, const valtype leaf_value = 0)
            : dim_(dim), is_leaf_(is_leaf), leaf_value_(leaf_value){};

        valtype predict(const SV& x) const {
            if(is_leaf_) return leaf_value_;
            return branch(x) ? tr_->predict(x) : tl_->predict(x);
        }
        bool branch(const SV& x) const;
        static bool branch(const SV& x, const std::pair<int, valtype>& bf);
        void train(const SExampleSet& data, std::vector<std::pair<int, valtype> >& branchfuncs);
    };

    class Forest {
        int dim_;
        std::vector<Tree> trees;
    public:
        Forest(const int dim) : dim_(dim){};
        valtype predict(const SV& x) const {
            valtype avg = 0;
            for(int i=0; i < trees.size(); i++) avg += trees[i].predict(x);
            avg /= (valtype)trees.size();
            return avg;
        }
        void train(const SExampleSet& data, int f = 3){
            trees.clear();
            for(int i=0; i < f; i++){
                std::vector<std::pair<int, valtype> > branchfuncs;
                trees.push_back(Tree(dim_));
                trees[i].train(data, branchfuncs);
            }
        }
    };
};
