#pragma once

#include "octrf/common.h"
#include "octrf/io.h"
#include "octrf/objfuncs.h"
#include "octrf/testfuncs.h"
#include "octrf/tree.h"

#if 0
#include "octrf/common.h"

namespace octrf {
    class Tree {
        int dim_;           // the number of features' dimension
        std::shared_ptr<bfs::Base> bf_;
        bool is_leaf_;
        valtype leaf_value_;
        std::shared_ptr<Tree> tr_;
        std::shared_ptr<Tree> tl_;
    public:
        double entropy_th_; // if the entropy is lesser than this value, growing is stopped
        int nexamples_th_; // if #data < this value, growing is stopped
        int nsamplings_;    // the number of random samplings

        Tree(const int dim, bfs::Base* bf, const double entropy_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
            : dim_(dim), bf_(bf), entropy_th_(entropy_th), nexamples_th_(nexamples_th), nsamplings_(nsamplings),
              is_leaf_(false), leaf_value_(0)
        {};
        Tree(const int dim, std::shared_ptr<bfs::Base> bf, const double entropy_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
            : dim_(dim), bf_(bf), entropy_th_(entropy_th), nexamples_th_(nexamples_th), nsamplings_(nsamplings),
              is_leaf_(false), leaf_value_(0)
        {};

        valtype predict(const SV& x) const {
            if(is_leaf_) return leaf_value_;
            return bf_->branch(x) ? tr_->predict(x) : tl_->predict(x);
        }
        void train(const SExampleSet& data);
        std::string serialize() const;
        void deserialize(const std::string& s);
        void recursive_serialize(std::deque<std::string>& dq) const;
        void recursive_deserialize(std::deque<std::string>& dq);
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    };

    class Forest {
        int ntrees_;
        int dim_;
        std::shared_ptr<bfs::Base> bf_;
        std::vector<Tree> trees_;
    public:
        double entropy_th_; // if the entropy is lesser than this value, growing is stopped
        int nexamples_th_;  // if #data < this value, growing is stopped
        int nsamplings_;    // the number of random samplings

        Forest(const int ntrees, const int dim, bfs::Base* bf, const double entropy_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
            : ntrees_(ntrees), dim_(dim), bf_(bf),
              entropy_th_(entropy_th), nexamples_th_(nexamples_th), nsamplings_(nsamplings)
        {};
        valtype predict(const SV& x) const {
            assert(trees_.size() == ntrees_);
            valtype avg = 0;
            for(int i=0; i < trees_.size(); i++) avg += trees_[i].predict(x);
            avg /= (valtype)trees_.size();
            return avg;
        }
        void train(const SExampleSet& data){
            trees_.clear();
            for(int i=0; i < ntrees_; i++){
                trees_.push_back(Tree(dim_, bf_, entropy_th_, nexamples_th_, nsamplings_));
                trees_[i].train(data);
            }
        }
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    };
}
#endif
