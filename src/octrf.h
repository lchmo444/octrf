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
#include <cfloat>
#include <deque>

namespace octrf {
    typedef double valtype; // type of feature values
    typedef std::vector< std::pair<int, valtype> > SV; // sparse vector
    typedef std::pair<valtype, SV> SExample;
    typedef std::vector<SExample> SExampleSet;
    //typedef std::vector<valtype> dv; // dense vector

    namespace io {
        int read_libsvmformat(const std::string& filename, SExampleSet& data);
        void save_libsvmformat(const std::string& filename, const SExampleSet& data);
    }

    double entropy(const SExampleSet& data);

    namespace bfs { // branching functions
        class Base {
        protected:
            int dim_;
        public:
            Base(const int dim) : dim_(dim){};
            virtual bool branch(const SV& x) const {return false;} // false->left, true->right
            virtual void random_sample(){} // change parameters randomly
            virtual std::string serialize() const {return std::string("");} // parameters->string
            virtual void deserialize(const std::string& s){} // string->parameters
            virtual Base* clone() const {return new Base(dim_);}
        };

        class BinaryStamp : public Base { // decision stamp for binary features
            int d_;
            valtype th_;
        public:
            BinaryStamp(const int dim, const valtype th = 0)
                : Base(dim), d_(0), th_(th){};
            bool branch(const SV& x) const {
                for(int i=0; i < x.size(); i++){
                    if(x[i].first == d_) return x[i].second > th_;
                }
                return  0 > th_;
            }
            void random_sample() { d_ = rand() % dim_; }
            std::string serialize() const {
                std::stringstream ss;
                ss << d_;
                return ss.str();
            }
            void deserialize(const std::string& s){
                std::stringstream ss(s);
                ss >> d_;
            }
            Base* clone() const { return new BinaryStamp(dim_, th_); }
        };
    }


    class Tree {
        int dim_;           // the number of features' dimension
        pfi::lang::shared_ptr<bfs::Base> bf_;
        bool is_leaf_;
        valtype leaf_value_;
        pfi::lang::shared_ptr<Tree> tr_;
        pfi::lang::shared_ptr<Tree> tl_;
    public:
        double entropy_th_; // if the entropy is lesser than this value, growing is stopped
        int nexamples_th_; // if #data < this value, growing is stopped
        int nsamplings_;    // the number of random samplings

        Tree(const int dim, bfs::Base* bf, const double entropy_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
            : dim_(dim), bf_(bf), entropy_th_(entropy_th), nexamples_th_(nexamples_th), nsamplings_(nsamplings),
              is_leaf_(false), leaf_value_(0)
        {};
        Tree(const int dim, pfi::lang::shared_ptr<bfs::Base> bf, const double entropy_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
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
        pfi::lang::shared_ptr<bfs::Base> bf_;
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
