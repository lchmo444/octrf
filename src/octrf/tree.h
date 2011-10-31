#pragma once

#include "common.h"

namespace octrf {
    template <typename YType, typename XType,
              typename TestFunc>
    class Tree {
        int dim_;           // the number of features' dimension
        TestFunc tf_;
        bool is_leaf_;
        YType leaf_value_;
        std::shared_ptr<Tree<YType, XType, TestFunc> > tr_;
        std::shared_ptr<Tree<YType, XType, TestFunc> > tl_;
    public:
        typedef ExampleSet<YType, XType> ES;
        double objfunc_th_; // if the entropy is lesser than this value, growing is stopped
        int nexamples_th_; // if #data < this value, growing is stopped
        int nsamplings_;    // the number of random samplings
        Tree(const int dim, TestFunc tf, const double objfunc_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
            : dim_(dim), tf_(tf), objfunc_th_(objfunc_th), nexamples_th_(nexamples_th), nsamplings_(nsamplings),
              is_leaf_(false), leaf_value_(0)
        {};
        YType predict(const XType& x) const {
            if(is_leaf_) return leaf_value_;
            return tf_(x) ? tr_->predict(x) : tl_->predict(x);
        }

        template <typename ObjFunc>
        void train(const ES& data, ObjFunc& objfunc){
            // cout << "Number of data: " << data.size() << ", Entropy: " << entropy(data) << endl;
            if(objfunc(data.Y_) < objfunc_th_ || data.size() <= nexamples_th_){
                YType avg = 0;
                for(int i=0; i < data.size(); i++) avg += data.Y_[i];
                is_leaf_ = true;
                leaf_value_ = (YType)(avg / (double)data.size());
                // cout << "Leaf Value: " << leaf_value_ << endl;
                return;
            }

            double mine = DBL_MAX;
            TestFunc best_tf;
            for(int c=0; c < nsamplings_; c++){
                TestFunc tf(tf_);
                tf.random_sample();
                ES rdata, ldata;
                for(int i=0; i < data.size(); i++){
                    if(tf(data.X_[i])) data.push_to(rdata, i);
                    else data.push_to(ldata, i);
                }
                double e = (double)rdata.size() * objfunc(rdata.Y_) + (double)ldata.size() * objfunc(ldata.Y_);
                if(e < mine){
                    mine = e;
                    best_tf = tf;
                }
            }

            ES rdata, ldata;
            for(int i=0; i < data.size(); i++){
                if(best_tf(data.X_[i])) data.push_to(rdata, i);
                else data.push_to(ldata, i);
            }
            if(rdata.size() == 0 || ldata.size() == 0){
                double avg = 0;
                for(int i=0; i < data.size(); i++) avg += data.Y_[i];
                is_leaf_ = true;
                leaf_value_ = (YType)(avg / (double)data.size());
                // cout << "Leaf Value: " << leaf_value_ << endl;
                return;
            }
            tf_ = best_tf;
            tr_ = std::shared_ptr<Tree>(new Tree<YType, XType, TestFunc>(dim_, tf_));
            tl_ = std::shared_ptr<Tree>(new Tree<YType, XType, TestFunc>(dim_, tf_));
            tr_->train(rdata, objfunc);
            tl_->train(ldata, objfunc);
        }

    };
} // octrf
