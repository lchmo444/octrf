#pragma once

#include "common.h"

namespace octrf {
    template <typename YType, typename XType,
              typename TestFunc>
    class Tree {
        typedef Tree<YType, XType, TestFunc> mytree;
        typedef ExampleSet<YType, XType> ES;

        int dim_;           // the number of features' dimension
        TestFunc tf_;
        bool is_leaf_;
        YType leaf_value_;
        std::shared_ptr<mytree> tr_;
        std::shared_ptr<mytree> tl_;
    public:
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
            tr_ = std::shared_ptr<mytree>(new mytree(dim_, tf_));
            tl_ = std::shared_ptr<mytree>(new mytree(dim_, tf_));
            tr_->train(rdata, objfunc);
            tl_->train(ldata, objfunc);
        }

        std::string serialize() const {
            std::stringstream ss;
            if(is_leaf_){
                ss << "1\t" << leaf_value_ << std::endl;
            } else {
                ss << "0\t" << tf_.serialize() << std::endl;
            }
            return ss.str();
        }

        void deserialize(const std::string& s){
            std::stringstream ss(s);
            int is_leaf = 0;
            ss >> is_leaf;
            is_leaf_ = is_leaf == 1;
            if(is_leaf){
                ss >> leaf_value_;
            } else {
                std::string str;
                ss >> str;
                tf_.deserialize(str);
            }
        }

        void recursive_serialize(std::deque<std::string>& dq) const {
            dq.push_back(serialize());
            if(!is_leaf_){
                tr_->recursive_serialize(dq);
                tl_->recursive_serialize(dq);
            }
        }

        void recursive_deserialize(std::deque<std::string>& dq){
            std::string s = dq[0];
            dq.pop_front();
            deserialize(s);
            if(!is_leaf_){
                tr_ = std::shared_ptr<mytree>(new mytree(dim_, tf_));
                tr_->recursive_deserialize(dq);
                tl_ = std::shared_ptr<mytree>(new mytree(dim_, tf_));
                tl_->recursive_deserialize(dq);
            }
        }

        void save(const std::string& filename) const {
            std::ofstream ofs(filename.c_str());
            if(ofs.fail()) throw std::runtime_error("Cannot open file : " + filename);
            std::deque<std::string> dq;
            recursive_serialize(dq);
            for(std::deque<std::string>::iterator it = dq.begin();
                it != dq.end(); ++it)
            {
                ofs << *it;
            }
            ofs.close();
        }

        void load(const std::string& filename){
            std::ifstream ifs(filename.c_str());
            if(ifs.fail()) throw std::runtime_error("Cannot open file : " + filename);
            std::deque<std::string> dq;
            std::string buf;
            while(getline(ifs, buf)){
                dq.push_back(buf);
            }
            recursive_deserialize(dq);
            ifs.close();
        }
    };
} // octrf
