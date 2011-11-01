#pragma once

#include "common.h"

namespace octrf {
    struct TreeTrainingParameters {
        double objfunc_th; // if the entropy is lesser than this value, growing is stopped
        int nexamples_th;  // if #data < this value, growing is stopped
        int nsamplings;    // the number of random samplings
        TreeTrainingParameters(const double objfunc_th = 0, int nexamples_th = 1, int nsamplings = 300)
            : objfunc_th(objfunc_th),
              nexamples_th(nexamples_th),
              nsamplings(nsamplings)
        {};
    };


    template <typename YType, typename XType, typename LeafType,
              typename TestFunc>
    class Tree {
        typedef Tree<YType, XType, LeafType, TestFunc> mytree;
        typedef ExampleSet<YType, XType> ES;

        int dim_;           // the number of features' dimension
        TestFunc tf_;
        std::pair<bool, std::shared_ptr<LeafType> > leaf_;
        std::shared_ptr<mytree> tr_;
        std::shared_ptr<mytree> tl_;
    public:
        Tree(const int dim, TestFunc tf)
            : dim_(dim), tf_(tf),
              leaf_(false, std::shared_ptr<LeafType>())
        {};

        LeafType predict(const XType& x) const {
            if(leaf_.first) return *(leaf_.second);
            return tf_(x) ? tr_->predict(x) : tl_->predict(x);
        }

        template <typename ObjFunc>
        void train(const ES& data, ObjFunc objfunc, const TreeTrainingParameters& trp){
            if(objfunc(data.Y_) < trp.objfunc_th || data.size() <= trp.nexamples_th){
                leaf_ = std::make_pair(true, std::shared_ptr<LeafType>(new LeafType(data.Y_)));
                return;
            }

            double mine = DBL_MAX;
            TestFunc best_tf;
            for(int c=0; c < trp.nsamplings; c++){
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
                leaf_ = std::make_pair(true, std::shared_ptr<LeafType>(new LeafType(data.Y_)));
                return;
            }
            tf_ = best_tf;
            tr_ = std::shared_ptr<mytree>(new mytree(dim_, tf_));
            tl_ = std::shared_ptr<mytree>(new mytree(dim_, tf_));
            tr_->train(rdata, objfunc, trp);
            tl_->train(ldata, objfunc, trp);
        }

        std::string serialize() const {
            std::stringstream ss;
            if(leaf_.first){
                ss << "1\t" << leaf_.second->serialize() << std::endl;
            } else {
                ss << "0\t" << tf_.serialize() << std::endl;
            }
            return ss.str();
        }

        void deserialize(const std::string& s){
            std::stringstream ss(s);
            int is_leaf = 0;
            ss >> is_leaf;
            if(is_leaf == 1){
                std::string str;
                ss >> str;
                leaf_ = std::make_pair(true, std::shared_ptr<LeafType>(new LeafType()));
                leaf_.second->deserialize(str);
            } else {
                std::string str;
                ss >> str;
                tf_.deserialize(str);
            }
        }

        void recursive_serialize(std::deque<std::string>& dq) const {
            dq.push_back(serialize());
            if(!leaf_.first){
                tr_->recursive_serialize(dq);
                tl_->recursive_serialize(dq);
            }
        }

        void recursive_deserialize(std::deque<std::string>& dq){
            assert(dq.size() > 0);
            std::string s = dq[0];
            dq.pop_front();
            deserialize(s);
            if(!leaf_.first){
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
