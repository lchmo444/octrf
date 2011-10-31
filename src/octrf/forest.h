#pragma once

#include "common.h"
#include "tree.h"

namespace octrf {
    template <typename YType,
              typename XType,
              typename TestFunc,
              typename PredictResultType>
    class Forest {
        typedef Tree<YType, XType, TestFunc> mytree;
        typedef ExampleSet<YType, XType> ES;

        int ntrees_;
        int dim_;
        TestFunc tf_;
        std::vector<mytree> trees_;
    public:
        double objfunc_th_; // if the objfunc is lesser than this value, growing is stopped
        int nexamples_th_;  // if #data < this value, growing is stopped
        int nsamplings_;    // the number of random samplings

        Forest(const int ntrees, const int dim, TestFunc tf, const double objfunc_th = 0.1, int nexamples_th = 1, int nsamplings = 300)
            : ntrees_(ntrees), dim_(dim), tf_(tf),
              objfunc_th_(objfunc_th), nexamples_th_(nexamples_th), nsamplings_(nsamplings)
        {};

        template <typename Predictor> // vector<YType> -> PredictResultType
        PredictResultType predict(const XType& x, Predictor& pr) const {
            assert(trees_.size() == ntrees_);
            std::vector<YType> results;
            for(int i=0; i < trees_.size(); i++)
                results.push_back(trees_[i].predict(x));
            return pr(results);
        }

        template <typename ObjFunc>
        void train(const ES& data, ObjFunc& objfunc){
            std::vector<int> idxs;
            for(int i = 0; i < data.size(); ++i) idxs.push_back(i);
            std::random_shuffle(idxs.begin(), idxs.end());
            auto it = idxs.begin();
            trees_.clear();
            for(int i=0; i < ntrees_; i++){
                std::vector<int> subidxs;
                for(int j=0; j < idxs.size()/ntrees_ && it != idxs.end(); ++j, ++it){
                    subidxs.push_back(*it);
                }
                ES partofdata;
                data.subset(subidxs, partofdata);
                trees_.push_back(mytree(dim_, tf_, objfunc_th_, nexamples_th_, nsamplings_));
                trees_[i].train(partofdata, objfunc);
            }
        }
 
        void save(const std::string& filename) const {
            std::ofstream ofs(filename.c_str());
            if(ofs.fail()) throw std::runtime_error("Cannot open file : " + filename);
            std::deque<std::string> dq;
            for(int i = 0; i < trees_.size(); ++i){
                trees_[i].recursive_serialize(dq);
            }
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
            trees_.clear();
            for(int i = 0; i < ntrees_; ++i){
                trees_.push_back(mytree(dim_, tf_, objfunc_th_, nexamples_th_, nsamplings_));
                trees_[i].recursive_deserialize(dq);
            }
            ifs.close();
        }
    };

    namespace predictors {
        template <typename YType, typename PredictResultType>
        PredictResultType average(const std::vector<YType>& Y){
            PredictResultType avg = 0;
            for(int i = 0; i < Y.size(); ++i){
                avg += Y[i];
            }
            avg /= (PredictResultType)Y.size();
            return avg;
        }
    } // predictors
}
