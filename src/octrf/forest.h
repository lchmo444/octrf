#pragma once

#include "common.h"
#include "tree.h"

namespace octrf {
    struct ForestTrainingParameters
    {
        int ntrees;
        TreeTrainingParameters tree_trp;
        ForestTrainingParameters(const int ntrees = 1,
                                 const TreeTrainingParameters tree_trp = TreeTrainingParameters())
            : ntrees(ntrees), tree_trp(tree_trp){};
    };


    template <typename YType,
              typename XType,
              typename LeafType,
              typename TestFunc,
              typename ResultType>
    class Forest {
        typedef Tree<YType, XType, LeafType, TestFunc> mytree;
        typedef ExampleSet<YType, XType> ES;

        int dim_;
        TestFunc tf_;
        std::vector<mytree> trees_;
    public:
        Forest(const int dim, TestFunc tf)
            : dim_(dim), tf_(tf)
        {};

        ResultType predict(const XType& x) const {
            std::vector<LeafType> results(trees_.size());
            for(int i=0; i < trees_.size(); i++)
                results[i] = trees_[i].predict(x);
            return LeafType::set2result(results);
        }

        void prepare(const ForestTrainingParameters& trp){
            while(trees_.size() < trp.ntrees) trees_.push_back(mytree(dim_, tf_));
        }

        template <typename ObjFunc>
        void train1(const std::pair<YType, XType>& example, ObjFunc& objfunc,
                    const ForestTrainingParameters& trp)
        {
            prepare(trp);
            trees_[rand() % trees_.size()].train1(example, objfunc, trp.tree_trp);
        }

        template <typename ObjFunc>
        void train(const ES& data, ObjFunc& objfunc, const ForestTrainingParameters& trp){
            std::vector<int> idxs;
            for(int i = 0; i < data.size(); ++i) idxs.push_back(i);
            std::random_shuffle(idxs.begin(), idxs.end());
            auto it = idxs.begin();
            std::vector< std::vector<int> > subidxs_set;
            for(int i=0; i < trp.ntrees; i++){
                std::vector<int> subidxs;
                for(int j=0; j < idxs.size()/trp.ntrees && it != idxs.end(); ++j, ++it){
                    subidxs.push_back(*it);
                }
                subidxs_set.push_back(subidxs);
            }

            trees_.clear();
            prepare(trp);
#pragma omp parallel for
            for(int i=0; i < trp.ntrees; i++){
                const std::vector<int>& subidxs = subidxs_set[i];
                ES partofdata;
                data.subset(subidxs, partofdata);
                trees_[i].train(partofdata, objfunc, trp.tree_trp);
            }
        }
 
        void save(const std::string& filename) const {
            std::ofstream ofs(filename.c_str());
            if(ofs.fail()) throw std::runtime_error("Cannot open file : " + filename);
            ofs << trees_.size() << std::endl;
            std::deque<std::string> dq;
            for(int i = 0; i < trees_.size(); ++i){
                trees_[i].recursive_serialize(dq);
                dq.push_back(std::string("\n"));
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
                if(buf != "")
                    dq.push_back(buf);
            }
            int ntrees;
            {
                std::stringstream ss(dq[0]);
                ss >> ntrees;
                dq.pop_front();
            }
            trees_.clear();
            for(int i = 0; i < ntrees; ++i){
                trees_.push_back(mytree(dim_, tf_));
                trees_[i].recursive_deserialize(dq);
            }
            ifs.close();
        }
    };
}

