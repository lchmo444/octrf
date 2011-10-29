#include "octrf.h"

using namespace std;
using namespace pfi::lang;

namespace octrf {
    /* Tree */
    void Tree::train(const SExampleSet& data){
        // cout << "Number of data: " << data.size() << ", Entropy: " << entropy(data) << endl;
        if(entropy(data) < entropy_th_ || data.size() <= nexamples_th_){
            valtype avg = 0;
            for(int i=0; i < data.size(); i++) avg += data[i].first;
            is_leaf_ = true;
            leaf_value_ = (valtype)(avg / (double)data.size());
            // cout << "Leaf Value: " << leaf_value_ << endl;
            return;
        }

        double mine = DBL_MAX;
        shared_ptr<bfs::Base> best_bf;
        for(int c=0; c < nsamplings_; c++){
            shared_ptr<bfs::Base> bf(bf_->clone());
            bf->random_sample();
            SExampleSet rdata, ldata;
            for(int i=0; i < data.size(); i++){
                if(bf->branch(data[i].second)) rdata.push_back(data[i]);
                else ldata.push_back(data[i]);
            }
            double e = (double)rdata.size() * entropy(rdata) + (double)ldata.size() * entropy(ldata);
            if(e < mine){
                mine = e;
                best_bf = bf;
            }
        }

        SExampleSet rdata, ldata;
        for(int i=0; i < data.size(); i++){
            if(best_bf->branch(data[i].second)) rdata.push_back(data[i]);
            else ldata.push_back(data[i]);
        }
        if(rdata.size() == 0 || ldata.size() == 0){
            double avg = 0;
            for(int i=0; i < data.size(); i++) avg += data[i].first;
            is_leaf_ = true;
            leaf_value_ = (valtype)(avg / (double)data.size());
            // cout << "Leaf Value: " << leaf_value_ << endl;
            return;
        }
        bf_ = best_bf;
        tr_ = shared_ptr<Tree>(new Tree(dim_, bf_));
        tl_ = shared_ptr<Tree>(new Tree(dim_, bf_));
        tr_->train(rdata);
        tl_->train(ldata);
    }

    string Tree::serialize() const {
        stringstream ss;
        if(is_leaf_){
            ss << "1\t" << leaf_value_ << endl;
        } else {
            ss << "0\t" << bf_->serialize() << endl;
        }
        return ss.str();
    }

    void Tree::deserialize(const string& s){
        stringstream ss(s);
        int is_leaf = 0;
        ss >> is_leaf;
        is_leaf_ = is_leaf == 1;
        if(is_leaf){
            ss >> leaf_value_;
        } else {
            string str;
            ss >> str;
            bf_->deserialize(str);
        }
    }

    void Tree::recursive_serialize(std::deque<string>& dq) const {
        dq.push_back(serialize());
        if(!is_leaf_){
            tr_->recursive_serialize(dq);
            tl_->recursive_serialize(dq);
        }
    }

    void Tree::recursive_deserialize(std::deque<string>& dq){
        string s = dq[0];
        dq.pop_front();
        deserialize(s);
        if(!is_leaf_){
            tr_ = shared_ptr<Tree>(new Tree(dim_, bf_->clone()));
            tr_->recursive_deserialize(dq);
            tl_ = shared_ptr<Tree>(new Tree(dim_, bf_->clone()));
            tl_->recursive_deserialize(dq);
        }
    }

    void Tree::save(const std::string& filename) const {
        std::ofstream ofs(filename.c_str());
        if(ofs.fail()) throw std::runtime_error("Cannot open file : " + filename);
        deque<string> dq;
        recursive_serialize(dq);
        for(deque<string>::iterator it = dq.begin();
            it != dq.end(); ++it)
        {
            ofs << *it;
        }
        ofs.close();
    }

    void Tree::load(const std::string& filename){
        std::ifstream ifs(filename.c_str());
        if(ifs.fail()) throw std::runtime_error("Cannot open file : " + filename);
        deque<string> dq;
        string buf;
        while(getline(ifs, buf)){
            dq.push_back(buf);
        }
        recursive_deserialize(dq);
        ifs.close();
    }

    /* Forest */
    void Forest::save(const std::string& filename) const {
        std::ofstream ofs(filename.c_str());
        if(ofs.fail()) throw std::runtime_error("Cannot open file : " + filename);
        deque<string> dq;
        for(int i = 0; i < trees_.size(); ++i){
            trees_[i].recursive_serialize(dq);
        }
        for(deque<string>::iterator it = dq.begin();
            it != dq.end(); ++it)
        {
            ofs << *it;
        }
        ofs.close();
    }

    void Forest::load(const std::string& filename){
        std::ifstream ifs(filename.c_str());
        if(ifs.fail()) throw std::runtime_error("Cannot open file : " + filename);
        deque<string> dq;
        string buf;
        while(getline(ifs, buf)){
            dq.push_back(buf);
        }
        trees_.clear();
        for(int i = 0; i < ntrees_; ++i){
            trees_.push_back(Tree(dim_, bf_, entropy_th_, nexamples_th_, nsamplings_));
            trees_[i].recursive_deserialize(dq);
        }
        ifs.close();
    }
    

    /* etc */
    double entropy(const SExampleSet& data){
        double e = 0;
        map<double, int> nums;
        for(int i=0; i < data.size(); i++){
            const valtype& y = data[i].first;
            if(!nums.count(y)) nums.insert(make_pair(y, 0));
            nums[y]++;
        }
        for(map<double, int>::iterator it = nums.begin();
            it != nums.end(); ++it)
        {
            double p = it->second / (double)data.size();
            e -= p * log(p);
        }
        return e;
    }
    
    namespace io {
        // retval means the number of dimensions of data
        int read_libsvmformat(const std::string& filename, SExampleSet& data){
            int dim = 0;
            std::ifstream ifs(filename.c_str());
            if(ifs.fail()) throw std::runtime_error("cannot open file: " + filename);
            string buf;
            while(getline(ifs, buf)){
                std::stringstream ss(buf);
                valtype y = 0;
                SV x;
                ss >> y;
                int i = 0;
                valtype v = 0;
                while(ss >> i){
                    char h;
                    ss >> h;
                    ss >> v;
                    x.push_back(std::make_pair(i-1, v));
                    dim = max(dim, i);
                }
                data.push_back(std::make_pair(y, x));
            }
            ifs.close();

            return dim;
        }

        void save_libsvmformat(const std::string& filename, const SExampleSet& data){
            std::ofstream ofs(filename.c_str());
            if(ofs.fail()) throw std::runtime_error("cannot open file: " + filename);
            for(int i=0; i < data.size(); i++){
                const valtype& y = data[i].first;
                const SV& x = data[i].second;
                ofs << y << " ";
                for(int j=0; j < x.size(); j++){
                    ofs << x[j].first+1 << ":" << x[j].second << " ";
                }
                ofs << endl;
            }
            ofs.close();
        }
    };
    
};
