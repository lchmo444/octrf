#include "octrf.h"

using namespace std;
using namespace pfi::lang;

namespace octrf {
    /* Tree */
    bool Tree::branch(const SV& x) const {
        for(int i=0; i < x.size(); i++){
            if(x[i].first == branchfunc_.first) return x[i].second < branchfunc_.second;
        }
        return 0 < branchfunc_.second;
    }

    bool Tree::branch(const SV& x, const std::pair<int, valtype>& bf) {
        for(int i=0; i < x.size(); i++){
            if(x[i].first == bf.first) return x[i].second < bf.second;
        }
        return 0 < bf.second;
    }

    void Tree::train(const SExampleSet& data, vector<pair<int, valtype> >& branchfuncs){
        if(branchfuncs.size() == 0){
            for(int i=0; i < dim_; i++){
                branchfuncs.push_back(make_pair(i, 0.3));
            }
        }

        cout << "Number of data: " << data.size() << endl;
        cout << "Entropy: " << entropy(data) << endl;
        if(entropy(data) < 0.1 || data.size() <= 1){
            is_leaf_ = true;
            double avg = 0;
            for(int i=0; i < data.size(); i++) avg += data[i].first;
            leaf_value_ = (valtype)(avg / (double)data.size());
            cout << "Leaf Value: " << leaf_value_ << endl;
            return;
        }

        double mine = 1e+5;
        int minidx = 0;
        for(int c=0; c < /*branchfuncs.size()*/min(1000, dim_); c++){
            int idx = rand() % (branchfuncs.size()-1);
            const std::pair<int, valtype>& bf = branchfuncs[idx];
            SExampleSet rdata, ldata;
            for(int i=0; i < data.size(); i++){
                if(branch(data[i].second, bf)) rdata.push_back(data[i]);
                else ldata.push_back(data[i]);
            }
            double e = (double)rdata.size() * entropy(rdata) + (double)ldata.size() * entropy(ldata);
            if(e < mine){
                mine = e;
                minidx = idx;
            }
        }

        branchfunc_ = branchfuncs[minidx];
        SExampleSet rdata, ldata;
        for(int i=0; i < data.size(); i++){
            if(branch(data[i].second, branchfunc_)) rdata.push_back(data[i]);
            else ldata.push_back(data[i]);
        }
        if(rdata.size() == 0 || ldata.size() == 0){
            is_leaf_ = true;
            double avg = 0;
            for(int i=0; i < data.size(); i++) avg += data[i].first;
            leaf_value_ = (valtype)(avg / (double)data.size());
            cout << "Leaf Value: " << leaf_value_ << endl;
            return;
        }
        tr_ = shared_ptr<Tree>(new Tree(dim_));
        tl_ = shared_ptr<Tree>(new Tree(dim_));
        tr_->train(rdata, branchfuncs);
        tl_->train(ldata, branchfuncs);
    }
    
    
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
