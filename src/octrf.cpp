#include "octrf.h"

using namespace std;
using namespace pfi::lang;

namespace octrf {
    /* Tree */
    void Tree::train(const SExampleSet& data){
        cout << "Number of data: " << data.size() << ", Entropy: " << entropy(data) << endl;
        if(entropy(data) < entropy_th_ || data.size() <= nexamples_th_){
            valtype avg = 0;
            for(int i=0; i < data.size(); i++) avg += data[i].first;
            is_leaf_ = true;
            leaf_value_ = (valtype)(avg / (double)data.size());
            cout << "Leaf Value: " << leaf_value_ << endl;
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
            cout << "Leaf Value: " << leaf_value_ << endl;
            return;
        }
        bf_ = best_bf;
        tr_ = shared_ptr<Tree>(new Tree(dim_, bf_));
        tl_ = shared_ptr<Tree>(new Tree(dim_, bf_));
        tr_->train(rdata);
        tl_->train(ldata);
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
