#include "octrf.h"

using namespace std;

namespace octrf {
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
        void read_libsvmformat(const std::string& filename, SExampleSet& data){
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
                    x.push_back(std::make_pair(i, v));
                }
                data.push_back(std::make_pair(y, x));
            }
            ifs.close();
        }

        void save_libsvmformat(const std::string& filename, const SExampleSet& data){
            std::ofstream ofs(filename.c_str());
            if(ofs.fail()) throw std::runtime_error("cannot open file: " + filename);
            for(int i=0; i < data.size(); i++){
                const valtype& y = data[i].first;
                const SV& x = data[i].second;
                ofs << y << " ";
                for(int j=0; j < x.size(); j++){
                    ofs << x[j].first << ":" << x[j].second << " ";
                }
                ofs << endl;
            }
            ofs.close();
        }
    };
    
};
