#include "objfuncs.h"

using namespace std;

namespace octrf {
    namespace objfuncs {
        double entropy(const std::vector<int>& Y, const std::vector<int>& idxs){
            double e = 0;
            unordered_map<int, int> nums;
            for(int i=0; i < idxs.size(); i++){
                const int& y = Y[idxs[i]];
                if(!nums.count(y)) nums.insert(make_pair(y, 0));
                nums[y]++;
            }
            for(unordered_map<int, int>::iterator it = nums.begin();
                it != nums.end(); ++it)
            {
                double p = it->second / (double)Y.size();
                e -= p * log(p);
            }
            return e;
        }
    } // objfuncs
} // octrf
