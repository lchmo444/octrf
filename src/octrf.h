#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <cassert>
#include <stdexcept>
#include <map>
#include <cmath>

namespace octrf {
    typedef double valtype; // type of feature values
    typedef std::vector< std::pair<int, valtype> > SV; // sparse vector
    typedef std::pair<valtype, SV> SExample;
    typedef std::vector<SExample> SExampleSet;
    //typedef std::vector<valtype> dv; // dense vector

    namespace io {
        void read_libsvmformat(const std::string& filename, SExampleSet& data);
        void save_libsvmformat(const std::string& filename, const SExampleSet& data);
    };

    double entropy(const SExampleSet& data);

    class Leaf {
        
    };

    /*
    class Node {
    };
    
    class Tree {
    };

    class Forest {
    };
    */
};
