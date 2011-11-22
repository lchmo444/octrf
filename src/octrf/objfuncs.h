#pragma once

#include "common.h"
#include <unordered_map>

namespace octrf {
    namespace objfuncs {
        double entropy(const std::vector<int>& y, const std::vector<int>& idxs);
    } // objfuncs
} // octrf
