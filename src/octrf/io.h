#pragma once

#include "common.h"

namespace octrf {
    namespace io {
        template <typename T>
        int read_libsvmformat(const std::string& filename, ExampleSet<int, std::vector<std::pair<int, T> > >& data);
        template <typename T>
        void save_libsvmformat(const std::string& filename, const ExampleSet<int, std::vector<std::pair<int, T> > >& data);
    }
} // octrf

