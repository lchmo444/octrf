#pragma once

#include "octrf/common.h"
#include "octrf/io.h"
#include "octrf/objfuncs.h"
#include "octrf/testfuncs.h"
#include "octrf/tree.h"
#include "octrf/forest.h"
#include "octrf/leafs.h"

namespace octrf {
    typedef Forest<int, dSV, leafs::Avg<int, double>, testfuncs::BinaryStamp<double>, double > dBinaryDecisionForest;
    typedef Forest<int, SV, leafs::Avg<int, float>, testfuncs::BinaryStamp<float>, float > BinaryDecisionForest;
    typedef Forest<int, dSV, leafs::Avg<int, double>, testfuncs::Stamp<double>, double > dDecisionForest;
    typedef Forest<int, SV, leafs::Avg<int, float>, testfuncs::Stamp<float>, float > DecisionForest;
} // octrf
