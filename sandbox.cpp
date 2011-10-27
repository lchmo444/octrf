#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <octrf.h>
#include <cassert>

using namespace std;
using namespace octrf;

int main(int argc, char** argv){
    // io
    SExampleSet data;
    io::read_libsvmformat("a1a", data);
    assert(data.size() == 1605);
    assert(data[1605-1].first == -1);
    assert(data[1598-1].first == +1);
    assert(data[0].second[1].first == 11);
    assert(data[0].second[1].second == +1);

    SExampleSet small_data(10);
    copy(data.begin(), data.begin() + 10, small_data.begin());
    io::save_libsvmformat("a1a.small", small_data);

    // entropy
    SExampleSet biases_data(data);
    for(int i=0; i < biases_data.size()-100; i++){
        biases_data[i].first = +1;
    }
    assert(entropy(data) > entropy(biases_data));

    cout << "All Tests Passed" << endl;
}


