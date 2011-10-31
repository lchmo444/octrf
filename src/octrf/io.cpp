#include "io.h"

using namespace std;

namespace octrf {
    namespace io {
        // retval means the number of dimensions of data
        template int read_libsvmformat(const std::string& filename, ExampleSet<int, SV>& data);
        template int read_libsvmformat(const std::string& filename, ExampleSet<int, dSV>& data);
        template <typename T>
        int read_libsvmformat(const std::string& filename, ExampleSet<int, std::vector<std::pair<int, T> > >& data){
            typedef std::vector<std::pair<int, T> > XType;
            int dim = 0;
            std::ifstream ifs(filename.c_str());
            if(ifs.fail()) throw std::runtime_error("cannot open file: " + filename);
            string buf;
            while(getline(ifs, buf)){
                std::stringstream ss(buf);
                int y = 0;
                XType x;
                ss >> y;
                int i = 0;
                T v = 0;
                while(ss >> i){
                    char h;
                    ss >> h;
                    ss >> v;
                    x.push_back(std::make_pair(i-1, v));
                    dim = max(dim, i);
                }
                data.push_back(y, x);
            }
            ifs.close();

            return dim;
        }

        template void save_libsvmformat(const std::string& filename, const ExampleSet<int, SV >& data);
        template void save_libsvmformat(const std::string& filename, const ExampleSet<int, dSV >& data);
        template <typename T>
        void save_libsvmformat(const std::string& filename, const ExampleSet<int, std::vector<std::pair<int, T> > >& data){
            typedef std::vector<std::pair<int, T> > XType;
            std::ofstream ofs(filename.c_str());
            if(ofs.fail()) throw std::runtime_error("cannot open file: " + filename);
            for(int i=0; i < data.size(); i++){
                const int& y = data.Y_[i];
                const XType& x = data.X_[i];
                ofs << y << " ";
                for(int j=0; j < x.size(); j++){
                    ofs << x[j].first+1 << ":" << x[j].second << " ";
                }
                ofs << endl;
            }
            ofs.close();
        }
        
    } // io
} // octrf
