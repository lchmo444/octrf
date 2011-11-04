#pragma once

#include "common.h"

namespace octrf {
    namespace testfuncs {
        template <typename T>
        class BinaryStamp { // decision stamp for binary features
            int dim_;
            T th_;
            int d_;
        public:
            BinaryStamp() : dim_(1), th_(0), d_(0){}
            BinaryStamp(const int dim, const int th = 0)
                :  dim_(dim), th_(th), d_(0){};
            bool operator() (const std::vector< std::pair<int, T> >& x) const { // for SV
                for(int i=0; i < x.size(); i++){
                    if(x[i].first == d_) return x[i].second > th_;
                }
                return  0 > th_;
            }
            bool operator() (const std::vector<T>& x) const { // for DV
                return x[d_] > th_;
            }
            void random_sample() { d_ = rand() % dim_; }
            std::string serialize() const {
                std::stringstream ss;
                ss << d_;
                return ss.str();
            }
            void deserialize(const std::string& s){
                std::stringstream ss(s);
                ss >> d_;
            }
        };
        
        template <typename T>
        class Stamp { // decision stamp
            int dim_;
            double delta_;
            T lb_, ub_;
            T th_;
            int d_;
        public:
            Stamp() : dim_(1), th_(0), d_(0){}
            Stamp(const int dim, const T lb = 0, const T ub = 1, const double delta = 1e-4)
                :  dim_(dim), delta_(delta), lb_(lb), ub_(ub), th_(0), d_(0){};
            bool operator() (const std::vector< std::pair<int, T> >& x) const { // for SV
                for(int i=0; i < x.size(); i++){
                    if(x[i].first == d_) return x[i].second > th_;
                }
                return  0 > th_;
            }
            bool operator() (const std::vector<T>& x) const { // for DV
                return x[d_] > th_;
            }
            void random_sample() {
                d_ = rand() % dim_;
                th_ = lb_ + (ub_ - lb_) * ((rand() % (int)(1/delta_)) * delta_);
            }
            std::string serialize() const {
                std::stringstream ss;
                ss << d_ << " " << th_;
                return ss.str();
            }
            void deserialize(const std::string& s){
                std::stringstream ss(s);
                ss >> d_;
                ss >> th_;
            }
        };
        
    } // testfuncs
} // octrf
