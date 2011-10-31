#pragma once

#include "common.h"

namespace octrf {
    namespace testfuncs {
        class Base {
        public:
            Base(){}
            virtual bool branch(const SV& x) const {return false;} // false->left, true->right
            virtual void random_sample(){} // change parameters randomly
            virtual std::string serialize() const {return std::string("");} // parameters->string
            virtual void deserialize(const std::string& s){} // string->parameters
            virtual Base* clone() const {return new Base;}
        };

        template <typename T>
        class BinaryStamp : public Base { // decision stamp for binary features
            int dim_;
            T th_;
            int d_;
        public:
            BinaryStamp() : dim_(1), th_(0), d_(0){}
            BinaryStamp(const int dim, const int th = 0)
                :  dim_(dim), th_(th), d_(0){};
            bool branch(const std::vector< std::pair<int, T> >& x) const {
                for(int i=0; i < x.size(); i++){
                    if(x[i].first == d_) return x[i].second > th_;
                }
                return  0 > th_;
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
            Base* clone() const { return new BinaryStamp(dim_, th_); }
        };
        
    } // testfuncs
} // octrf
