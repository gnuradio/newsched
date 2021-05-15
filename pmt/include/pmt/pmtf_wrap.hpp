#pragma once

#include <pmt/pmtf_map.hpp>
#include <pmt/pmtf_vector.hpp>
#include <pmt/pmtf_scalar.hpp>

namespace pmtf {

class pmt_wrap {
  // Generic class to wrap a pmt.
  // Should accept the following types in the constructor:
  //  1) Any pmt shared_pointer
  //  2) Any pmt wrapper class
  //  3) Almost any object that can be used to construct one of the pmt classes.
  //      I don't want to deal with initializer lists.
  public:
    pmt_wrap() : d_ptr(nullptr) {}
    template <class T, class alloc>
    pmt_wrap(const std::vector<T, alloc>& x) {
        auto value = pmt_vector(x);
        d_ptr = value.ptr();
    }
    pmt_wrap(pmt_base::sptr x): d_ptr(x) {}
    //template <class T>
    //pmt_wrap(const std::map<std::string, T>& x);
    template <class T>
    pmt_wrap(const T& x) {
        auto value = pmt_scalar(x);
        d_ptr = value.ptr();  
    };
    operator typename pmt_base::sptr() const { return d_ptr; }
    typename pmt_base::sptr ptr() { return d_ptr; }
  private:
        pmt_base::sptr d_ptr;
};

}




