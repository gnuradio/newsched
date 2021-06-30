#pragma once

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
    typename pmt_base::sptr ptr() const { return d_ptr; }
  private:
        pmt_base::sptr d_ptr;
};

template <class T>
pmt_scalar<T> get_scalar(const pmt_wrap& x) {
    if (x.ptr()->is_scalar())
        return pmt_scalar<T>(std::dynamic_pointer_cast<pmt_scalar_value<T>>(x.ptr()));
    else
        throw std::runtime_error("Cannot cast pmt to scalar<T>");
}

template <class T>
pmt_vector<T> get_vector(const pmt_wrap& x) {
    if (x.ptr()->is_vector())
        return pmt_vector<T>(std::dynamic_pointer_cast<pmt_vector_value<T>>(x.ptr()));
    else
        throw std::runtime_error("Cannot cast pmt to vector<T>");
}

std::ostream& operator<<(std::ostream& os, const pmt_wrap& x);

/*template <class U>
bool operator==(const pmt_wrap& x1, const U& x2) {
    // We need to try and make the type match up.
    if constexpr(std::is_arithmetic_v<U>)
        if (x.ptr()->is_scalar())
            return get_scalar<U>
    return x1.ptr() == x2;
}*/
/*template <class U>
bool operator!=(const pmt_wrap& x1, const U& x2) {
    return !(x1 == x2);
}*/
}
#include <pmt/pmtf_map.hpp>

/* How to handle a generic container?
1) Have it hold a pmt ptr.  It has to dynamically ask anything it wants to know.
2) We could do a hybrid pmt_vector class that takes in any pmt_vector_value.  Then I can cast a
  pmt to a pmt_vector.  What would a pmt_vector iterator do?  You would still have to know the 
  type to do anything useful.
  

*/




