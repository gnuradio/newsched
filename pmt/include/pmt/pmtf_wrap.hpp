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
    template <class T, class alloc>
    pmt_wrap(const std::vector<T, alloc>& x) {
        auto value = pmt_vector_wrapper(x);
        d_ptr = value.ptr();
    }
    //template <class T>
    //pmt_wrap(const std::map<std::string, T>& x);
    template <class T>
    pmt_wrap(const T& x) {
        auto value = pmt_scalar_wrapper(x);
        d_ptr = value.ptr();  
    };
    typename pmt_base::sptr ptr() { return ptr; }
  private:
        pmt_base::sptr d_ptr;
};

template <class T>
class pmt_map_wrapper {
public:
    using key_type = T;
    using mapped_type = pmt_sptr;
    using value_type = std::pair<const key_type, mapped_type>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using map_type = std::map<key_type, mapped_type>; 
    // Don't allow for custom allocators.  Flatbuffer is our allocator
    pmt_map_wrapper(): d_ptr(pmt_map<T>::make(map_type())) {}
    template <class InputIterator>
    pmt_map_wrapper(InputIterator first, InputIterator last): 
        d_ptr(pmt_map<T>::make(map_type(first, last))) {}
    // TODO: Allow for custom allocators like volk.
    pmt_map_wrapper(const map_type& x):
        d_ptr(pmt_map<T>::make(x)) {}
    pmt_map_wrapper(std::initializer_list<value_type> il):
        d_ptr(pmt_map<T>::make(map_type(il))) {}

    pmt_map_wrapper(typename pmt_map<T>::sptr ptr) :
        d_ptr(ptr) {}
    pmt_map_wrapper(const pmt_map_wrapper& x) :
        d_ptr(x.d_ptr) {}

    //mapped_type& operator[] (const key_type& k) {
        // This is tricky.  We need to return a "useable" object here.
        // For example:
        //  map["key"] = 4 should do the right thing.
        //  which is create an entry for map["key"] if it doesn't exist.
        //  which will create a null pmt_sptr
        //  Then we will call operator equals with 4 as an argument.
        //  So I need a factory function, that will return the proper type of
        //  sptr based upon input.
    //}
    
private:
    typename pmt_map<T>::sptr d_ptr;
};
}
