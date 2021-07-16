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


// Fix this later with SFINAE
/*template <class T, Data dt>
T _get_as(const pmt_wrap& x) {
    auto value = get_scalar<typename cpp_type<dt>::type>(x);
    if constexpr(std::is_convertible_v<typename cpp_type<dt>::type, T>)
        return T(value.ptr()->value());
    else
        throw std::runtime_error("Cannot convert types");
}*/

/*template <class T, Data dt>
T _get_as_vector(const pmt_wrap& x) {
    auto value = get_vector<typename cpp_type<dt>::type>(x);
    if constexpr(std::is_same_v<typename cpp_type<dt>::type, T>)
        return T(value.ptr()->value());
    else
        throw std::runtime_error("Cannot convert vector types");
}*/

template <class T, Data dt>
pmt_scalar<T> _get_pmt_scalar(const pmt_wrap& x) {
    if constexpr(std::is_same_v<typename cpp_type<dt>::type, T>)
        return pmt_scalar<T>(std::dynamic_pointer_cast<pmt_scalar_value<T>>(x.ptr()));
    else
        throw std::runtime_error("Cannot convert scalar types");
}

template <class T>
pmt_scalar<T> get_pmt_scalar(const pmt_wrap& x) {
    // Make sure that this is the right type.
    switch(auto dt = x.ptr()->data_type()) {
        case Data::ScalarFloat32: return _get_pmt_scalar<T, Data::ScalarFloat32>(x);
        case Data::ScalarFloat64: return _get_pmt_scalar<T, Data::ScalarFloat64>(x);
        case Data::ScalarComplex64: return _get_pmt_scalar<T, Data::ScalarComplex64>(x);
        case Data::ScalarComplex128: return _get_pmt_scalar<T, Data::ScalarComplex128>(x);
        case Data::ScalarInt8: return _get_pmt_scalar<T, Data::ScalarInt8>(x);
        case Data::ScalarInt16: return _get_pmt_scalar<T, Data::ScalarInt16>(x);
        case Data::ScalarInt32: return _get_pmt_scalar<T, Data::ScalarInt32>(x);
        case Data::ScalarInt64: return _get_pmt_scalar<T, Data::ScalarInt64>(x);
        case Data::ScalarUInt8: return _get_pmt_scalar<T, Data::ScalarUInt8>(x);
        case Data::ScalarUInt16: return _get_pmt_scalar<T, Data::ScalarUInt16>(x);
        case Data::ScalarUInt32: return _get_pmt_scalar<T, Data::ScalarUInt32>(x);
        case Data::ScalarUInt64: return _get_pmt_scalar<T, Data::ScalarUInt64>(x);
        case Data::ScalarBool: return _get_pmt_scalar<T, Data::ScalarBool>(x);
        default:
            throw std::runtime_error("Cannot convert non scalar pmt.");
    }
}

template <class T>
T get_scalar(const pmt_wrap& x) {
    return get_pmt_scalar<T>(x).ptr()->value();
}
    
template <class T, Data dt>
pmt_scalar<T> _get_pmt_vector(const pmt_wrap& x) {
    if constexpr(std::is_same_v<typename cpp_type<dt>::type, T>)
        return pmt_vector<T>(std::dynamic_pointer_cast<pmt_vector_value<T>>(x.ptr()));
    else
        throw std::runtime_error("Cannot convert vector types");
}

template <class T>
pmt_vector<T> get_pmt_vector(const pmt_wrap& x) {
    // TODO: I can flip this around and make functions to convert T to a dt at compile time.
    //   Then just check if vector_data_type<T> == x.ptr()->data_type()
    // Make sure that this is the right type.
    switch(auto dt = x.ptr()->data_type()) {
        case Data::VectorFloat32: return _get_pmt_vector<T, Data::VectorFloat32>(x);
        case Data::VectorFloat64: return _get_pmt_vector<T, Data::VectorFloat64>(x);
        case Data::VectorComplex64: return _get_pmt_vector<T, Data::VectorComplex64>(x);
        case Data::VectorComplex128: return _get_pmt_vector<T, Data::VectorComplex128>(x);
        case Data::VectorInt8: return _get_pmt_vector<T, Data::VectorInt8>(x);
        case Data::VectorInt16: return _get_pmt_vector<T, Data::VectorInt16>(x);
        case Data::VectorInt32: return _get_pmt_vector<T, Data::VectorInt32>(x);
        case Data::VectorInt64: return _get_pmt_vector<T, Data::VectorInt64>(x);
        case Data::VectorUInt8: return _get_pmt_vector<T, Data::VectorUInt8>(x);
        case Data::VectorUInt16: return _get_pmt_vector<T, Data::VectorUInt16>(x);
        case Data::VectorUInt32: return _get_pmt_vector<T, Data::VectorUInt32>(x);
        case Data::VectorUInt64: return _get_pmt_vector<T, Data::VectorUInt64>(x);
        case Data::VectorBool: return _get_pmt_vector<T, Data::VectorBool>(x);
        default:
            throw std::runtime_error("Cannot convert non scalar pmt.");
    }
}

/*pmt_map<std::string> get_map(const pmt_wrap& x) {
    if (x.ptr()->data_type() == Data::PmtMap) {
        return pmt_map<std::string>(std::dynamic_pointer_cast<pmt_map<std::string>>(x.ptr()));
    else
        throw std::runtime_erro("Cannot convert to map");
}*/

/*template <class T>
T get_as(const pmt_wrap& x) {
    switch(auto dt = x.ptr()->data_type()) {
        case Data::ScalarFloat32: return _get_as<T, Data::ScalarFloat32>(x);
        case Data::ScalarFloat64: return _get_as<T, Data::ScalarFloat64>(x);
        case Data::ScalarComplex64: return _get_as<T, Data::ScalarComplex64>(x);
        case Data::ScalarComplex128: return _get_as<T, Data::ScalarComplex128>(x);
        case Data::ScalarInt8: return _get_as<T, Data::ScalarInt8>(x);
        case Data::ScalarInt16: return _get_as<T, Data::ScalarInt16>(x);
        case Data::ScalarInt32: return _get_as<T, Data::ScalarInt32>(x);
        case Data::ScalarInt64: return _get_as<T, Data::ScalarInt64>(x);
        case Data::ScalarUInt8: return _get_as<T, Data::ScalarUInt8>(x);
        case Data::ScalarUInt16: return _get_as<T, Data::ScalarUInt16>(x);
        case Data::ScalarUInt32: return _get_as<T, Data::ScalarUInt32>(x);
        case Data::ScalarUInt64: return _get_as<T, Data::ScalarUInt64>(x);
        case Data::ScalarBool: return _get_as<T, Data::ScalarBool>(x);
        case Data::VectorFloat32: return _get_as<T, Data::VectorFloat32>(x);
        case Data::VectorFloat64: return _get_as<T, Data::VectorFloat64>(x);
        case Data::VectorComplex64: return _get_as<T, Data::VectorComplex64>(x);
        case Data::VectorComplex128: return _get_as<T, Data::VectorComplex128>(x);
        case Data::VectorInt8: return _get_as<T, Data::VectorInt8>(x);
        case Data::VectorInt16: return _get_as<T, Data::VectorInt16>(x);
        case Data::VectorInt32: return _get_as<T, Data::VectorInt32>(x);
        case Data::VectorInt64: return _get_as<T, Data::VectorInt64>(x);
        case Data::VectorUInt8: return _get_as<T, Data::VectorUInt8>(x);
        case Data::VectorUInt16: return _get_as<T, Data::VectorUInt16>(x);
        case Data::VectorUInt32: return _get_as<T, Data::VectorUInt32>(x);
        case Data::VectorUInt64: return _get_as<T, Data::VectorUInt64>(x);
        case Data::VectorBool: return _get_as<T, Data::VectorBool>(x);
    }
}*/

// Fix this later with SFINAE
template <class T, Data dt>
bool _can_be(const pmt_wrap& x) {
    auto value = get_pmt_scalar<typename cpp_type<dt>::type>(x);
    return std::is_convertible_v<typename cpp_type<dt>::type, T>;
}
// What does this function mean?
// Scalar - Simple can we convert the underlying type.
// Vector - Could be one of
//  Does the type match exactly?
//  Could I make an std::vector with this data?
//     With or without conversions
// I think this only makes sense with scalars.

template <class T>
bool can_be(const pmt_wrap& x) {
    switch(auto dt = x.ptr()->data_type()) {
        case Data::ScalarFloat32: return _can_be<T, Data::ScalarFloat32>(x);
        case Data::ScalarFloat64: return _can_be<T, Data::ScalarFloat64>(x);
        case Data::ScalarComplex64: return _can_be<T, Data::ScalarComplex64>(x);
        case Data::ScalarComplex128: return _can_be<T, Data::ScalarComplex128>(x);
        case Data::ScalarInt8: return _can_be<T, Data::ScalarInt8>(x);
        case Data::ScalarInt16: return _can_be<T, Data::ScalarInt16>(x);
        case Data::ScalarInt32: return _can_be<T, Data::ScalarInt32>(x);
        case Data::ScalarInt64: return _can_be<T, Data::ScalarInt64>(x);
        case Data::ScalarUInt8: return _can_be<T, Data::ScalarUInt8>(x);
        case Data::ScalarUInt16: return _can_be<T, Data::ScalarUInt16>(x);
        case Data::ScalarUInt32: return _can_be<T, Data::ScalarUInt32>(x);
        case Data::ScalarUInt64: return _can_be<T, Data::ScalarUInt64>(x);
        case Data::ScalarBool: return _can_be<T, Data::ScalarBool>(x);
        //case Data::PmtString: return _can_be<T, Data::PmtString>(x);
        default: return false;
    }
    
}

template <class T>
bool operator==(const pmt_wrap& x, const T& other) {
    if (can_be<T>(x)) {
        auto value = get_pmt_scalar<T>(x);
        return x == other;
    } else
        return false;
}

template <class T>
bool operator!=(const pmt_wrap& x, const T& other) {
    return !operator==(x, other);
}

//bool operator==(const pmt_wrap& x, const pmt_wrap& other) {
//    return false;
    //throw std::runtime_error("Not implemented Yet"); 
//}

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
    }
}

std::ostream& operator<<(std::ostream& os, const pmt_wrap& x);

How to handle a generic container?
1) Have it hold a pmt ptr.  It has to dynamically ask anything it wants to know.
2) We could do a hybrid pmt_vector class that takes in any pmt_vector_value.  Then I can cast a
  pmt to a pmt_vector.  What would a pmt_vector iterator do?  You would still have to know the 
  type to do anything useful.
  

*/




