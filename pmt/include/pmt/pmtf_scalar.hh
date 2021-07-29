#pragma once

#include <pmt/pmt_generated.h>
#include <pmt/pmtf.hh>
#include <pmt/pmtf_wrap.hh>
#include <complex>
#include <ostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <type_traits>


namespace pmtf {

/**
 * @brief Class holds the implementation of a scalar pmt.
 *
 * The scalar types are defined in pmtf_scalar.cpp.  This class should not be
 * used directly. It would be nice to move it to the .cpp and instantiate all
 * of the templates in there. This would involve a fairly large refactoring of
 * the code.
 */
template <class T>
class pmt_scalar_value : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_scalar_value> sptr;
    static sptr make(const T value) { return std::make_shared<pmt_scalar_value<T>>(value); }
    static sptr from_buffer(const uint8_t* buf)
    {
        return std::make_shared<pmt_scalar_value<T>>(buf);
    }
    static sptr from_pmt(const pmtf::Pmt* fb_pmt)
    {
        return std::make_shared<pmt_scalar_value<T>>(fb_pmt);
    }

    void set_value(const T& val);
    T value();
    const T value() const;

    pmt_scalar_value& operator=(const T& other) // copy assignment
    {
        set_value(other);
        return *this;
    }
    pmt_scalar_value& operator=(const pmt_scalar_value& other)
    {
        if (this == &other) return *this;
        this->set_value(other.value());
        return *this;
    }

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

    pmt_scalar_value(const T& val);
    pmt_scalar_value(const uint8_t* buf);
    pmt_scalar_value(const pmtf::Pmt* fb_pmt);

    bool is_scalar() const noexcept { return true; }
    void print(std::ostream& os) const { os << value(); }
    
};

// These structures allow us to see if a arbitrary type is a pmt_scalar_value
// or not.
template <class T>
struct is_pmt_scalar_value : std::false_type {};

template <class T>
struct is_pmt_scalar_value<pmt_scalar_value<T>> : std::true_type {};

/**
 * @brief compare pmt_scalar_value against something else
 *
 * Allow for comparisons against other pmt scalars and other types.
 * For example pmt_scalar_value<int>(4) == 4.0 will be true.
 */
template <class T, class U>
bool operator==(const pmt_scalar_value<T>& x, const U& y) {
    if constexpr(std::is_same_v<T, U>)
        return x.value() == y;
    else if constexpr(is_pmt_scalar_value<U>::value)
        return x.value() == y.value();
    else if constexpr(std::is_convertible_v<U, T>)
        return x.value() == T(y);
    return false;
}

// These structures allow us to write template functions that depend on the
// flatbuffer data type.  This allows us to do things like verify that the
// datatype is correct when we want to interpret a pmt as another type.
template <> struct cpp_type<Data::ScalarInt8> { using type=int8_t; };
template <> struct cpp_type<Data::ScalarInt16> { using type=int16_t; };
template <> struct cpp_type<Data::ScalarInt32> { using type=int32_t; };
template <> struct cpp_type<Data::ScalarInt64> { using type=int64_t; };
template <> struct cpp_type<Data::ScalarUInt8> { using type=uint8_t; };
template <> struct cpp_type<Data::ScalarUInt16> { using type=uint16_t; };
template <> struct cpp_type<Data::ScalarUInt32> { using type=uint32_t; };
template <> struct cpp_type<Data::ScalarUInt64> { using type=uint64_t; };
template <> struct cpp_type<Data::ScalarFloat32> { using type=float; };
template <> struct cpp_type<Data::ScalarFloat64> { using type=double; };
template <> struct cpp_type<Data::ScalarComplex64> { using type=std::complex<float>; };
template <> struct cpp_type<Data::ScalarComplex128> { using type=std::complex<double>; };
template <> struct cpp_type<Data::ScalarBool> { using type=bool; };

/**
 * @brief "Print" out a pmt_scalar_value
 */
template <class T>
std::ostream& operator<<(std::ostream& os, const pmt_scalar_value<T>& value) {
    os << value;
    return os;
}

/**
 * @brief Wrapper class around a smart pointer to a pmt_scalar_value.
 *
 * This is the interface that should be used for dealing with scalar values.
 */
template <class T>
class pmt_scalar {
public:
    using sptr = typename pmt_scalar_value<T>::sptr;
    //! Construct a pmt_scalar from a scalar value
    pmt_scalar(const T& val): d_ptr(pmt_scalar_value<T>::make(val)) {}
    //! Construct a pmt_scalar from a pmt_scalar_value pointer.
    pmt_scalar(sptr ptr):
        d_ptr(ptr) {}
    //! Copy constructor.
    pmt_scalar(const pmt_scalar<T>& x):
        d_ptr(x.d_ptr) {}
   
    //! Get at the smart pointer.
    sptr ptr() const { return d_ptr; }
    bool operator==(const T& val) const { return *d_ptr == val;}
    bool operator==(const pmt_scalar<T>& val) const { return *d_ptr == *val.d_ptr; }
    auto data_type() { return d_ptr->data_type(); }
    T value() const { return d_ptr->value(); }


    // Make it act like a pointer.  Probably need a better way
    // to think about it.
    T& operator*() const { return *d_ptr; }
    // Cast operators
    //! Cast to a T value.
    //! Explicit means that the user must do something like T(pmt_scalar<T>(val));
    //! Implicit conversions can cause lots of problems, so we are avoiding them.
    explicit operator T() const { return d_ptr->value(); }
    //! Cast to another type
    //! Will cause a compilation failure if we can't do the cast.
    template <class U>
    explicit operator U() const { return U(d_ptr->value()); }
    
private:
    sptr d_ptr;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const pmt_scalar<T>& value) {
    os << *(value.ptr());
    return os;
}

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
// Fix this later with SFINAE
template <class T, Data dt>
bool _can_be(const pmt_wrap& x) {
    auto value = get_pmt_scalar<typename cpp_type<dt>::type>(x);
    return std::is_convertible_v<typename cpp_type<dt>::type, T>;
}

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

template <class T, Data dt>
T _get_as(const pmt_wrap& x) {
    auto value = get_scalar<typename cpp_type<dt>::type>(x);
    if constexpr(std::is_convertible_v<typename cpp_type<dt>::type, T>)
        return T(value.ptr()->value());
    else
        throw std::runtime_error("Cannot convert types");
}

template <class T>
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
    }
}

// Define constructors for pmt_wrap for the scalar types
// In c++20, I think we could do this with a concept.
// I'm struggling to get SFINAE working.  I'm not sure if it is possible here, so I'm using macros.  Sorry.
// Construct a pmt_wrap from a scalar type
#define WrapConstruct(type) \
    template <> pmt_wrap::pmt_wrap<type>(const type& x);
// Construct a pmt_wrap from a pmt_scalar
#define WrapConstructPmt(type) \
    template <> pmt_wrap::pmt_wrap<pmt_scalar<type>>(const pmt_scalar<type>& x);

#define Equals(type) \
    template <> bool operator==<type>(const pmt_wrap& x, const type& other);

#define EqualsPmt(type) \
    template <> bool operator==<pmt_scalar<type>>(const pmt_wrap& x, const pmt_scalar<type>& other);

#define Apply(func) \
func(uint8_t) \
func(uint16_t) \
func(uint32_t) \
func(uint64_t) \
func(int8_t) \
func(int16_t) \
func(int32_t) \
func(int64_t) \
func(bool) \
func(float) \
func(double) \
func(std::complex<float>) \
func(std::complex<double>)

Apply(WrapConstruct)
Apply(WrapConstructPmt)
Apply(Equals)
Apply(EqualsPmt)

#undef WrapConstruct
#undef WrapConstantPmt
#undef Equals
#undef EqualsPmt
#undef Apply

#define IMPLEMENT_PMT_SCALAR(datatype, fbtype)                      \
    template <>                                                     \
    datatype pmt_scalar_value<datatype>::value()                          \
    {                                                               \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());     \
        return pmt->data_as_Scalar##fbtype()->value();              \
    }                                                               \
                                                                    \
    template <>                                                     \
    const datatype pmt_scalar_value<datatype>::value() const              \
    {                                                               \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());     \
        return pmt->data_as_Scalar##fbtype()->value();              \
    }                                                               \
                                                                    \
    template <>                                                     \
    flatbuffers::Offset<void> pmt_scalar_value<datatype>::rebuild_data(   \
        flatbuffers::FlatBufferBuilder& fbb)                        \
    {                                                               \
        Scalar##fbtype##Builder sb(fbb);                            \
        auto val = value();                                         \
        sb.add_value(val);                                          \
        return sb.Finish().Union();                                 \
    }                                                               \
                                                                    \
    template <>                                                     \
    void pmt_scalar_value<datatype>::set_value(const datatype& val)       \
    {                                                               \
        Scalar##fbtype##Builder sb(_fbb);                           \
        sb.add_value(val);                                          \
        _data = sb.Finish().Union();                                \
        build();                                                    \
    }                                                               \
                                                                    \
    template <>                                                     \
    pmt_scalar_value<datatype>::pmt_scalar_value(const datatype& val)           \
        : pmt_base(Data::Scalar##fbtype)                            \
    {                                                               \
        set_value(val);                                             \
    }                                                               \
                                                                    \
    template <>                                                     \
    pmt_scalar_value<datatype>::pmt_scalar_value(const uint8_t* buf)            \
        : pmt_base(Data::Scalar##fbtype)                            \
    {                                                               \
        auto data = GetPmt(buf)->data_as_Scalar##fbtype()->value(); \
        set_value(data);                                            \
    }                                                               \
                                                                    \
    template <>                                                     \
    pmt_scalar_value<datatype>::pmt_scalar_value(const pmtf::Pmt* fb_pmt)       \
        : pmt_base(Data::Scalar##fbtype)                            \
    {                                                               \
        auto data = fb_pmt->data_as_Scalar##fbtype()->value();      \
        set_value(data);                                            \
    }                                                               \
                                                                    \
    template class pmt_scalar_value<datatype>;


#define IMPLEMENT_PMT_SCALAR_CPLX(datatype, fbtype)                    \
    template <>                                                        \
    datatype pmt_scalar_value<datatype>::value()                             \
    {                                                                  \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());        \
        return *((datatype*)(pmt->data_as_Scalar##fbtype()->value())); \
    }                                                                  \
                                                                       \
    template <>                                                        \
    const datatype pmt_scalar_value<datatype>::value() const                 \
    {                                                                  \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());        \
        return *((datatype*)(pmt->data_as_Scalar##fbtype()->value())); \
    }                                                                  \
                                                                       \
    template <>                                                        \
    flatbuffers::Offset<void> pmt_scalar_value<datatype>::rebuild_data(      \
        flatbuffers::FlatBufferBuilder& fbb)                           \
    {                                                                  \
        Scalar##fbtype##Builder sb(fbb);                               \
        auto val = value();                                            \
        sb.add_value((fbtype*)&val);                                   \
        return sb.Finish().Union();                                    \
    }                                                                  \
                                                                       \
    template <>                                                        \
    void pmt_scalar_value<datatype>::set_value(const datatype& val)          \
    {                                                                  \
        Scalar##fbtype##Builder sb(_fbb);                              \
        sb.add_value((fbtype*)&val);                                   \
        _data = sb.Finish().Union();                                   \
        build();                                                       \
    }                                                                  \
                                                                       \
    template <>                                                        \
    pmt_scalar_value<datatype>::pmt_scalar_value(const datatype& val)              \
        : pmt_base(Data::Scalar##fbtype)                               \
    {                                                                  \
        set_value(val);                                                \
    }                                                                  \
                                                                       \
    template <>                                                        \
    pmt_scalar_value<datatype>::pmt_scalar_value(const uint8_t* buf)               \
        : pmt_base(Data::Scalar##fbtype)                               \
    {                                                                  \
        auto data = GetPmt(buf)->data_as_Scalar##fbtype()->value();    \
        set_value(*((const datatype*)data));                           \
    }                                                                  \
                                                                       \
    template <>                                                        \
    pmt_scalar_value<datatype>::pmt_scalar_value(const pmtf::Pmt* fb_pmt)          \
        : pmt_base(Data::Scalar##fbtype)                               \
    {                                                                  \
        auto data = fb_pmt->data_as_Scalar##fbtype()->value();         \
        set_value(*((const datatype*)data));                           \
    }                                                                  \
                                                                       \
                                                                       \
    template class pmt_scalar_value<datatype>;

} // namespace pmtf
