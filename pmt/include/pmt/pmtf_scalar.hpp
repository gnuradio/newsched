#pragma once

#include <pmt/pmt_generated.h>
#include <pmt/pmtf.hpp>
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
