#pragma once

#include <pmt/pmt_generated.h>
#include <pmt/pmtf.hpp>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>


namespace pmtf {

template <class T>
class pmt_scalar : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_scalar> sptr;
    static sptr make(const T value) { return std::make_shared<pmt_scalar<T>>(value); }
    static sptr from_buffer(const uint8_t* buf)
    {
        return std::make_shared<pmt_scalar<T>>(buf);
    }
    static sptr from_pmt(const pmtf::Pmt* fb_pmt)
    {
        return std::make_shared<pmt_scalar<T>>(fb_pmt);
    }

    void set_value(T val);
    T value();

    void operator=(const T& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const T& other) { return other == value(); }
    bool operator!=(const T& other) { return other != value(); }

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

    pmt_scalar(const T& val);
    pmt_scalar(const uint8_t* buf);
    pmt_scalar(const pmtf::Pmt* fb_pmt);
};


#define IMPLEMENT_PMT_SCALAR(datatype, fbtype)                      \
    template <>                                                     \
    datatype pmt_scalar<datatype>::value()                          \
    {                                                               \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());     \
        return pmt->data_as_Scalar##fbtype()->value();              \
    }                                                               \
                                                                    \
    template <>                                                     \
    flatbuffers::Offset<void> pmt_scalar<datatype>::rebuild_data(   \
        flatbuffers::FlatBufferBuilder& fbb)                        \
    {                                                               \
        Scalar##fbtype##Builder sb(fbb);                            \
        auto val = value();                                         \
        sb.add_value(val);                                          \
        return sb.Finish().Union();                                 \
    }                                                               \
                                                                    \
    template <>                                                     \
    void pmt_scalar<datatype>::set_value(datatype val)              \
    {                                                               \
        Scalar##fbtype##Builder sb(_fbb);                           \
        sb.add_value(val);                                          \
        _data = sb.Finish().Union();                                \
        build();                                                    \
    }                                                               \
                                                                    \
    template <>                                                     \
    pmt_scalar<datatype>::pmt_scalar(const datatype& val)           \
        : pmt_base(Data::Scalar##fbtype)                            \
    {                                                               \
        set_value(val);                                             \
    }                                                               \
                                                                    \
    template <>                                                     \
    pmt_scalar<datatype>::pmt_scalar(const uint8_t* buf)            \
        : pmt_base(Data::Scalar##fbtype)                            \
    {                                                               \
        auto data = GetPmt(buf)->data_as_Scalar##fbtype()->value(); \
        set_value(data);                                            \
    }                                                               \
                                                                    \
    template <>                                                     \
    pmt_scalar<datatype>::pmt_scalar(const pmtf::Pmt* fb_pmt)       \
        : pmt_base(Data::Scalar##fbtype)                            \
    {                                                               \
        auto data = fb_pmt->data_as_Scalar##fbtype()->value();      \
        set_value(data);                                            \
    }                                                               \
                                                                    \
    template class pmt_scalar<datatype>;


#define IMPLEMENT_PMT_SCALAR_CPLX(datatype, fbtype)                    \
    template <>                                                        \
    datatype pmt_scalar<datatype>::value()                             \
    {                                                                  \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());        \
        return *((datatype*)(pmt->data_as_Scalar##fbtype()->value())); \
    }                                                                  \
                                                                       \
    template <>                                                        \
    flatbuffers::Offset<void> pmt_scalar<datatype>::rebuild_data(      \
        flatbuffers::FlatBufferBuilder& fbb)                           \
    {                                                                  \
        Scalar##fbtype##Builder sb(fbb);                               \
        auto val = value();                                            \
        sb.add_value((fbtype*)&val);                                   \
        return sb.Finish().Union();                                    \
    }                                                                  \
                                                                       \
    template <>                                                        \
    void pmt_scalar<datatype>::set_value(datatype val)                 \
    {                                                                  \
        Scalar##fbtype##Builder sb(_fbb);                              \
        sb.add_value((fbtype*)&val);                                   \
        _data = sb.Finish().Union();                                   \
        build();                                                       \
    }                                                                  \
                                                                       \
    template <>                                                        \
    pmt_scalar<datatype>::pmt_scalar(const datatype& val)              \
        : pmt_base(Data::Scalar##fbtype)                               \
    {                                                                  \
        set_value(val);                                                \
    }                                                                  \
                                                                       \
    template <>                                                        \
    pmt_scalar<datatype>::pmt_scalar(const uint8_t* buf)               \
        : pmt_base(Data::Scalar##fbtype)                               \
    {                                                                  \
        auto data = GetPmt(buf)->data_as_Scalar##fbtype()->value();    \
        set_value(*((const datatype*)data));                           \
    }                                                                  \
                                                                       \
    template <>                                                        \
    pmt_scalar<datatype>::pmt_scalar(const pmtf::Pmt* fb_pmt)          \
        : pmt_base(Data::Scalar##fbtype)                               \
    {                                                                  \
        auto data = fb_pmt->data_as_Scalar##fbtype()->value();         \
        set_value(*((const datatype*)data));                           \
    }                                                                  \
                                                                       \
                                                                       \
    template class pmt_scalar<datatype>;

} // namespace pmtf
