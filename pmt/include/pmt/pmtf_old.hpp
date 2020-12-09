#pragma once

#include <pmt/pmt_generated.h>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace flatbuffers {
pmtf::Complex64 Pack(const std::complex<float>& obj)
{
    return pmtf::Complex64(obj.real(), obj.imag());
}

const std::complex<float> UnPack(const pmtf::Complex64& obj)
{
    return std::complex<float>(obj.re(), obj.im());
}
} // namespace flatbuffers

namespace pmtf {

class pmt_base : public std::enable_shared_from_this<pmt_base>
{
public:
    typedef std::shared_ptr<pmt_base> sptr;
    pmt_base(Data data_type)
        : _data_type(data_type){

          };
    Data data_type() const { return _data_type; };

    bool serialize(std::streambuf& sb)
    {
        uint8_t* buf = _fbb.GetBufferPointer();
        int size = _fbb.GetSize();
        return sb.sputn((const char*)buf, size) != std::streambuf::traits_type::eof();
    }
    static sptr deserialize(std::streambuf& sb)
    {
        char buf[4];
        sb.sgetn(buf, 4);
        // assuming little endian for now
        uint32_t size = *((uint32_t*)&buf[0]);
        uint8_t tmp_buf[size];
        sb.sgetn((char*)tmp_buf, size);
        auto pmt = GetPmt(tmp_buf);

        return std::make_shared<pmt_base>(pmt->data_type());
    }

    void build()
    {
        // std::cout << "fb size: " << _fbb.GetSize() << std::endl;
        PmtBuilder pb(_fbb);
        pb.add_data_type(_data_type);
        pb.add_data(_data);
        _blob = pb.Finish();
        _fbb.FinishSizePrefixed(_blob);
        // std::cout << "fb size: " << _fbb.GetSize() << std::endl;
    }

protected:
    Data _data_type;
    flatbuffers::FlatBufferBuilder _fbb;
    flatbuffers::Offset<void> _data;
    flatbuffers::Offset<Pmt> _blob;
    // PmtBuilder _builder;
};

typedef pmt_base::sptr pmt_sptr;


template <class T>
class pmt_scalar : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_scalar> sptr;
    static sptr make(const T value)
    {
        return std::make_shared<pmt_scalar<T>>(pmt_scalar<T>(value));
    }

    pmt_scalar(const T& val);
    void set_value(T val);
    T value();

    void operator=(const T& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const T& other) { return other == value(); }
    bool operator!=(const T& other) { return other != value(); }

};

template <class T>
class pmt_vector : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_vector> sptr;
    static sptr make(const T value)
    {
        return std::make_shared<pmt_vector<T>>(pmt_vector<T>(value));
    }

    pmt_vector(const std::vector<T>& val);

    void set_value(const std::vector<T>& val);
    // void deserialize(std::streambuf& sb) override;
    std::vector<T>& value();
    const T* data();
    size_t size();

    void operator=(const std::vector<T>& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const std::vector<T>& other) { return other == value(); }
    bool operator!=(const std::vector<T>& other) { return other != value(); }


private:
    std::vector<T> _value;
};

} // namespace pmtf

#define IMPLEMENT_PMT_SCALAR(datatype, fbtype)                  \
    template <>                                                 \
    void pmt_scalar<datatype>::set_value(datatype val)          \
    {                                                           \
        Scalar##fbtype##Builder sb(_fbb);                       \
        sb.add_value(val);                                      \
        _data = sb.Finish().Union();                            \
        build();                                                \
    }                                                           \
                                                                \
    template <>                                                 \
    pmt_scalar<datatype>::pmt_scalar(const datatype& val)       \
        : pmt_base(Data::Scalar##fbtype)                        \
    {                                                           \
        set_value(val);                                         \
    }                                                           \
                                                                \
    template <>                                                 \
    datatype pmt_scalar<datatype>::value()                      \
    {                                                           \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer()); \
        return pmt->data_as_Scalar##fbtype()->value();          \
    }

#define IMPLEMENT_PMT_SCALAR_CPLX(datatype, fbtype)                    \
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
    datatype pmt_scalar<datatype>::value()                             \
    {                                                                  \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());        \
        return *((datatype*)(pmt->data_as_Scalar##fbtype()->value())); \
    }


#define IMPLEMENT_PMT_VECTOR(datatype, fbtype)                                     \
    template <>                                                                    \
    void pmt_vector<datatype>::set_value(const std::vector<datatype>& val)         \
    {                                                                              \
        auto vec = _fbb.CreateVector(val.data(), val.size());                      \
        Vector##fbtype##Builder vb(_fbb);                                          \
        vb.add_value(vec);                                                         \
        _data = vb.Finish().Union();                                               \
        build();                                                                   \
    }                                                                              \
                                                                                   \
    /*    template <>                                                              \
        void pmt_vector<datatype>::deserialize(std::streambuf& sb)                 \
        {                                                                          \
            auto pmt = GetSizePrefixedPmt(sb);                                     \
            auto fb_vec = pmt->data_as_Vector##fbtype()->value();                  \
            _value.assign(fb_vec->begin(), fb_vec->end());                         \
        }                                                                      \*/ \
                                                                                   \
                                                                                   \
    template <>                                                                    \
    pmt_vector<datatype>::pmt_vector(const std::vector<datatype>& val)             \
        : pmt_base(Data::Vector##fbtype)                                           \
    {                                                                              \
        set_value(val);                                                            \
    }                                                                              \
                                                                                   \
    template <>                                                                    \
    std::vector<datatype>& pmt_vector<datatype>::value()                           \
    {                                                                              \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());                    \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                      \
        _value.assign(fb_vec->begin(), fb_vec->end());                             \
        return _value;                                                             \
    }                                                                              \
                                                                                   \
    template <>                                                                    \
    const datatype* pmt_vector<datatype>::data()                                   \
    {                                                                              \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());                    \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                      \
        return fb_vec->data();                                                     \
    }                                                                              \
                                                                                   \
    template <>                                                                    \
    size_t pmt_vector<datatype>::size()                                            \
    {                                                                              \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());                    \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                      \
        return fb_vec->size();                                                     \
    }

#define IMPLEMENT_PMT_VECTOR_CPLX(datatype, fbtype)                        \
    template <>                                                            \
    void pmt_vector<datatype>::set_value(const std::vector<datatype>& val) \
    {                                                                      \
        auto vec = _fbb.CreateVector((fbtype*)val.data(), val.size());     \
        Vector##fbtype##Builder vb(_fbb);                                  \
        vb.add_value(vec);                                                 \
        _data = vb.Finish().Union();                                       \
        build();                                                           \
    }                                                                      \
                                                                           \
    template <>                                                            \
    pmt_vector<datatype>::pmt_vector(const std::vector<datatype>& val)     \
        : pmt_base(Data::Vector##fbtype)                                   \
    {                                                                      \
        set_value(val);                                                    \
    }                                                                      \
                                                                           \
    template <>                                                            \
    std::vector<datatype>& pmt_vector<datatype>::value()                   \
    {                                                                      \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();              \
        _value.assign(fb_vec->begin(), fb_vec->end());                     \
        return _value;                                                     \
    }                                                                      \
                                                                           \
    template <>                                                            \
    const datatype* pmt_vector<datatype>::data()                           \
    {                                                                      \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();              \
        return fb_vec->data();                                             \
    }                                                                      \
                                                                           \
    template <>                                                            \
    size_t pmt_vector<datatype>::size()                                    \
    {                                                                      \
        auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();              \
        return fb_vec->size();                                             \
    }
