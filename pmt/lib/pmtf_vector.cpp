#include <pmt/pmtf_vector.hpp>

namespace pmtf {


//////////////////////////////////////////////////////////////////////////////////
// int_32_t vector stuff
//////////////////////////////////////////////////////////////////////////////////

template <>
flatbuffers::Offset<void>
pmt_vector<int32_t>::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    // fbb.Reset();
    auto pmt = GetSizePrefixedPmt(buffer_pointer());
    auto fb_vec = pmt->data_as_VectorInt32()->value();
    auto vec = fbb.CreateVector(fb_vec->data(), fb_vec->size());
    VectorInt32Builder vb(fbb);
    vb.add_value(vec);
    return vb.Finish().Union();
}

template <>
void pmt_vector<int32_t>::set_value(const std::vector<int32_t>& val)
{
    _fbb.Reset();
    auto vec = _fbb.CreateVector(val.data(), val.size());
    VectorInt32Builder vb(_fbb);
    vb.add_value(vec);
    _data = vb.Finish().Union();
    build();
}

template <>
void pmt_vector<int32_t>::set_value(const int32_t* data, size_t len)
{
    _fbb.Reset();
    auto vec = _fbb.CreateVector(data, len);
    VectorInt32Builder vb(_fbb);
    vb.add_value(vec);
    _data = vb.Finish().Union();
    build();
}

template <>
pmt_vector<int32_t>::pmt_vector(const std::vector<int32_t>& val)
    : pmt_base(Data::VectorInt32)
{
    set_value(val);
}

template <>
pmt_vector<int32_t>::pmt_vector(const int32_t* data, size_t len)
    : pmt_base(Data::VectorInt32)
{
    set_value(data, len);
}

template <>
pmt_vector<int32_t>::pmt_vector(const uint8_t* buf) : pmt_base(Data::VectorInt32)
{
    auto data = GetPmt(buf)->data_as_VectorInt32()->value();
    size_t len = data->size();
    set_value((const int32_t*)data->Data(), len);
}

template <>
pmt_vector<int32_t>::pmt_vector(const pmtf::Pmt* fb_pmt) : pmt_base(Data::VectorInt32)
{
    auto data = fb_pmt->data_as_VectorInt32()->value();
    size_t len = data->size();
    set_value((const int32_t*)data->Data(), len);
}

template <>
std::vector<int32_t> pmt_vector<int32_t>::value() const
{
    auto pmt = GetSizePrefixedPmt(buffer_pointer());
    auto fb_vec = pmt->data_as_VectorInt32()->value();
    // _value.assign(fb_vec->begin(), fb_vec->end());
    std::vector<int32_t> ret(fb_vec->begin(), fb_vec->end());
    return ret;
}

template <>
const int32_t* pmt_vector<int32_t>::data()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorInt32()->value();
    return fb_vec->data();
}

template <>
size_t pmt_vector<int32_t>::size()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorInt32()->value();
    return fb_vec->size();
}

template <>
int32_t pmt_vector<int32_t>::ref(size_t k)
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorInt32()->value();
    if (k >= fb_vec->size())
        throw std::runtime_error("PMT Vector index out of range");
    return (*fb_vec)[k];
}

template <>
void pmt_vector<int32_t>::set(size_t k, int32_t val)
{
    auto pmt = GetMutablePmt(buffer_pointer() + 4); // assuming size prefix is 32 bit
    auto fb_vec = ((pmtf::VectorInt32*)pmt->mutable_data())->mutable_value();
    if (k >= fb_vec->size())
        throw std::runtime_error("PMT Vector index out of range");
    fb_vec->Mutate(k, val);
}

// template <>
// pmt_vector<int32_t>::sptr pmt_vector_from_buffer<int32_t>(const uint8_t* buf)
// {
//     auto data = GetPmt(buf)->data_as_VectorInt32()->value();
//     size_t size = GetPmt(buf)->data_as_VectorInt32()->value()->size();
//     return std::make_shared<pmt_vector<int32_t>>(data->data(), size);
// }


#if 0
//////////////////////////////////////////////////////////////////////////////////
// std::complex<float> vector stuff
//////////////////////////////////////////////////////////////////////////////////

template <>
void pmt_vector<std::complex<float>>::set_value(const std::vector<std::complex<float>>& val)
{
    _fbb.Reset();
    auto vec = _fbb.CreateVectorOfNativeStructs<Complex64,std::complex<float>>(val.data(), val.size());
    VectorComplex64Builder vb(_fbb);
    vb.add_value(vec);
    _data = vb.Finish().Union();
    build();
}

template <>
void pmt_vector<std::complex<float>>::set_value(const std::complex<float>* data, size_t len)
{
    _fbb.Reset();
    auto vec = _fbb.CreateVectorOfNativeStructs<Complex64,std::complex<float>>(data, len);
    VectorComplex64Builder vb(_fbb);
    vb.add_value(vec);
    _data = vb.Finish().Union();
    build();
}

template <>
pmt_vector<std::complex<float>>::pmt_vector(const std::vector<std::complex<float>>& val)
    : pmt_base(Data::VectorComplex64)
{
    set_value(val);
}

template <>
pmt_vector<std::complex<float>>::pmt_vector(const std::complex<float>* data, size_t len)
    : pmt_base(Data::VectorComplex64)
{
    set_value(data, len);
}


template <>
std::vector<std::complex<float>> pmt_vector<std::complex<float>>::value()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    // _value.assign(fb_vec->begin(), fb_vec->end());
    std::vector<std::complex<float>> ret(fb_vec->size());

    return ret;
}

template <>
const std::complex<float>* pmt_vector<std::complex<float>>::data()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    return (std::complex<float>*)fb_vec->Data(); // no good native conversions in API, just cast here
}

template <>
size_t pmt_vector<std::complex<float>>::size()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    return fb_vec->size();
}

template <>
std::complex<float> pmt_vector<std::complex<float>>::ref(pmt_base::sptr ptr, size_t k)
{
    if (ptr->data_type() != Data::VectorComplex64)
        throw std::runtime_error("PMT not of type std::complex<float>");
    auto pmt = GetSizePrefixedPmt(ptr->buffer_pointer());
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    if (k >= fb_vec->size())
        throw std::runtime_error("PMT Vector index out of range");
    return *((std::complex<float>*)(*fb_vec)[k]);  // hacky cast
}

template <>
const std::complex<float>* pmt_vector<std::complex<float>>::elements(pmt_base::sptr ptr)
{
    if (ptr->data_type() != Data::VectorComplex64)
        throw std::runtime_error("PMT not of type std::complex<float>");
    auto pmt = GetSizePrefixedPmt(ptr->buffer_pointer());
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    return (std::complex<float>*)(fb_vec->Data());  // hacky cast
}

template <>
void pmt_vector<std::complex<float>>::set(pmt_base::sptr ptr, size_t k, std::complex<float> val)
{
    if (ptr->data_type() != Data::VectorComplex64)
        throw std::runtime_error("PMT not of type std::complex<float>");
    auto pmt = GetMutablePmt(ptr->buffer_pointer()+4); // assuming size prefix is 32 bit
    auto fb_vec = ((pmtf::VectorComplex64 *)pmt->mutable_data())->mutable_value();
    if (k >= fb_vec->size())
        throw std::runtime_error("PMT Vector index out of range");
    fb_vec->Mutate(k,(Complex64*)&val);  // hacky cast
}

template <>
pmt_vector<std::complex<float>>::sptr pmt_vector_from_buffer<std::complex<float>>(const uint8_t *buf)
{
    auto data = GetPmt(buf)->data_as_VectorComplex64()->value();
    size_t size = GetPmt(buf)->data_as_VectorComplex64()->value()->size();
    return std::make_shared<pmt_vector<std::complex<float>>>((std::complex<float>*)data->Data(), size);
}

#endif


template class pmt_vector<std::int32_t>;

} // namespace pmtf
