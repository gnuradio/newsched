#include <pmt/pmtf_vector.hpp>

namespace pmtf {


IMPLEMENT_PMT_VECTOR(int8_t, Int8)
IMPLEMENT_PMT_VECTOR(uint8_t, UInt8)
IMPLEMENT_PMT_VECTOR(int16_t, Int16)
IMPLEMENT_PMT_VECTOR(uint16_t, UInt16)
IMPLEMENT_PMT_VECTOR(int32_t, Int32)
IMPLEMENT_PMT_VECTOR(uint32_t, UInt32)
IMPLEMENT_PMT_VECTOR(int64_t, Int64)
IMPLEMENT_PMT_VECTOR(uint64_t, UInt64)
// IMPLEMENT_PMT_VECTOR(bool, Bool)
IMPLEMENT_PMT_VECTOR(float, Float32)
IMPLEMENT_PMT_VECTOR(double, Float64)

// IMPLEMENT_PMT_VECTOR_CPLX(std::complex<float>, Complex64)
// IMPLEMENT_PMT_VECTOR_CPLX(std::complex<double>, Complex128)


#if 1
//////////////////////////////////////////////////////////////////////////////////
// std::complex<float> vector stuff
//////////////////////////////////////////////////////////////////////////////////

template <>
flatbuffers::Offset<void>
pmt_vector<std::complex<float>>::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    /* fbb.Reset(); */
    auto pmt = GetSizePrefixedPmt(buffer_pointer());
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    auto vec = fbb.CreateVector(fb_vec->data(), fb_vec->size());
    VectorComplex64Builder vb(fbb);
    vb.add_value(vec);
    return vb.Finish().Union();
}

template <>
void pmt_vector<std::complex<float>>::set_value(
    const std::vector<std::complex<float>>& val)
{
    _fbb.Reset();
    auto vec = _fbb.CreateVectorOfNativeStructs<Complex64, std::complex<float>>(
        val.data(), val.size());
    VectorComplex64Builder vb(_fbb);
    vb.add_value(vec);
    _data = vb.Finish().Union();
    build();
}

template <>
void pmt_vector<std::complex<float>>::set_value(const std::complex<float>* data,
                                                size_t len)
{
    _fbb.Reset();
    auto vec =
        _fbb.CreateVectorOfNativeStructs<Complex64, std::complex<float>>(data, len);
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
pmt_vector<std::complex<float>>::pmt_vector(const uint8_t* buf)
    : pmt_base(Data::VectorComplex64)
{
    auto data = GetPmt(buf)->data_as_VectorComplex64()->value();
    size_t len = data->size();
    set_value((const std::complex<float>*)data->Data(), len);
}

template <>
pmt_vector<std::complex<float>>::pmt_vector(const pmtf::Pmt* fb_pmt)
    : pmt_base(Data::VectorComplex64)
{
    auto data = fb_pmt->data_as_VectorComplex64()->value();
    size_t len = data->size();
    set_value((const std::complex<float>*)data->Data(), len);
}


template <>
std::vector<std::complex<float>> pmt_vector<std::complex<float>>::value() const
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    std::vector<std::complex<float>> ret(fb_vec->size());
    /* because flatbuffers returns ptr to std::complex */
    for (int i = 0; i < fb_vec->size(); i++) {
        ret[i] = *(std::complex<float>*)fb_vec->Get(i);
    }
    return ret;
}

template <>
const std::complex<float>* pmt_vector<std::complex<float>>::data()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    return (std::complex<float>*)
        fb_vec->Data(); // no good native conversions in API, just cast here
}

template <>
size_t pmt_vector<std::complex<float>>::size()
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    return fb_vec->size();
}

template <>
std::complex<float> pmt_vector<std::complex<float>>::ref(size_t k)
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto fb_vec = pmt->data_as_VectorComplex64()->value();
    if (k >= fb_vec->size())
        throw std::runtime_error("PMT Vector index out of range");
    return *((std::complex<float>*)(*fb_vec)[k]); // hacky cast
}

// template <>
// const std::complex<float>* pmt_vector<std::complex<float>>::elements(pmt_base::sptr
// ptr)
// {
//     if (ptr->data_type() != Data::VectorComplex64)
//         throw std::runtime_error("PMT not of type std::complex<float>");
//     auto pmt = GetSizePrefixedPmt(ptr->buffer_pointer());
//     auto fb_vec = pmt->data_as_VectorComplex64()->value();
//     return (std::complex<float>*)(fb_vec->Data());  // hacky cast
// }

template <>
void pmt_vector<std::complex<float>>::set(size_t k, std::complex<float> val)
{
    auto pmt = GetMutablePmt(buffer_pointer() + 4); // assuming size prefix is 32 bit
    auto fb_vec = ((pmtf::VectorComplex64*)pmt->mutable_data())->mutable_value();
    if (k >= fb_vec->size())
        throw std::runtime_error("PMT Vector index out of range");
    fb_vec->Mutate(k, (Complex64*)&val); // hacky cast
}

// template <>
// pmt_vector<std::complex<float>>::sptr pmt_vector_from_buffer<std::complex<float>>(const
// uint8_t *buf)
// {
//     auto data = GetPmt(buf)->data_as_VectorComplex64()->value();
//     size_t size = GetPmt(buf)->data_as_VectorComplex64()->value()->size();
//     return
//     std::make_shared<pmt_vector<std::complex<float>>>((std::complex<float>*)data->Data(),
//     size);
// }

#endif


} // namespace pmtf
