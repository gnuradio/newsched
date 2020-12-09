#include <pmt/pmtf_scalar.hpp>

namespace pmtf {

template <>
std::complex<float> pmt_scalar<std::complex<float>>::value()
{
    auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());
    return *((std::complex<float>*)(pmt->data_as_ScalarComplex64()->value()));
}

template <>
flatbuffers::Offset<void>
pmt_scalar<std::complex<float>>::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    ScalarComplex64Builder sb(fbb);
    auto val = value();
    sb.add_value((Complex64*)&val);
    return sb.Finish().Union();
}

template <>
void pmt_scalar<std::complex<float>>::set_value(std::complex<float> val)
{
    ScalarComplex64Builder sb(_fbb);
    sb.add_value((Complex64*)&val);
    _data = sb.Finish().Union();
    build();
}

template <>
pmt_scalar<std::complex<float>>::pmt_scalar(const std::complex<float>& val)
    : pmt_base(Data::ScalarComplex64)
{
    set_value(val);
}

template <>
pmt_scalar<std::complex<float>>::pmt_scalar(const uint8_t* buf)
    : pmt_base(Data::ScalarComplex64)
{
    auto data = GetPmt(buf)->data_as_ScalarComplex64()->value();
    set_value(*((const std::complex<float>*)data));
}

template <>
pmt_scalar<std::complex<float>>::pmt_scalar(const pmtf::Pmt* fb_pmt)
    : pmt_base(Data::ScalarComplex64)
{
    auto data = fb_pmt->data_as_ScalarComplex64()->value();
    set_value(*((const std::complex<float>*)data));
}


//////////////////////////////////////////////////////////////////////////////////
// int_32_t scalar stuff
//////////////////////////////////////////////////////////////////////////////////


template <>
int32_t pmt_scalar<int32_t>::value()
{
    auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());
    return pmt->data_as_ScalarInt32()->value();
}

template <>
flatbuffers::Offset<void>
pmt_scalar<int32_t>::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    ScalarInt32Builder sb(fbb);
    auto val = value();
    sb.add_value(val);
    return sb.Finish().Union();
}

template <>
void pmt_scalar<int32_t>::set_value(int32_t val)
{
    ScalarInt32Builder sb(_fbb);
    sb.add_value(val);
    _data = sb.Finish().Union();
    build();
}

template <>
pmt_scalar<int32_t>::pmt_scalar(const int32_t& val) : pmt_base(Data::ScalarInt32)
{
    set_value(val);
}

template <>
pmt_scalar<int32_t>::pmt_scalar(const uint8_t* buf) : pmt_base(Data::ScalarInt32)
{
    auto data = GetPmt(buf)->data_as_ScalarInt32()->value();
    set_value(data);
}

template <>
pmt_scalar<int32_t>::pmt_scalar(const pmtf::Pmt* fb_pmt) : pmt_base(Data::ScalarInt32)
{
    auto data = fb_pmt->data_as_ScalarInt32()->value();
    set_value(data);
}


template class pmt_scalar<std::complex<float>>;

} // namespace pmtf
