#include <pmt/pmtf.hpp>
#include <pmt/pmtf_scalar.hpp>
#include <pmt/pmtf_string.hpp>
#include <pmt/pmtf_vector.hpp>
#include <map>

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


///////////////////////////////////////////////////////////////////////////////////
// Static PMT Base functions
///////////////////////////////////////////////////////////////////////////////////

pmt_base::sptr pmt_base::from_pmt(const pmtf::Pmt* fb_pmt)
{
    switch (fb_pmt->data_type()) {
    case Data::PmtString:
        return std::static_pointer_cast<pmt_base>(pmt_string::from_pmt(fb_pmt));
    case Data::ScalarComplex64:
        return std::static_pointer_cast<pmt_base>(
            pmt_scalar_value<std::complex<float>>::from_pmt(fb_pmt));
    case Data::ScalarComplex128:
        return std::static_pointer_cast<pmt_base>(
            pmt_scalar_value<std::complex<double>>::from_pmt(fb_pmt));
    case Data::ScalarInt8:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<int8_t>::from_pmt(fb_pmt));
    case Data::ScalarUInt8:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<uint8_t>::from_pmt(fb_pmt));
    case Data::ScalarInt16:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<int16_t>::from_pmt(fb_pmt));
    case Data::ScalarUInt16:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<uint16_t>::from_pmt(fb_pmt));
    case Data::ScalarInt32:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<int32_t>::from_pmt(fb_pmt));
    case Data::ScalarUInt32:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<uint32_t>::from_pmt(fb_pmt));
    case Data::ScalarInt64:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<int64_t>::from_pmt(fb_pmt));
    case Data::ScalarUInt64:
        return std::static_pointer_cast<pmt_base>(pmt_scalar_value<uint64_t>::from_pmt(fb_pmt));
    case Data::VectorInt8:
        return std::static_pointer_cast<pmt_base>(pmt_vector<int8_t>::from_pmt(fb_pmt));
    case Data::VectorUInt8:
        return std::static_pointer_cast<pmt_base>(pmt_vector<uint8_t>::from_pmt(fb_pmt));
    case Data::VectorInt16:
        return std::static_pointer_cast<pmt_base>(pmt_vector<int16_t>::from_pmt(fb_pmt));
    case Data::VectorUInt16:
        return std::static_pointer_cast<pmt_base>(pmt_vector<uint16_t>::from_pmt(fb_pmt));
    case Data::VectorInt32:
        return std::static_pointer_cast<pmt_base>(pmt_vector<int32_t>::from_pmt(fb_pmt));
    case Data::VectorUInt32:
        return std::static_pointer_cast<pmt_base>(pmt_vector<uint32_t>::from_pmt(fb_pmt));
    case Data::VectorInt64:
        return std::static_pointer_cast<pmt_base>(pmt_vector<int64_t>::from_pmt(fb_pmt));
    case Data::VectorUInt64:
        return std::static_pointer_cast<pmt_base>(pmt_vector<uint64_t>::from_pmt(fb_pmt));
    case Data::VectorFloat32:
        return std::static_pointer_cast<pmt_base>(pmt_vector<float>::from_pmt(fb_pmt));
    case Data::VectorFloat64:
        return std::static_pointer_cast<pmt_base>(pmt_vector<double>::from_pmt(fb_pmt));
    case Data::VectorComplex64:
        return std::static_pointer_cast<pmt_base>(pmt_vector<std::complex<float>>::from_pmt(fb_pmt));
    // case Data::VectorComplex128:
    //     return std::static_pointer_cast<pmt_base>(pmt_vector<std::complex<double>>::from_pmt(fb_pmt));

    default:
        throw std::runtime_error("Unsupported PMT Type");
    }
}

pmt_base::sptr pmt_base::from_buffer(const uint8_t* buf, size_t size)
{
    auto PMT = GetPmt(buf);
    return from_pmt(PMT);
}

template class pmt_scalar_value<std::complex<float>>;
template class pmt_vector<std::int32_t>;
// template class pmt_vector<std::complex<float>>;

} // namespace pmtf
