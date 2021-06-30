#include <pmt/pmtf_scalar.hpp>

namespace pmtf {


IMPLEMENT_PMT_SCALAR(int8_t, Int8)
IMPLEMENT_PMT_SCALAR(uint8_t, UInt8)
IMPLEMENT_PMT_SCALAR(int16_t, Int16)
IMPLEMENT_PMT_SCALAR(uint16_t, UInt16)
IMPLEMENT_PMT_SCALAR(int32_t, Int32)
IMPLEMENT_PMT_SCALAR(uint32_t, UInt32)
IMPLEMENT_PMT_SCALAR(int64_t, Int64)
IMPLEMENT_PMT_SCALAR(uint64_t, UInt64)
IMPLEMENT_PMT_SCALAR(bool, Bool)
IMPLEMENT_PMT_SCALAR(float, Float32)
IMPLEMENT_PMT_SCALAR(double, Float64)

IMPLEMENT_PMT_SCALAR_CPLX(std::complex<float>, Complex64)
IMPLEMENT_PMT_SCALAR_CPLX(std::complex<double>, Complex128)



} // namespace pmtf
