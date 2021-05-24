#include <pmt/pmtf_vector.hh>

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

IMPLEMENT_PMT_VECTOR_CPLX(std::complex<float>, Complex64)
// IMPLEMENT_PMT_VECTOR_CPLX(std::complex<double>, Complex128)

} // namespace pmtf
