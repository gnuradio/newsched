#include <pmt/pmtf_vector.hpp>
#include <algorithm>

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

// I hate macros, but I'm going to use one here.
#define Apply(func) \
func(uint8_t) \
func(uint16_t) \
func(uint32_t) \
func(uint64_t) \
func(int8_t) \
func(int16_t) \
func(int32_t) \
func(int64_t) \
func(float) \
func(double) \
func(std::complex<float>)

#define VectorWrapImpl(T) template <> pmt_wrap::pmt_wrap<std::vector<T>>(const std::vector<T>& x) \
    { d_ptr = pmt_vector(x).ptr(); }
#define VectorWrapPmtImpl(T) template <> pmt_wrap::pmt_wrap<pmt_vector<T>>(const pmt_vector<T>& x) \
    {                                                                                        \
        pmt_vector<T> val(x.begin(), x.end());                                               \
        d_ptr = val.ptr();                                                                   \
    }
#define VectorEqualsImpl(T) \
    template <> bool operator==<std::vector<T>>(const pmt_wrap& x, const std::vector<T>& other) {   \
        if (is_pmt_vector<T>(x)) {                                                                  \
            auto xx = get_pmt_vector<T>(x);                                                         \
            if (xx.size() == other.size())                                                          \
                return std::equal(xx.begin(), xx.end(), other.begin());                              \
        }                                                                                           \
        return false;                                                                               \
    }
#define VectorEqualsPmtImpl(T) \
    template <> bool operator==<pmt_vector<T>>(const pmt_wrap& x, const pmt_vector<T>& other) {     \
        if (is_pmt_vector<T>(x)) {                                                                  \
            auto xx = get_pmt_vector<T>(x);                                                         \
            if (xx.size() == other.size())                                                          \
                return std::equal(xx.begin(), xx.end(), other.begin());                              \
        }                                                                                           \
        return false;                                                                               \
    }


Apply(VectorWrapImpl)
Apply(VectorWrapPmtImpl)
Apply(VectorEqualsImpl)
Apply(VectorEqualsPmtImpl)

} // namespace pmtf
