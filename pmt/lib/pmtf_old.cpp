#include <pmt/pmtf.hpp>
#include <map>

namespace pmtf {

IMPLEMENT_PMT_SCALAR(float,Float32)
IMPLEMENT_PMT_SCALAR(double,Float64)
// IMPLEMENT_PMT_SCALAR(std::complex<float>,Complex64)
// IMPLEMENT_PMT_SCALAR_CPLX(std::complex<double>,Complex128)
IMPLEMENT_PMT_SCALAR(int8_t,Int8)
IMPLEMENT_PMT_SCALAR(int16_t,Int16)
IMPLEMENT_PMT_SCALAR(int32_t,Int32)
IMPLEMENT_PMT_SCALAR(int64_t,Int64)
IMPLEMENT_PMT_SCALAR(uint8_t,UInt8)
IMPLEMENT_PMT_SCALAR(uint16_t,UInt16)
IMPLEMENT_PMT_SCALAR(uint32_t,UInt32)
IMPLEMENT_PMT_SCALAR(uint64_t,UInt64)

IMPLEMENT_PMT_VECTOR(float,Float32)
IMPLEMENT_PMT_VECTOR(double,Float64)
// IMPLEMENT_PMT_VECTOR_CPLX(std::complex<float>,Complex64)
// IMPLEMENT_PMT_SCALAR_CPLX(std::complex<double>,Complex128)
IMPLEMENT_PMT_VECTOR(int8_t,Int8)
IMPLEMENT_PMT_VECTOR(int16_t,Int16)
IMPLEMENT_PMT_VECTOR(int32_t,Int32)
IMPLEMENT_PMT_VECTOR(int64_t,Int64)
IMPLEMENT_PMT_VECTOR(uint8_t,UInt8)
IMPLEMENT_PMT_VECTOR(uint16_t,UInt16)
IMPLEMENT_PMT_VECTOR(uint32_t,UInt32)
IMPLEMENT_PMT_VECTOR(uint64_t,UInt64)


// template <> 
// void pmt_vector<std::complex<float>>::set_value(const std::vector<std::complex<float>>& val) 
// { 
//     auto vec = _fbb.CreateVector((const Complex64 **)val.data(), val.size()); 
//     VectorComplex64Builder vb(_fbb); 
//     vb.add_value(vec); 
//     _data = vb.Finish().Union(); 
//     build(); 
// } 

// template <> 
// pmt_vector<std::complex<float>>::pmt_vector(const std::vector<std::complex<float>>& val) 
//     : pmt_base(Data_VectorComplex64) 
// { 
//     set_value(val); 
// } 

// template <> 
// std::vector<std::complex<float>>& pmt_vector<std::complex<float>>::value() 
// { 
//     auto pmt = GetPmt(_fbb.GetBufferPointer()); 
//     auto fb_vec = pmt->data_as_VectorComplex64()->value(); 
//     _value.assign((std::complex<float> *)fb_vec->begin(), (std::complex<float> *)fb_vec->end()); 
//     return _value; 
// } 

// template <> 
// const std::complex<float>* pmt_vector<std::complex<float>>::data() 
// { 
//     auto pmt = GetPmt(_fbb.GetBufferPointer()); 
//     auto fb_vec = pmt->data_as_VectorComplex64()->value(); 
//     return (std::complex<float> *)fb_vec->data(); 
// } 

// template <> 
// size_t pmt_vector<std::complex<float>>::size() 
// { 
//     auto pmt = GetPmt(_fbb.GetBufferPointer()); 
//     auto fb_vec = pmt->data_as_VectorComplex64()->value(); 
//     return fb_vec->size(); 
// }


} // namespace pmtf