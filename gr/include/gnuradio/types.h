#pragma once

#include <pmtf/base.hpp>
#include <stddef.h> // size_t
#include <stdint.h>
#include <complex>
#include <typeindex>
#include <vector>

using gr_vector_int = std::vector<int>;
using gr_vector_uint = std::vector<unsigned int>;
using gr_vector_float = std::vector<float>;
using gr_vector_double = std::vector<double>;
using gr_vector_void_star = std::vector<void*>;
using gr_vector_const_void_star = std::vector<const void*>;
using gr_complex = std::complex<float>;
using gr_complexd = std::complex<double>;

using pmt_sptr = std::shared_ptr<pmtf::pmt>;
