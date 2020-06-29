/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_TYPES_HPP
#define INCLUDED_GR_TYPES_HPP

// #include <gnuradio/api.h>
#include <stddef.h> // size_t
#include <stdint.h>
#include <complex>
#include <typeindex>
#include <vector>

typedef std::vector<int> gr_vector_int;
typedef std::vector<unsigned int> gr_vector_uint;
typedef std::vector<float> gr_vector_float;
typedef std::vector<double> gr_vector_double;
typedef std::vector<void*> gr_vector_void_star;
typedef std::vector<const void*> gr_vector_const_void_star;
typedef std::complex<float> gr_complex;
typedef std::complex<double> gr_complexd;

#endif /* INCLUDED_GR_TYPES_HPP */
