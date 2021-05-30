/* -*- c++ -*- */
/*
 * Copyright 2003,2008,2012,2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

/*
 * Wrappers for cuFFT single precision 1d dft
 */

#include <gnuradio/fft/api.h>
#include <gnuradio/types.hh>
#include <gnuradio/logging.hh>

#include <mutex>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>


namespace gr {
namespace fft {

template <class T, bool forward>
struct cufft_inbuf {
    typedef T type;
};

template <>
struct cufft_inbuf<float, false> {
    typedef gr_complex type;
};

template <class T, bool forward>
struct cufft_outbuf {
    typedef T type;
};

template <>
struct cufft_outbuf<float, true> {
    typedef gr_complex type;
};

template <class T, bool forward>
class FFT_API cufft
{
private:

    cufftHandle plan;

    gr::logger_sptr d_logger;
    gr::logger_sptr d_debug_logger;

    // void initialize_plan(int fft_size);
    size_t d_fft_size;
    size_t d_batch_size;

public:
    cufft(size_t fft_size, size_t batch_size);
    
    cufft(const cufft&) = delete;
    cufft& operator=(const cufft&) = delete;
    virtual ~cufft();

    /*!
     * compute FFT. 
     */
    void execute(typename cufft_inbuf<T, forward>::type* in, typename cufft_outbuf<T, forward>::type* out);
};

using cufft_complex_fwd = cufft<gr_complex, true>;
// using cufft_complex_rev = cufft<gr_complex, false>;
// using cufft_real_fwd = cufft<float, true>;
// using cufft_real_rev = cufft<float, false>;

} /* namespace fft */
} /*namespace gr */
