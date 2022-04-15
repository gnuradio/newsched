/* -*- c++ -*- */
/*
 * Copyright 2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/kernel/api.h>
#include <gnuradio/kernel/filter/fir_filter.h>

namespace gr {
namespace kernel {
namespace filter {


/*!
 * \brief Polyphase filterbank arbitrary resampler with
 *        gr_complex input, gr_complex output and float taps
 * \ingroup resamplers_blk
 *
 * \details
 * This  takes in a signal stream and performs arbitrary
 * resampling. The resampling rate can be any real number
 * <EM>r</EM>. The resampling is done by constructing <EM>N</EM>
 * filters where <EM>N</EM> is the interpolation rate.  We then
 * calculate <EM>D</EM> where <EM>D = floor(N/r)</EM>.
 *
 * Using <EM>N</EM> and <EM>D</EM>, we can perform rational
 * resampling where <EM>N/D</EM> is a rational number close to
 * the input rate <EM>r</EM> where we have <EM>N</EM> filters
 * and we cycle through them as a polyphase filterbank with a
 * stride of <EM>D</EM> so that <EM>i+1 = (i + D) % N</EM>.
 *
 * To get the arbitrary rate, we want to interpolate between two
 * points. For each value out, we take an output from the
 * current filter, <EM>i</EM>, and the next filter <EM>i+1</EM>
 * and then linearly interpolate between the two based on the
 * real resampling rate we want.
 *
 * The linear interpolation only provides us with an
 * approximation to the real sampling rate specified. The error
 * is a quantization error between the two filters we used as
 * our interpolation points.  To this end, the number of
 * filters, <EM>N</EM>, used determines the quantization error;
 * the larger <EM>N</EM>, the smaller the noise. You can design
 * for a specified noise floor by setting the filter size
 * (parameters <EM>filter_size</EM>). The size defaults to 32
 * filters, which is about as good as most implementations need.
 *
 * The trick with designing this filter is in how to specify the
 * taps of the prototype filter. Like the PFB interpolator, the
 * taps are specified using the interpolated filter rate. In
 * this case, that rate is the input sample rate multiplied by
 * the number of filters in the filterbank, which is also the
 * interpolation rate. All other values should be relative to
 * this rate.
 *
 * For example, for a 32-filter arbitrary resampler and using
 * the GNU Radio's firdes utility to build the filter, we build
 * a low-pass filter with a sampling rate of <EM>fs</EM>, a 3-dB
 * bandwidth of <EM>BW</EM> and a transition bandwidth of
 * <EM>TB</EM>. We can also specify the out-of-band attenuation
 * to use, <EM>ATT</EM>, and the filter window function (a
 * Blackman-harris window in this case). The first input is the
 * gain of the filter, which we specify here as the
 * interpolation rate (<EM>32</EM>).
 *
 *   <B><EM>self._taps = filter.firdes.low_pass_2(32, 32*fs, BW, TB,
 *      attenuation_dB=ATT, window=fft.window.WIN_BLACKMAN_hARRIS)</EM></B>
 *
 * The theory behind this block can be found in Chapter 7.5 of
 * the following book:
 *
 *   <B><EM>f. harris, "Multirate Signal Processing for Communication
 *      Systems", Upper Saddle River, NJ: Prentice Hall, Inc. 2004.</EM></B>
 */

template <class T_IN, class T_OUT, class TAP_T>
class pfb_arb_resampler
{
private:
    std::vector<fir_filter<T_IN, T_OUT, TAP_T>> d_filters;
    std::vector<fir_filter<T_IN, T_OUT, TAP_T>> d_diff_filters;
    std::vector<std::vector<TAP_T>> d_taps;
    std::vector<std::vector<TAP_T>> d_dtaps;
    size_t d_int_rate;        // the number of filters (interpolation rate)
    size_t d_dec_rate;        // the stride through the filters (decimation rate)
    float d_flt_rate;               // residual rate for the linear interpolation
    float d_acc;                    // accumulator; holds fractional part of sample
    size_t d_last_filter;     // stores filter for re-entry
    size_t d_taps_per_filter; // num taps for each arm of the filterbank
    int d_delay;                    // filter's group delay
    float d_est_phase_change;       // est. of phase change of a sine wave through filt.

    /*!
     * Takes in the taps and convolves them with [-1,0,1], which
     * creates a differential set of taps that are used in the
     * difference filterbank.
     * \param newtaps (vector of floats) The prototype filter.
     * \param difftaps (vector of floats) (out) The differential filter taps.
     */
    void create_diff_taps(const std::vector<TAP_T>& newtaps,
                          std::vector<TAP_T>& difftaps);

    /*!
     * Resets the filterbank's filter taps with the new prototype filter
     * \param newtaps    (vector of floats) The prototype filter to populate the
     * filterbank. The taps should be generated at the interpolated sampling rate. \param
     * ourtaps    (vector of floats) Reference to our internal member of holding the taps.
     * \param ourfilter  (vector of filters) Reference to our internal filter to set the
     * taps for.
     */
    void
    create_taps(const std::vector<TAP_T>& newtaps,
                std::vector<std::vector<TAP_T>>& ourtaps,
                std::vector<kernel::filter::fir_filter<T_IN, T_OUT, TAP_T>>& ourfilter);

public:
    /*!
     * Creates a kernel to perform arbitrary resampling on a set of samples.
     * \param rate  (float) Specifies the resampling rate to use
     * \param taps  (vector/list of floats) The prototype filter to populate the
     * filterbank. The taps       *              should be generated at the filter_size
     * sampling rate. \param filter_size (size_t) The number of filters in the
     * filter bank. This is directly related to quantization noise introduced during the
     * resampling. Defaults to 32 filters.
     */
    pfb_arb_resampler(float rate,
                      const std::vector<TAP_T>& taps,
                      size_t filter_size);

    // Don't allow copy.
    pfb_arb_resampler(const pfb_arb_resampler&) = delete;
    pfb_arb_resampler& operator=(const pfb_arb_resampler&) = delete;

    /*!
     * Resets the filterbank's filter taps with the new prototype filter
     * \param taps (vector/list of floats) The prototype filter to populate the
     * filterbank.
     */
    void set_taps(const std::vector<TAP_T>& taps);

    /*!
     * Return a vector<vector<>> of the filterbank taps
     */
    std::vector<std::vector<TAP_T>> taps() const;

    /*!
     * Print all of the filterbank taps to screen.
     */
    void print_taps();

    /*!
     * Sets the resampling rate of the block.
     */
    void set_rate(float rate);

    /*!
     * Sets the current phase offset in radians (0 to 2pi).
     */
    void set_phase(float ph);

    /*!
     * Gets the current phase of the resampler in radians (2 to 2pi).
     */
    float phase() const;

    /*!
     * Gets the number of taps per filter.
     */
    size_t taps_per_filter() const { return d_taps_per_filter; };

    size_t interpolation_rate() const { return d_int_rate; }
    size_t decimation_rate() const { return d_dec_rate; }
    float fractional_rate() const { return d_flt_rate; }

    /*!
     * Get the group delay of the filter.
     */
    int group_delay() const { return d_delay; }

    /*!
     * Calculates the phase offset expected by a sine wave of
     * frequency \p freq and sampling rate \p fs (assuming input
     * sine wave has 0 degree phase).
     */
    float phase_offset(float freq, float fs) const;

    /*!
     * Performs the filter operation that resamples the signal.
     *
     * This block takes in a stream of samples and outputs a
     * resampled and filtered stream. This block should be called
     * such that the output has \p rate * \p n_to_read amount of
     * space available in the \p output buffer.
     *
     * \param output The output samples at the new sample rate.
     * \param input An input vector of samples to be resampled
     * \param n_to_read Number of samples to read from \p input.
     * \param n_read (out) Number of samples actually read from \p input.
     * \return Number of samples put into \p output.
     */
    int filter(T_OUT* output, const T_IN* input, int n_to_read, int& n_read);
};

} // namespace filter
} // namespace kernel
} /* namespace gr */
