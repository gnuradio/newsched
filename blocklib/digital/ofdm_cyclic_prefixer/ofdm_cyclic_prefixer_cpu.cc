/* -*- c++ -*- */
/*
 * Copyright 2013, 2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "ofdm_cyclic_prefixer_cpu.h"
#include "ofdm_cyclic_prefixer_cpu_gen.h"

namespace gr {
namespace digital {

ofdm_cyclic_prefixer_cpu::ofdm_cyclic_prefixer_cpu(block_args args)
    : INHERITED_CONSTRUCTORS,
      d_fft_len(args.fft_len),
      d_rolloff_len(args.rolloff_len),
      d_cp_lengths(args.cp_lengths),
      d_up_flank((args.rolloff_len ? args.rolloff_len - 1 : 0), 0),
      d_down_flank((args.rolloff_len ? args.rolloff_len - 1 : 0), 0),
      d_delay_line(0, 0)
{

    // Sanity
    if (d_cp_lengths.empty()) {
        throw std::invalid_argument(this->alias() +
                                    std::string(": CP lengths vector can not be empty."));
    }
    for (size_t i = 0; i < d_cp_lengths.size(); i++) {
        if (d_cp_lengths[i] != 0) {
            break;
        }
        if (i == d_cp_lengths.size() - 1) {
            throw std::invalid_argument(
                this->alias() +
                std::string(": Please provide at least one CP which is != 0."));
        }
    }
    for (const size_t cp_length : d_cp_lengths) {
        d_cp_max = std::max(d_cp_max, cp_length);
        d_cp_min = std::min(d_cp_min, cp_length);
    }
    if (d_cp_min < 0) {
        throw std::invalid_argument(this->alias() +
                                    std::string(": The minimum CP allowed is 0."));
    }
    // Give the buffer allocator and scheduler a hint about the ratio between input and
    // output.
    set_relative_rate(d_cp_max + d_fft_len);

    // Flank of length 1 would just be rectangular.
    if (d_rolloff_len == 1) {
        d_rolloff_len = 0;
        d_logger->warn("Set rolloff to 0, because 1 would result in a boxcar function.");
    }
    if (d_rolloff_len) {
        d_delay_line.resize(d_rolloff_len - 1, 0);
        // More sanity
        if (d_rolloff_len > d_cp_min) {
            throw std::invalid_argument(
                this->alias() + std::string(": Rolloff length must be smaller than any "
                                            "of the cyclic prefix lengths."));
        }
        /* The actual flanks are one sample shorter than d_rolloff_len, because the
           first sample of the up- and down flank is always zero and one, respectively.*/
        for (int i = 1; i < d_rolloff_len; i++) {
            d_up_flank[i - 1] = 0.5 * (1 + cos(M_PI * i / args.rolloff_len - M_PI));
            d_down_flank[i - 1] =
                0.5 * (1 + cos(M_PI * (args.rolloff_len - i) / args.rolloff_len - M_PI));
        }
    }

    // noutput_items is set to be a multiple of the largest possible output size.
    // It is always OK to return less (in case of the shorter CP).
    set_output_multiple(d_fft_len + d_cp_max);
    set_tag_propagation_policy(tag_propagation_policy_t::TPP_DONT);
}

work_return_code_t
ofdm_cyclic_prefixer_cpu::work(std::vector<block_work_input_sptr>& work_input,
                               std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<gr_complex>();
    auto out = work_output[0]->items<gr_complex>();

    auto noutput_items = work_output[0]->n_items;
    auto ninput_items = work_input[0]->n_items;

    size_t nin = 0;
    size_t nout = 0;
    bool output_limited = false;
    bool input_limited = false;
    size_t output_size = 0;
    while (true) {
        output_size = d_fft_len + d_cp_lengths[d_state];

        if (nout + output_size > noutput_items) {
            output_limited = true;
        }
        if (nin + 1 > ninput_items) {
            input_limited = true;
        }
        if (input_limited || output_limited) {
            break;
        }

        std::copy(in, in + d_fft_len, out + d_cp_lengths[d_state]);
        std::copy(in + d_fft_len - d_cp_lengths[d_state], in + d_fft_len, out);

        if (d_rolloff_len) {
            for (int i = 0; i < d_rolloff_len - 1; i++) {
                out[i] = out[i] * d_up_flank[i] + d_delay_line[i];
                /* This is basically a cyclic suffix, but completely shifted into the next
                   symbol. The data rate does not change. */
                d_delay_line[i] = in[i] * d_down_flank[i];
            }
        }

        ++d_state;
        d_state %= d_cp_lengths.size();

        nout += output_size;
        nin += 1;

        in += d_fft_len;
        out += output_size;
    }

    if (nout == 0 && output_limited) {
        // based on what we were given as input, we would need
        // at least output_size
        // work_output[0]->n_requested = output_size;
        return work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS;
    }
    else if (nin == 0 && input_limited) {
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }


    // int symbols_to_read = 0;
    // symbols_to_read = std::min(noutput_items / (d_fft_len + d_cp_max), ninput_items);

    // noutput_items = 0;
    // // 2) Do the cyclic prefixing and, optionally, the pulse shaping.
    // for (int sym_idx = 0; sym_idx < symbols_to_read; sym_idx++) {
    //     memcpy(static_cast<void*>(out + d_cp_lengths[d_state]),
    //            static_cast<void*>(in),
    //            d_fft_len * sizeof(gr_complex));
    //     memcpy(static_cast<void*>(out),
    //            static_cast<void*>(in + d_fft_len - d_cp_lengths[d_state]),
    //            d_cp_lengths[d_state] * sizeof(gr_complex));
    //     if (d_rolloff_len) {
    //         for (int i = 0; i < d_rolloff_len - 1; i++) {
    //             out[i] = out[i] * d_up_flank[i] + d_delay_line[i];
    //             /* This is basically a cyclic suffix, but completely shifted into the
    //             next
    //                symbol. The data rate does not change. */
    //             d_delay_line[i] = in[i] * d_down_flank[i];
    //         }
    //     }
    //     in += d_fft_len;
    //     out += d_fft_len + d_cp_lengths[d_state];
    //     // Raise the number of noutput_items depending on how long the current output
    //     was. noutput_items += d_fft_len + d_cp_lengths[d_state];
    //     // Propagate tags.
    //     unsigned last_state = d_state > 0 ? d_state - 1 : d_cp_lengths.size() - 1;
    //     std::vector<tag_t> tags;
    //     get_tags_in_range(
    //         tags, 0, nitems_read(0) + sym_idx, nitems_read(0) + sym_idx + 1);
    //     for (unsigned i = 0; i < tags.size(); i++) {
    //         tags[i].offset = ((tags[i].offset - nitems_read(0)) *
    //                           (d_fft_len + d_cp_lengths[last_state])) +
    //                          nitems_written(0);
    //         add_item_tag(0, tags[i].offset, tags[i].key, tags[i].value);
    //     }
    //     // Finally switch to next state.
    //     ++d_state;
    //     d_state %= d_cp_lengths.size();
    // }
    // /* 3) If we're in packet mode:
    //       - flush the delay line, if applicable */
    // if (!d_len_tag_key.empty()) {
    //     if (d_rolloff_len) {
    //         std::memcpy(static_cast<void*>(out),
    //                     static_cast<void*>(d_delay_line.data()),
    //                     sizeof(gr_complex) * d_delay_line.size());
    //         d_delay_line.assign(d_delay_line.size(), 0);
    //         // Make last symbol a bit longer.
    //         noutput_items += d_delay_line.size();
    //     }
    // }
    // else {
    //     consume_each(symbols_to_read);
    // }

    consume_each(nin, work_input);
    produce_each(nout, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace digital
} // namespace gr