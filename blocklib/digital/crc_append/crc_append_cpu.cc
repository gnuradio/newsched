/* -*- c++ -*- */
/*
 * Copyright 2022 FIXME
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "crc_append_cpu.h"
#include "crc_append_cpu_gen.h"

namespace gr {
namespace digital {

crc_append_cpu::crc_append_cpu(block_args args)
    : INHERITED_CONSTRUCTORS,
      d_num_bits(args.num_bits),
      d_swap_endianness(args.swap_endianness),
      d_crc(kernel::digital::crc(args.num_bits,
                                 args.poly,
                                 args.initial_value,
                                 args.final_xor,
                                 args.input_reflected,
                                 args.result_reflected)),
      d_header_bytes(args.skip_header_bytes)
{
    if (args.num_bits % 8 != 0) {
        throw std::runtime_error("CRC number of bits must be divisible by 8");
    }
}


} // namespace digital
} // namespace gr