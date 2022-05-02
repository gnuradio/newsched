/* -*- c++ -*- */
/*
 * Copyright 2022 FIXME
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/digital/crc_append.h>

#include <pmtf/map.hpp>
#include <pmtf/vector.hpp>

#include <gnuradio/kernel/digital/crc.h>

namespace gr {
namespace digital {

class crc_append_cpu : public virtual crc_append
{
public:
    crc_append_cpu(block_args args);

private:
    unsigned d_num_bits;
    bool d_swap_endianness;
    kernel::digital::crc d_crc;
    unsigned d_header_bytes;

    virtual void handle_msg_in(pmtf::pmt msg) override
    {
        auto meta = pmtf::get_as<std::map<std::string, pmtf::pmt>>(
            pmtf::map(msg)["meta"]);
        auto samples = pmtf::get_as<std::vector<uint8_t>>(
            pmtf::map(msg)["data"]);

        const auto size = samples.size();
        if (size <= d_header_bytes) {
            d_logger->warn("PDU too short; dropping");
            return;
        }

        uint64_t crc = d_crc.compute(&samples[d_header_bytes], size - d_header_bytes);

        unsigned num_bytes = d_num_bits / 8;
        if (d_swap_endianness) {
            for (unsigned i = 0; i < num_bytes; ++i) {
                samples.push_back(crc & 0xff);
                crc >>= 8;
            }
        }
        else {
            for (unsigned i = 0; i < num_bytes; ++i) {
                samples.push_back((crc >> (d_num_bits - 8 * (i + 1))) & 0xff);
            }
        }

        meta["packet_len"] = pmtf::get_as<size_t>(meta["packet_len"]) + num_bytes;
        auto pdu = pmtf::map({ { "data", samples }, { "meta", meta } });

        this->get_message_port("out")->post(pdu);
    }
};

} // namespace digital
} // namespace gr