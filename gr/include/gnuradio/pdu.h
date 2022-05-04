/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <pmtf/map.hpp>
#include <pmtf/vector.hpp>

namespace gr {


/**
 * @brief PDU representation of PMT class
 *
 * Eventually this should go into the PMTF library, but gathering concepts here
 *
 */
template <class T>
class pdu
{
public:
    pdu(size_t size) : _data(size * sizeof(T)) { set_data_type(); }
    // pdu(const std::vector<T>& vec)
    //     : _data((std::vector<uint8_t>::iterator)vec.begin(),
    //             (std::vector<uint8_t>::iterator)vec.begin() + vec.size() * sizeof(T))
    // {
    //     set_data_type();
    // }
    pdu(std::initializer_list<T> il)
        : _data((uint8_t*)il.begin(), (uint8_t*)il.begin() + il.size() * sizeof(T))
    {
        set_data_type();
    }
    pdu(T* d, size_t size) : _data((uint8_t*)d, (uint8_t*)d + size * sizeof(T))
    {
        set_data_type();
    }

    // From a Pmt Buffer
    template <class U, typename = pmtf::IsPmt<U>>
    pdu(const U& other)
    {
        // do better checking here
        auto pmtvec = pmtf::vector<pmtf::pmt>(other);
        _meta = pmtf::map(pmtvec[0]);
        _data = pmtf::vector<uint8_t>(pmtvec[1]);
    }

    size_t size();
    T* data() { return (T*)_data.data(); }
    pmtf::pmt& operator[](const std::string& key) { return _meta[key]; }
    T& operator[](size_t n)
    {
        // operator[] doesn't do bounds checking, use at for that
        // TODO: implement at
        return data()[n];
    }

    pmtf::pmt get_pmt_buffer() const
    {
        auto pmt_vec = pmtf::vector<pmtf::pmt>(2);
        pmt_vec[0] = pmtf::pmt(_meta);
        pmt_vec[1] = pmtf::pmt(_data);
        // _underlying_pmt = pmt_vec;
        return pmt_vec;
    }

private:
    pmtf::map _meta;
    // Try representing any vector as bytes internally
    pmtf::vector<uint8_t> _data;

    void set_data_type();

    pmtf::pmt _underlying_pmt;
};

} // namespace gr
