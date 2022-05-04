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

namespace pmtf {


/**
 * @brief PDU representation of PMT class
 *
 * Eventually this should go into the PMTF library, but gathering concepts here
 *
 */
class pdu
{
public:
    pdu() {}
    pdu(size_t size, size_t numels) : _data(size * numels) {}
    template <typename T>
    pdu(const std::vector<T>& vec)
        : _data((const uint8_t*)vec.data(), (const uint8_t*)(vec.data() + vec.size()))
    {
    }
    template <typename T>
    pdu(const pmtf::vector<T>& vec)
        : _data((const uint8_t*)vec.data(), (const uint8_t*)(vec.data() + vec.size()))
    {
    }
    template <typename T>
    pdu(std::initializer_list<T> il)
        : _data((uint8_t*)il.begin(), (uint8_t*)il.begin() + il.size() * sizeof(T))
    {
    }
    pdu(void* d, size_t size) : _data((uint8_t*)d, (uint8_t*)d + size) {}

    // From a Pmt Buffer
    template <class U, typename = pmtf::IsPmt<U>>
    pdu(const U& other)
    {
        // do better checking here
        auto pmtvec = pmtf::vector<pmtf::pmt>(other);
        _meta = pmtf::map(pmtvec[0]);
        _data = pmtf::vector<uint8_t>(pmtvec[1]);
    }

    size_t size_bytes() { return _data.size(); }
    void resize_bytes(size_t new_size)
    {
        if (new_size < _data.size()) {
            _data = pmtf::vector<uint8_t>(_data.data(), _data.data() + new_size);
        }
        else {
            pmtf::vector<uint8_t> tmp(new_size);
            std::copy(tmp.begin(), tmp.end(), _data.begin());
            _data = tmp;
        }
    }
    template <typename T>
    size_t size()
    {
        return _data.size() / sizeof(T);
    }
    template <typename T>
    T* data()
    {
        return (T*)_data.data();
    }
    uint8_t* raw() { return _data.data(); }
    pmtf::pmt& operator[](const std::string& key) { return _meta[key]; }

    // template <typename T>
    // T& operator[](size_t n)
    // {
    //     // operator[] doesn't do bounds checking, use at for that
    //     // TODO: implement at
    //     return data<T>()[n];
    // }
    template <typename T>
    T& at(size_t n)
    {
        return data<T>()[n];
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
