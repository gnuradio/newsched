#pragma once

#include <pmt/pmt_generated.h>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <pmt/pmtf.hpp>

namespace pmtf {


template <class T>
class pmt_map : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_map> sptr;
    static sptr make(const std::map<T, pmt_sptr>& val)
    {
        return std::make_shared<pmt_map<T>>(val);
    }
    static sptr from_buffer(const uint8_t* buf, size_t size)
    {
        return std::make_shared<pmt_map<T>>(buf, size);
    }
    static sptr from_pmt(const pmtf::Pmt* fb_pmt)
    {
        return std::make_shared<pmt_map<T>>(fb_pmt);
    }

    /**
     * @brief Construct a new pmt map object from a std::map
     *
     * @param val
     */
    pmt_map(const std::map<T, pmt_sptr>& val);
    /**
     * @brief Construct a new pmt map object from a serialized flatbuffer
     *
     * @param buf
     */
    pmt_map(const uint8_t* buf, size_t size);
    /**
     * @brief Construct a new pmt map object from a flatbuffers interpreted Pmt object
     *
     * @param fb_pmt
     */
    pmt_map(const pmtf::Pmt* fb_pmt);

    void set_value(const std::map<T, pmt_sptr>& val);


    /**
     * @brief return a copy of the data stored in the flatbuffer
     *
     * @return const std::map<T, pmt_sptr>
     */
    std::map<T, pmt_sptr> value() const;
    const pmt_sptr data();
    size_t size();

    void operator=(const std::map<T, pmt_sptr>& other) // copy assignment
    {
        set_value(other);
    }

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

    pmt_sptr ref(const T& key);
    void set(const T& key, pmt_sptr value);
};


} // namespace pmtf
