#pragma once

#include <pmt/pmt_generated.h>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <pmt/pmtf_wrap.hpp>

namespace pmtf {

template <class T>
class pmt_map2 : public pmt_base
{
public:
    using key_type = std::string;
    using mapped_type = pmt_wrap;
    using value_type = std::pair<const key_type, mapped_type>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using map_type = std::map<T, pmt_wrap>;

    typedef std::shared_ptr<pmt_map2> sptr;
    static sptr make(const map_type& val)
    {
        return std::make_shared<pmt_map2<T>>(val);
    }
    static sptr from_buffer(const uint8_t* buf, size_t size)
    {
        return std::make_shared<pmt_map2<T>>(buf, size);
    }
    static sptr from_pmt(const pmtf::Pmt* fb_pmt)
    {
        return std::make_shared<pmt_map2<T>>(fb_pmt);
    }

    /**
     * @brief Construct a new pmt map object that is empty
     *
     * @param val
     */
    pmt_map2();

    /**
     * @brief Construct a new pmt map object from a std::map
     *
     * @param val
     */
    pmt_map2(const map_type& val);
    /**
     * @brief Construct a new pmt map object from a serialized flatbuffer
     *
     * @param buf
     */
    pmt_map2(const uint8_t* buf, size_t size);
    /**
     * @brief Construct a new pmt map object from a flatbuffers interpreted Pmt object
     *
     * @param fb_pmt
     */
    pmt_map2(const pmtf::Pmt* fb_pmt);


    /*void operator=(const std::map<T, pmt_sptr>& other) // copy assignment
    {
        set_value(other);
    }*/

    typename map_type::iterator begin() noexcept { return _map.begin(); }
    //typename std::map<T, pmt_sptr>::const_iterator begin() const noexcept { return _map.begin(); }
    typename map_type::iterator end() noexcept { return _map.end(); }
    //typename const std::map<T, pmt_sptr>::iterator end() const noexcept { return _map.end(); }

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

    mapped_type& operator[](const key_type& key);
    


private:
    // This stores the actual data.
    map_type _map;

    void fill_flatbuffer();
    virtual void serialize_setup();


};

}
