#pragma once

#include <pmt/pmt_generated.h>
#include <pmt/pmtf.hpp>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>


namespace pmtf {

class pmt_string : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_string> sptr;
    static sptr make(const std::string& value)
    {
        return std::make_shared<pmt_string>(value);
    }
    static sptr from_buffer(const uint8_t* buf)
    {
        return std::make_shared<pmt_string>(buf);
    }
    static sptr from_pmt(const pmtf::Pmt *fb_pmt)
    {
        return std::make_shared<pmt_string>(fb_pmt);
    }


    void set_value(const std::string& val);
    std::string value();

    void operator=(const std::string& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const std::string& other) { return other == value(); }
    bool operator!=(const std::string& other) { return other != value(); }

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

    pmt_string(const std::string& val);
    pmt_string(const uint8_t* buf);
    pmt_string(const pmtf::Pmt *fb_pmt);
    void print(std::ostream& os) { os << value(); }
};


} // namespace pmtf
