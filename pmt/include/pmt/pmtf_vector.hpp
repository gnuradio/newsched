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
class pmt_vector : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_vector> sptr;
    static sptr make(const std::vector<T>& val)
    {
        return std::make_shared<pmt_vector<T>>(val);
    }
    static sptr make(const T* data, size_t len)
    {
        return std::make_shared<pmt_vector<T>>(data, len);
    }
    static sptr from_buffer(const uint8_t* buf) { return std::make_shared<pmt_vector<T>>(buf); }
    static sptr from_pmt(const pmtf::Pmt *fb_pmt) { return std::make_shared<pmt_vector<T>>(fb_pmt); }
    

    /**
     * @brief Construct a new pmt vector object from a std::vector
     *
     * @param val
     */
    pmt_vector(const std::vector<T>& val);
    /**
     * @brief Construct a new pmt vector object from an array
     *
     * @param data
     * @param len
     */
    pmt_vector(const T* data, size_t len);
    /**
     * @brief Construct a new pmt vector object from a serialized flatbuffer
     *
     * @param buf
     */
    pmt_vector(const uint8_t* buf);
    pmt_vector(const pmtf::Pmt *fb_pmt);

    void set_value(const std::vector<T>& val);
    void set_value(const T* data, size_t len);
    // void deserialize(std::streambuf& sb) override;
    std::vector<T> value() const; // returns a copy of the data stored in the flatbuffer
    const T* data();
    size_t size();

    void operator=(const std::vector<T>& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const std::vector<T>& other) { return other == value(); }
    bool operator!=(const std::vector<T>& other) { return other != value(); }

    T ref(size_t k); // overload operator []
    void set(size_t k, T val); // overload [] =
    T* writable_elements();
    const T* elements();

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

};


typedef std::function<std::shared_ptr<pmt_base>(uint8_t*)> pmt_from_buffer_function;

} // namespace pmtf
