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
#include <pmt/pmtf_wrap.hpp>

// What if this was just a map?
// Then I wouldn't have a serialize function in it. and it wouldn't be derived from pmt_base.
/*
What should the class hierarchy be???
Presently we have a pmt_base and then several classes dervice from that.
I then define wrappers around pointers to those classes that make it 
easy to work with them.
Can I just cut out the middle man and have the wrapper class be the main class?
Then we don't need all of the static make functions.  It handles all of that for
us.  Can I do this in a useful way?
So I have a pmt base class and derive from that pmt_scalar, uniform vector, pmt_vector, and pmt_map.
In the scalar case and uniform vector case I can just store it.  The pmt_vector would need to store
variants or pointers.
1) pmt is pointer, classes are wrappers to make it convenient.
    Need one class and one wrapper for each type
2) pmt is class with data.  Polymorphism doesn't buy me anything here, because I am avoiding creating
    pointers.  I have to use variants.

Let's start with polymorphism.
I need the following set of classes.
pmt_scalar
    uniform_vector
    vector
    map
I need a wrapper class for each one.
I need a generator class that can produce any one of them.

*/

namespace pmtf {

template <class T>
class pmt_map_value : public pmt_base
{
public:
    using key_type = std::string;
    using mapped_type = pmt_wrap;
    using value_type = std::pair<const key_type, mapped_type>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using map_type = std::map<T, pmt_wrap>;

    typedef std::shared_ptr<pmt_map_value> sptr;
    static sptr make(const map_type& val)
    {
        return std::make_shared<pmt_map_value<T>>(val);
    }
    static sptr from_buffer(const uint8_t* buf, size_t size)
    {
        return std::make_shared<pmt_map_value<T>>(buf, size);
    }
    static sptr from_pmt(const pmtf::Pmt* fb_pmt)
    {
        return std::make_shared<pmt_map_value<T>>(fb_pmt);
    }

    /**************************************************************************
    * Constructors
    **************************************************************************/

    /**
     * @brief Construct a new pmt map object that is empty
     *
     * @param
     */
    pmt_map_value();

    /**
     * @brief Construct a new pmt map object from a std::map
     *
     * @param val
     */
    pmt_map_value(const map_type& val);
    /**
     * @brief Construct a new pmt map object from a pmt_map_value
     *
     * @param val
     */
    pmt_map_value(const pmt_map_value& val);
    /**
     * @brief Construct a new pmt map object from a serialized flatbuffer
     *
     * @param buf
     * @param size
     */
    pmt_map_value(const uint8_t* buf, size_t size);
    /**
     * @brief Construct a new pmt map object from a flatbuffers interpreted Pmt object
     *
     * @param fb_pmt
     */
    pmt_map_value(const pmtf::Pmt* fb_pmt);

    /**************************************************************************
    * Copy Assignment
    **************************************************************************/
    pmt_map_value& operator=(const pmt_map_value& other);
    pmt_map_value& operator=(pmt_map_value&& other) noexcept;

    /**************************************************************************
    * Element Access
    **************************************************************************/
    mapped_type& at(const key_type& key) { return _map.at(key); }
    const mapped_type& at(const key_type& key ) const { return _map.at(key); }
    mapped_type& operator[]( const key_type& key);

    /**************************************************************************
    * Iterators
    **************************************************************************/
    typename map_type::iterator begin() noexcept { return _map.begin(); }
    typename map_type::const_iterator begin() const noexcept { return _map.begin(); }
    typename map_type::iterator end() noexcept { return _map.end(); }
    typename map_type::const_iterator end() const noexcept { return _map.end(); }

    /**************************************************************************
    * Capacity
    **************************************************************************/
    bool empty() const noexcept { return _map.empty(); }
    size_t size() const noexcept { return _map.size(); }
    size_t max_size() const noexcept { return _map.max_size(); }

    /**************************************************************************
    * Operations
    **************************************************************************/
    size_t count(const key_type& key) const { return _map.count(key); }


    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);

    void print(std::ostream& os) const {
        os << "{";
        for (const auto& [k, v]: *this) {
            os << k << ": " << v << ", "; 
        }
        os << "}";
    } 


private:
    // This stores the actual data.
    map_type _map;

    void fill_flatbuffer();
    virtual void serialize_setup();


};

template <class T>
class pmt_map {
public:
    using sptr = typename pmt_map_value<T>::sptr;
    using key_type = typename pmt_map_value<T>::key_type;
    using mapped_type = typename pmt_map_value<T>::mapped_type;
    using map_type = typename pmt_map_value<T>::map_type;
    
    pmt_map() : d_ptr(pmt_map_value<T>::make(map_type())) {}
    pmt_map(const map_type& x) : d_ptr(pmt_map_value<T>::make(x)) {}
    pmt_map(sptr p) : d_ptr(p) {}
    
    mapped_type& operator[]( const key_type& key) { return (*d_ptr)[key]; }
    typename map_type::iterator begin() noexcept { return d_ptr->begin(); }
    typename map_type::const_iterator begin() const noexcept { return d_ptr->begin(); }
    typename map_type::iterator end() noexcept { return d_ptr->end(); }
    typename map_type::const_iterator end() const noexcept { return d_ptr->end(); }

    size_t size() const noexcept { return d_ptr->size(); }
    size_t count(const key_type& key) const { return d_ptr->count(key); }
    const mapped_type& at(const key_type& key ) const { return d_ptr->at(key); }

    sptr ptr() { return d_ptr; }
private:
    sptr d_ptr;

};

template <class T>
pmt_map<T> get_pmt_map(const pmt_wrap& x) {
    if (x.ptr()->data_type() == Data::MapString) {
        return pmt_map<T>(std::dynamic_pointer_cast<pmt_map_value<T>>(x.ptr()));
    } else 
        throw std::runtime_error("Cannot convert to map type");
}

template <> pmt_wrap::pmt_wrap<std::map<std::string, pmt_wrap>>(const std::map<std::string, pmt_wrap>& x); 
template <> pmt_wrap::pmt_wrap<pmt_map<std::string>>(const pmt_map<std::string>& x);
template <> bool operator==<std::map<std::string, pmt_wrap>>(const pmt_wrap& x, const std::map<std::string, pmt_wrap>& y);
template <> bool operator==<pmt_map<std::string>>(const pmt_wrap& x, const pmt_map<std::string>& y);

template <class T>
bool is_pmt_map(const pmt_wrap& x) {
    return x.ptr()->data_type() == Data::MapString;
}

/*pmt_map<std::string> get_map(const pmt_wrap& x) {
    if (x.ptr()->data_type() == Data::PmtMap)
        return pmt_map<std::string>(std::dynamic_pointer_cast<pmt_map<std::string>>(x.ptr()));
    else
        throw std::runtime_error("Cannot convert to map");
}*/

}
