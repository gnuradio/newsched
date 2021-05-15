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
class pmt_vector_value : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_vector_value> sptr;
    static sptr make(const std::vector<T>& val)
    {
        return std::make_shared<pmt_vector_value<T>>(val);
    }
    static sptr make(const T* data, size_t len)
    {
        return std::make_shared<pmt_vector_value<T>>(data, len);
    }
    static sptr from_buffer(const uint8_t* buf)
    {
        return std::make_shared<pmt_vector_value<T>>(buf);
    }
    static sptr from_pmt(const pmtf::Pmt* fb_pmt)
    {
        return std::make_shared<pmt_vector_value<T>>(fb_pmt);
    }


    /**
     * @brief Construct a new pmt vector object from a std::vector
     *
     * @param val
     */
    pmt_vector_value(const std::vector<T>& val);
    /**
     * @brief Construct a new pmt vector object from an array
     *
     * @param data
     * @param len
     */
    pmt_vector_value(const T* data, size_t len);
    /**
     * @brief Construct a new pmt vector object from a serialized flatbuffer
     *
     * @param buf
     */
    pmt_vector_value(const uint8_t* buf);
    pmt_vector_value(const pmtf::Pmt* fb_pmt);

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

    T ref(size_t k);           // overload operator []
    void set(size_t k, T val); // overload [] =
    T* writable_elements();
    const T* elements();

    flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb);
};


// If we like this idea, I would rename this to pmt_vector and rename pmt_vector to
// something like _pmt_vector.
template <class T>
class pmt_vector {
public:
    using sptr = typename pmt_vector_value<T>::sptr;
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    // Constructors.  Try to match std vector constructors.
    // Right now I'm not allowing for a custom allocator, because I would believe 
    // that we would say that flatbuffer is our allocator.
    pmt_vector(size_type n): 
        d_ptr(pmt_vector_value<T>::make(std::vector<T>(n))) {}
    pmt_vector(size_type n, const value_type& val):
        d_ptr(pmt_vector_value<T>::make(std::vector<T>(n, val))) {}
    template <class InputIterator>
    pmt_vector(InputIterator first, InputIterator last):
        d_ptr(pmt_vector_value<T>::make(std::vector<T>(first, last))) {}
    pmt_vector(const pmt_vector& x):
        d_ptr(x.d_ptr) {}
    pmt_vector(std::initializer_list<value_type> il):
        d_ptr(pmt_vector_value<T>::make(std::vector<T>(il))) {}
    
    // Add in a few more constructors because we are wrapping a vector.
    pmt_vector(sptr p):
        d_ptr(p) {}
    // Allow for custom allocators (such as volk) in the constructor
    template <class alloc>
    pmt_vector(const std::vector<T, alloc>& x):
        d_ptr(pmt_vector_value<T>::make(x)) {}

    // TODO: Think about real iterators instead of pointers.
    value_type* begin() { return d_ptr->writable_elements(); }
    value_type* end() { return d_ptr->writable_elements() + size(); }
    const value_type* cbegin() { return d_ptr->writable_elements(); }
    const value_type* cend() { return d_ptr->writable_elements() + size(); }

    reference operator[] (size_type n) {
        // operator[] doesn't do bounds checking, use at for that
        // TODO: implement at
        auto data = d_ptr->writable_elements();
        return data[n];
    }
    const reference operator[] (size_type n) const {
        auto data = d_ptr->elements();
        return data[n];
    }
    
    sptr ptr() { return d_ptr; }
    size_type size() { return d_ptr->size(); }
    auto data_type() { return d_ptr->data_type(); }
    
    // When we switch to c++20, make this a concept.
    template <class U>
    bool operator==(U&& other) {
        if (other.size() != size()) return false;
        auto my_val = begin();
        for (auto&& val : other) {
            if (*my_val != val) return false;
            my_val++;
        }
        return true;
    }
private:
    sptr d_ptr;
};

typedef std::function<std::shared_ptr<pmt_base>(uint8_t*)> pmt_from_buffer_function;


#define IMPLEMENT_PMT_VECTOR(datatype, fbtype)                                        \
    template <>                                                                       \
    flatbuffers::Offset<void> pmt_vector_value<datatype>::rebuild_data(                     \
        flatbuffers::FlatBufferBuilder& fbb)                                          \
    {                                                                                 \
        /* fbb.Reset(); */                                                            \
        auto pmt = GetSizePrefixedPmt(buffer_pointer());                              \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                         \
        auto vec = fbb.CreateVector(fb_vec->data(), fb_vec->size());                  \
        Vector##fbtype##Builder vb(fbb);                                              \
        vb.add_value(vec);                                                            \
        return vb.Finish().Union();                                                   \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    void pmt_vector_value<datatype>::set_value(const std::vector<datatype>& val)            \
    {                                                                                 \
        _fbb.Reset();                                                                 \
        auto vec = _fbb.CreateVector(val.data(), val.size());                         \
        Vector##fbtype##Builder vb(_fbb);                                             \
        vb.add_value(vec);                                                            \
        _data = vb.Finish().Union();                                                  \
        build();                                                                      \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    void pmt_vector_value<datatype>::set_value(const datatype* data, size_t len)            \
    {                                                                                 \
        _fbb.Reset();                                                                 \
        auto vec = _fbb.CreateVector(data, len);                                      \
        Vector##fbtype##Builder vb(_fbb);                                             \
        vb.add_value(vec);                                                            \
        _data = vb.Finish().Union();                                                  \
        build();                                                                      \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    pmt_vector_value<datatype>::pmt_vector_value(const std::vector<datatype>& val)                \
        : pmt_base(Data::Vector##fbtype)                                              \
    {                                                                                 \
        set_value(val);                                                               \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    pmt_vector_value<datatype>::pmt_vector_value(const datatype* data, size_t len)                \
        : pmt_base(Data::Vector##fbtype)                                              \
    {                                                                                 \
        set_value(data, len);                                                         \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    pmt_vector_value<datatype>::pmt_vector_value(const uint8_t* buf)                              \
        : pmt_base(Data::Vector##fbtype)                                              \
    {                                                                                 \
        auto data = GetPmt(buf)->data_as_Vector##fbtype()->value();                   \
        size_t len = data->size();                                                    \
        set_value((const datatype*)data->Data(), len);                                \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    pmt_vector_value<datatype>::pmt_vector_value(const pmtf::Pmt* fb_pmt)                         \
        : pmt_base(Data::Vector##fbtype)                                              \
    {                                                                                 \
        auto data = fb_pmt->data_as_Vector##fbtype()->value();                        \
        size_t len = data->size();                                                    \
        set_value((const datatype*)data->Data(), len);                                \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    std::vector<datatype> pmt_vector_value<datatype>::value() const                         \
    {                                                                                 \
        auto pmt = GetSizePrefixedPmt(buffer_pointer());                              \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                         \
        /* _value.assign(fb_vec->begin(), fb_vec->end()); */                          \
        std::vector<datatype> ret(fb_vec->begin(), fb_vec->end());                    \
        return ret;                                                                   \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    const datatype* pmt_vector_value<datatype>::data()                                      \
    {                                                                                 \
        auto pmt = GetSizePrefixedPmt(_buf);                                          \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                         \
        return fb_vec->data();                                                        \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    size_t pmt_vector_value<datatype>::size()                                               \
    {                                                                                 \
        auto pmt = GetSizePrefixedPmt(_buf);                                          \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                         \
        return fb_vec->size();                                                        \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    datatype pmt_vector_value<datatype>::ref(size_t k)                                      \
    {                                                                                 \
        auto pmt = GetSizePrefixedPmt(_buf);                                          \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                         \
        if (k >= fb_vec->size())                                                      \
            throw std::runtime_error("PMT Vector index out of range");                \
        return (*fb_vec)[k];                                                          \
    }                                                                                 \
                                                                                      \
    template <>                                                                       \
    void pmt_vector_value<datatype>::set(size_t k, datatype val)                            \
    {                                                                                 \
        auto pmt =                                                                    \
            GetMutablePmt(buffer_pointer() + 4); /* assuming size prefix is 32 bit */ \
        auto fb_vec = ((pmtf::Vector##fbtype*)pmt->mutable_data())->mutable_value();  \
        if (k >= fb_vec->size())                                                      \
            throw std::runtime_error("PMT Vector index out of range");                \
        fb_vec->Mutate(k, val);                                                       \
    }                                                                                 \
    template <>                                                                       \
    const datatype* pmt_vector_value<datatype>::elements()                                  \
    {                                                                                 \
        auto pmt = GetSizePrefixedPmt(_buf);                                          \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                         \
        return (datatype*)(fb_vec->Data());                                           \
    }                                                                                 \
    template <>                                                                       \
    datatype* pmt_vector_value<datatype>::writable_elements()                               \
    {                                                                                 \
        auto pmt =                                                                    \
            GetMutablePmt(buffer_pointer() + 4); /* assuming size prefix is 32 bit */ \
        return (datatype*)(((pmtf::Vector##fbtype*)pmt->mutable_data())               \
                               ->mutable_value()                                      \
                               ->Data());                                             \
    }                                                                                 \
    template class pmt_vector_value<datatype>;

#define IMPLEMENT_PMT_VECTOR_CPLX(datatype, fbtype)                                     \
    template <>                                                                         \
    flatbuffers::Offset<void> pmt_vector_value<datatype>::rebuild_data(                       \
        flatbuffers::FlatBufferBuilder& fbb)                                            \
    {                                                                                   \
        auto pmt = GetSizePrefixedPmt(buffer_pointer());                                \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                           \
        auto vec = fbb.CreateVector(fb_vec->data(), fb_vec->size());                    \
        Vector##fbtype##Builder vb(fbb);                                                \
        vb.add_value(vec);                                                              \
        return vb.Finish().Union();                                                     \
    }                                                                                   \
    template <>                                                                         \
    void pmt_vector_value<datatype>::set_value(const std::vector<datatype>& val)              \
    {                                                                                   \
        _fbb.Reset();                                                                   \
        auto vec =                                                                      \
            _fbb.CreateVectorOfNativeStructs<fbtype, datatype>(val.data(), val.size()); \
        Vector##fbtype##Builder vb(_fbb);                                               \
        vb.add_value(vec);                                                              \
        _data = vb.Finish().Union();                                                    \
        build();                                                                        \
    }                                                                                   \
    template <>                                                                         \
    void pmt_vector_value<datatype>::set_value(const datatype* data, size_t len)              \
    {                                                                                   \
        _fbb.Reset();                                                                   \
        auto vec = _fbb.CreateVectorOfNativeStructs<fbtype, datatype>(data, len);       \
        Vector##fbtype##Builder vb(_fbb);                                               \
        vb.add_value(vec);                                                              \
        _data = vb.Finish().Union();                                                    \
        build();                                                                        \
    }                                                                                   \
    template <>                                                                         \
    pmt_vector_value<datatype>::pmt_vector_value(const std::vector<datatype>& val)                  \
        : pmt_base(Data::Vector##fbtype)                                                \
    {                                                                                   \
        set_value(val);                                                                 \
    }                                                                                   \
    template <>                                                                         \
    pmt_vector_value<datatype>::pmt_vector_value(const datatype* data, size_t len)                  \
        : pmt_base(Data::Vector##fbtype)                                                \
    {                                                                                   \
        set_value(data, len);                                                           \
    }                                                                                   \
    template <>                                                                         \
    pmt_vector_value<datatype>::pmt_vector_value(const uint8_t* buf)                                \
        : pmt_base(Data::Vector##fbtype)                                                \
    {                                                                                   \
        auto data = GetPmt(buf)->data_as_Vector##fbtype()->value();                     \
        size_t len = data->size();                                                      \
        set_value((const datatype*)data->Data(), len);                                  \
    }                                                                                   \
    template <>                                                                         \
    pmt_vector_value<datatype>::pmt_vector_value(const pmtf::Pmt* fb_pmt)                           \
        : pmt_base(Data::Vector##fbtype)                                                \
    {                                                                                   \
        auto data = fb_pmt->data_as_Vector##fbtype()->value();                          \
        size_t len = data->size();                                                      \
        set_value((const datatype*)data->Data(), len);                                  \
    }                                                                                   \
    template <>                                                                         \
    std::vector<datatype> pmt_vector_value<datatype>::value() const                           \
    {                                                                                   \
        auto pmt = GetSizePrefixedPmt(_buf);                                            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                           \
        std::vector<datatype> ret(fb_vec->size());                                      \
        /* because flatbuffers returns ptr to std::complex */                           \
        for (unsigned i = 0; i < fb_vec->size(); i++) {                                 \
            ret[i] = *(datatype*)fb_vec->Get(i);                                        \
        }                                                                               \
        return ret;                                                                     \
    }                                                                                   \
    template <>                                                                         \
    const datatype* pmt_vector_value<datatype>::data()                                        \
    {                                                                                   \
        auto pmt = GetSizePrefixedPmt(_buf);                                            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                           \
        return (datatype*)                                                              \
            fb_vec->Data(); /* no good native conversions in API, just cast here*/      \
    }                                                                                   \
    template <>                                                                         \
    size_t pmt_vector_value<datatype>::size()                                                 \
    {                                                                                   \
        auto pmt = GetSizePrefixedPmt(_buf);                                            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                           \
        return fb_vec->size();                                                          \
    }                                                                                   \
    template <>                                                                         \
    datatype pmt_vector_value<datatype>::ref(size_t k)                                        \
    {                                                                                   \
        auto pmt = GetSizePrefixedPmt(_buf);                                            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                           \
        if (k >= fb_vec->size())                                                        \
            throw std::runtime_error("PMT Vector index out of range");                  \
        return *((datatype*)(*fb_vec)[k]); /* hacky cast */                             \
    }                                                                                   \
    template <>                                                                         \
    const datatype* pmt_vector_value<datatype>::elements()                                    \
    {                                                                                   \
        auto pmt = GetSizePrefixedPmt(_buf);                                            \
        auto fb_vec = pmt->data_as_Vector##fbtype()->value();                           \
        return (datatype*)(fb_vec->Data()); /* hacky cast*/                             \
    }                                                                                   \
    template <>                                                                         \
    void pmt_vector_value<datatype>::set(size_t k, datatype val)                              \
    {                                                                                   \
        auto pmt =                                                                      \
            GetMutablePmt(buffer_pointer() + 4); /* assuming size prefix is 32 bit */   \
        auto fb_vec = ((pmtf::Vector##fbtype*)pmt->mutable_data())->mutable_value();    \
        if (k >= fb_vec->size())                                                        \
            throw std::runtime_error("PMT Vector index out of range");                  \
        fb_vec->Mutate(k, (fbtype*)&val); /* hacky cast */                              \
    }                                                                                   \
    template <>                                                                         \
    datatype* pmt_vector_value<datatype>::writable_elements()                                 \
    {                                                                                   \
        auto pmt =                                                                      \
            GetMutablePmt(buffer_pointer() + 4); /* assuming size prefix is 32 bit */   \
        auto mutable_obj = ((pmtf::Vector##fbtype*)pmt->mutable_data())                 \
                               ->mutable_value()                                        \
                               ->GetMutableObject(0);                                   \
        return (datatype*)(mutable_obj); /* hacky cast */                               \
    }                                                                                   \
    template class pmt_vector_value<datatype>;

} // namespace pmtf
