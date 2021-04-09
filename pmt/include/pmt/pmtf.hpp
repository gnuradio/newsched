#pragma once

#include <pmt/pmt_generated.h>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace pmtf {

class pmt_base : public std::enable_shared_from_this<pmt_base>
{
public:
    typedef std::shared_ptr<pmt_base> sptr;

    virtual ~pmt_base() {}

    Data data_type() const { return _data_type; };
    virtual flatbuffers::Offset<void> rebuild_data(flatbuffers::FlatBufferBuilder& fbb) = 0;

    bool serialize(std::streambuf& sb)
    {
        serialize_setup();
        uint8_t* buf = _fbb.GetBufferPointer();
        int size = _fbb.GetSize();
        return sb.sputn((const char*)buf, size) != std::streambuf::traits_type::eof();
    }

    static sptr from_buffer(const uint8_t* buf, size_t size);
    static sptr from_pmt(const pmtf::Pmt *fb_pmt);
    static sptr deserialize(std::streambuf& sb)
    {
        char buf[4];
        sb.sgetn(buf, 4);
        // assuming little endian for now
        uint32_t size = *((uint32_t*)&buf[0]);
        uint8_t tmp_buf[size];
        sb.sgetn((char*)tmp_buf, size);

        return from_buffer(tmp_buf, size);
    }

    void build()
    {
        // std::cout << "fb size: " << _fbb.GetSize() << std::endl;
        PmtBuilder pb(_fbb);
        pb.add_data_type(_data_type);
        pb.add_data(_data);
        _blob = pb.Finish();
        _fbb.FinishSizePrefixed(_blob);
        _buf = _fbb.GetBufferPointer();
        _pmt_fb = GetSizePrefixedPmt(_buf);
        // std::cout << "fb size: " << _fbb.GetSize() << std::endl;
    }

    flatbuffers::Offset<Pmt> build(flatbuffers::FlatBufferBuilder& fbb)
    {
        auto data_offset = rebuild_data(fbb);
        PmtBuilder pb(fbb);
        pb.add_data_type(_data_type);
        pb.add_data(data_offset);
        return pb.Finish();
    }

    size_t size() const { return _fbb.GetSize(); }
    uint8_t* buffer_pointer() const
    {
        if (_vec_buf.size() > 0) {
            return (uint8_t *)&_vec_buf[0];
        } else {
            return _fbb.GetBufferPointer();
        }
    }

    void set_buffer(const uint8_t *data, size_t len)
    {
        _vec_buf.resize(len);
        memcpy(&_vec_buf[0], data, len);
    }

    bool operator==(const pmt_base& other)
    {
        auto eq_types = (data_type() == other.data_type());
        auto eq_size = (size() == other.size());
        auto eq_data = !memcmp(buffer_pointer(), other.buffer_pointer(), size());
        return (eq_types && eq_size && eq_data);
    }

protected:
    pmt_base(Data data_type) : _data_type(data_type){};
    virtual void serialize_setup() {}
    Data _data_type;
    flatbuffers::FlatBufferBuilder _fbb;
    flatbuffers::Offset<void> _data;
    flatbuffers::Offset<Pmt> _blob;
    const pmtf::Pmt* _pmt_fb = nullptr;
    const uint8_t* _buf = nullptr;
    std::vector<uint8_t> _vec_buf; // if the buffer came from serialized data, just store that here

    // PmtBuilder _builder;
};

typedef pmt_base::sptr pmt_sptr;


} // namespace pmtf
