#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gnuradio/buffer.h>
#include <pmtf/map.hpp>
#include <pmtf/vector.hpp>

namespace gr {

class buffer_pdu_reader;

/**
 * @brief PDU based buffer
 * 
 * @details Contrived buffer that wraps a PDU for passing to work function
 * 
 */

class buffer_pdu : public buffer
{
private:
    uint8_t * p_buffer;

public:
    using sptr = std::shared_ptr<buffer_pdu>;
    buffer_pdu(pmtf::map pdu, size_t num_items,
                            size_t item_size,
                      std::shared_ptr<buffer_properties> buf_properties);

    static buffer_sptr make(pmtf::map pdu, size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);

    void* read_ptr(size_t index) override;
    void* write_ptr() override;

    void post_write(int num_items) override;

    std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize) override;
};

class buffer_pdu_reader : public buffer_reader
{
public:
    static buffer_reader_sptr make(pmtf::pmt pdu)
    {
        // decompose the PDU
        auto meta = pmtf::map(pdu)["meta"];
        auto data = pmtf::map(pdu)["data"];

        
        

        // switch (data.data_type())
        // {
        //     case VectorFloat32 :
        //     case VectorFloat64 :
        //     case VectorComplex64 :
        //     case VectorComplex128 :
        //     case VectorInt8 :
        //     case VectorInt16:
        //     case VectorInt32:
        //     case VectorInt64:
        //     case VectorUInt8 :
        //     case VectorUInt16 :
        //     case VectorUInt32 :
        //     case VectorUInt64 :
        // }

    }
    buffer_pdu_reader(pmtf::pmt pdu, size_t itemsize)
        : buffer_reader(nullptr, std::make_shared<buffer_properties>(), itemsize, 0)
    {
    }

    void post_read(int num_items) override;
};

} // namespace gr
