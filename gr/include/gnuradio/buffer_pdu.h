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
    uint8_t* _buffer;
    pmtf::pmt _pdu;


public:
    using sptr = std::shared_ptr<buffer_pdu>;
    buffer_pdu(pmtf::map pdu,
               size_t num_items,
               size_t item_size,
               std::shared_ptr<buffer_properties> buf_properties);

    static buffer_sptr make(pmtf::map pdu,
                            size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);

    void* read_ptr(size_t index) override;
    void* write_ptr() override;

    void post_write(int num_items) override;

    std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize) override
    {
        // do nothing because readers are detached from the writer
        return nullptr;
    }
};

class buffer_pdu_reader : public buffer_reader
{
private:
    uint8_t* _buffer;
    pmtf::pmt _pdu;
    size_t _n_items;

public:
    static buffer_reader_sptr
    make(size_t num_items, size_t itemsize, uint8_t* items, pmtf::pmt pdu)
    {
        // // decompose the PDU
        // auto meta = pmtf::map(pdu)["meta"];
        // auto data = pmtf::map(pdu)["data"];

        // _pdu = pdu; // hold the pdu  -- do i need to??
        // _buffer = items;

        return buffer_reader_sptr(new buffer_pdu_reader(num_items, itemsize, items, pdu));
    }

    buffer_pdu_reader(size_t num_items, size_t itemsize, uint8_t* items, pmtf::pmt pdu)
        : buffer_reader(nullptr, std::make_shared<buffer_properties>(), itemsize, 0)
    {
        _pdu = pdu; // hold the pdu  -- do i need to??
        _buffer = items;
        _n_items = num_items;
    }

    void post_read(int num_items) override;
};

} // namespace gr
