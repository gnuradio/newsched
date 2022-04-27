#include <gnuradio/buffer_pdu.h>

namespace gr {

buffer_pdu::buffer_pdu(pmtf::map pdu,
                       size_t num_items,
                       size_t item_size,
                       std::shared_ptr<buffer_properties> buf_properties)
    : buffer(num_items, item_size, buf_properties)
{

    set_type("buffer_pdu");
}

buffer_sptr buffer_pdu::make(pmtf::map pdu,
                             size_t num_items,
                             size_t item_size,
                             std::shared_ptr<buffer_properties> buffer_properties)
{
    return buffer_sptr(new buffer_pdu(pdu, num_items, item_size, buffer_properties));
}

void* buffer_pdu::read_ptr(size_t index) { return (void*)&_buffer[index]; }
void* buffer_pdu::write_ptr() { return (void*)&_buffer[_write_index]; }

void buffer_pdu::post_write(int num_items)
{
    _total_written += num_items;
}

std::shared_ptr<buffer_reader>
buffer_pdu::add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize)
{
    std::shared_ptr<buffer_pdu_reader> r(
        new buffer_pdu_reader(shared_from_this(), buf_props, itemsize, _write_index));
    _readers.push_back(r.get());
    return r;
}


void buffer_pdu_reader::post_read(int num_items)
{
    _total_read += num_items;
}

} // namespace gr
