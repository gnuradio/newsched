#include <gnuradio/buffer_pdu.h>

namespace gr {

void buffer_pdu::post_write(int num_items)
{
    _total_written += num_items;
}

void buffer_pdu_reader::post_read(int num_items)
{
    _total_read += num_items;
}

} // namespace gr
