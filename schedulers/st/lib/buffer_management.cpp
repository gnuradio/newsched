#include "buffer_management.hpp"

namespace gr {
namespace schedulers {


void buffer_manager::initialize_buffers(flat_graph_sptr fg,
                                        buffer_factory_function buf_factory,
                                        std::shared_ptr<buffer_properties> buf_props)
{
    // not all edges may be used
    for (auto e : fg->edges()) {
        // every edge needs a buffer
        auto num_items = get_buffer_num_items(e, fg);

        // Give the edge a buffer
        buffer_sptr buf;
        if (e->has_custom_buffer()) {
            buf = e->buffer_factory()(num_items, e->itemsize(), e->buf_properties());
        } else {
            buf = buf_factory(num_items, e->itemsize(), buf_props);
        }

        // FIXME: Using a string for edge map lookup is inefficient
        d_edge_buffers[e->identifier()] = buf;
        gr_log_info(_logger, "Edge: {}, Buf: {}", e->identifier(), buf->type());
    }

    for (auto& b : fg->calc_used_blocks()) {
        port_vector_t input_ports = b->input_stream_ports();
        port_vector_t output_ports = b->output_stream_ports();

        for (auto p : input_ports) {
            d_block_buffers[p] = std::vector<buffer_sptr>{};
            edge_vector_t ed = fg->find_edge(p);
            for (auto e : ed)
                d_block_buffers[p].push_back(d_edge_buffers[e->identifier()]);
        }

        for (auto p : output_ports) {
            d_block_buffers[p] = std::vector<buffer_sptr>{};
            edge_vector_t ed = fg->find_edge(p);
            for (auto e : ed)
                d_block_buffers[p].push_back(d_edge_buffers[e->identifier()]);
        }
    }
}

int buffer_manager::get_buffer_num_items(edge_sptr e, flat_graph_sptr fg)
{
    size_t item_size = e->itemsize();

    // *2 because we're now only filling them 1/2 way in order to
    // increase the available parallelism when using the TPB scheduler.
    // (We're double buffering, where we used to single buffer)
    size_t nitems = (s_fixed_buf_size * 2) / item_size;

    return nitems;
}


} // namespace schedulers
} // namespace gr