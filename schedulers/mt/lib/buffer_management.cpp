#include "buffer_management.hpp"
#include <gnuradio/domain_adapter.hpp>
#include <numeric>

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

        // Determine whether the blocks on either side of the edge are domain adapters
        // If so, Domain adapters need their own buffer explicitly set
        // Edge buffer becomes the domain adapter - edges are between actual blocks

        // Terminology for Block/Domain Adapter connections at Domain Crossings
        //               SRC                   DST
        //     +-----------+  DST         SRC  +-----------+       +---
        //     |           |  +----+   +----+  |           |       |
        //     |   BLK1    +->+ DA +-->+ DA +->+   BLK2    +------>+
        //     |           |  +----+   +----+  |           |       |
        //     +-----------+                   +-----------+       +---
        //        DOMAIN1                               DOMAIN2


        auto src_da_cast = std::dynamic_pointer_cast<domain_adapter>(e->src().node());
        auto dst_da_cast = std::dynamic_pointer_cast<domain_adapter>(e->dst().node());

        if (src_da_cast != nullptr) {
            if (src_da_cast->buffer_location() == buffer_location_t::LOCAL) {
                buffer_sptr buf;

                if (e->has_custom_buffer()) {
                    buf = e->buffer_factory()(num_items, e->itemsize(), e->buf_properties());
                } else {
                    buf = buf_factory(num_items, e->itemsize(), buf_props);
                }

                src_da_cast->set_buffer(buf);
                auto tmp = std::dynamic_pointer_cast<buffer>(src_da_cast);
                d_edge_buffers[e->identifier()] = tmp;
                GR_LOG_INFO(_logger, "Edge: {}, Buf: {}, {} bytes, {} items of size {}", e->identifier(), buf->type(), buf->size(), buf->num_items(), buf->item_size());
            } else {
                d_edge_buffers[e->identifier()] =
                    std::dynamic_pointer_cast<buffer>(src_da_cast);
                GR_LOG_INFO(_logger, "Edge: {}, Buf: SRC_DA", e->identifier());
            }
        } else if (dst_da_cast != nullptr) {
            if (dst_da_cast->buffer_location() == buffer_location_t::LOCAL) {
                buffer_sptr buf;

                if (e->has_custom_buffer()) {
                    buf = e->buffer_factory()(num_items, e->itemsize(), e->buf_properties());
                } else {
                    buf = buf_factory(num_items, e->itemsize(), buf_props);
                }

                dst_da_cast->set_buffer(buf);
                auto tmp = std::dynamic_pointer_cast<buffer>(dst_da_cast);
                d_edge_buffers[e->identifier()] = tmp;
                GR_LOG_INFO(_logger, "Edge: {}, Buf: {}, {} bytes, {} items of size {}", e->identifier(), buf->type(), buf->size(), buf->num_items(), buf->item_size());
            } else {
                d_edge_buffers[e->identifier()] =
                    std::dynamic_pointer_cast<buffer>(dst_da_cast);
                GR_LOG_INFO(_logger, "Edge: {}, Buf: DST_DA", e->identifier());
            }

        }
        // If there are no domain adapter involved, then simply give this edge a
        // buffer
        else {
            buffer_sptr buf;
            if (e->has_custom_buffer()) {
                buf = e->buffer_factory()(num_items, e->itemsize(), e->buf_properties());
            } else {
                buf = buf_factory(num_items, e->itemsize(), buf_props);
            }

        // FIXME: Using a string for edge map lookup is inefficient
            d_edge_buffers[e->identifier()] = buf;
            GR_LOG_INFO(_logger, "Edge: {}, Buf: {}, {} bytes, {} items of size {}", e->identifier(), buf->type(), buf->size(), buf->num_items(), buf->item_size());
        }
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

    auto grblock = std::dynamic_pointer_cast<block>(e->src().node());
    if (grblock == nullptr) // might be a domain adapter, not a block
    {
        grblock = std::dynamic_pointer_cast<block>(e->dst().node());
    }

    if (grblock->output_multiple_set())
    {
        nitems =
            std::max(nitems, static_cast<size_t>(2 * (grblock->output_multiple())));
    }

    // FIXME: Downstream block connections get messed up by domain adapters
    //   Need to tag the blocks before they get partitioned
    //   and store the information in the edge objects
    //   also allow for different rates out of different ports

    // // If any downstream blocks are decimators and/or have a large output_multiple,
    // // ensure we have a buffer at least twice their decimation
    // // factor*output_multiple

    auto blocks = fg->calc_downstream_blocks(grblock, e->src().port());

    for (auto&  p : blocks) {
        // block_sptr dgrblock = cast_to_block_sptr(*p);
        // if (!dgrblock)
        //     throw std::runtime_error("allocate_buffer found non-gr::block");

        // double decimation = (1.0 / dgrblock->relative_rate());
        double decimation = (1.0 / p->relative_rate());
        int multiple = p->output_multiple();
        nitems =
            std::max(nitems, static_cast<size_t>(2 * (decimation * multiple)));
            // std::max(nitems, static_cast<int>(2 * (decimation * multiple)));
    }

    return nitems;
}


} // namespace schedulers
} // namespace gr
