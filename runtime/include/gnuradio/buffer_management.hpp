#pragma once

#include <gnuradio/flat_graph.hpp>
#include <gnuradio/logging.hpp>

namespace gr {

class buffer_manager
{
private:
    const int s_fixed_buf_size;
    static const int s_min_items_to_process = 1;
    const size_t s_min_buf_items = 1;

    std::string _name = "buffer_manager";
    logger_sptr _logger;
    logger_sptr _debug_logger;

public:
    typedef std::shared_ptr<buffer_manager> sptr;
    buffer_manager(const unsigned int default_buffer_size_in_bytes)
        : s_fixed_buf_size(default_buffer_size_in_bytes)
    {
        _logger = logging::get_logger(_name, "default");
        _debug_logger = logging::get_logger(_name + "_dbg", "debug");
    }
    ~buffer_manager() {}

    void initialize_buffers(flat_graph_sptr fg,
                            std::shared_ptr<buffer_properties> buf_props);

private:
    int get_buffer_num_items(edge_sptr e, flat_graph_sptr fg);
};

} // namespace gr