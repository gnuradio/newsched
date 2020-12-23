#pragma once

#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/flat_graph.hpp>
#include <gnuradio/logging.hpp>

namespace gr {
namespace schedulers {

class buffer_manager
{
private:
    const int s_fixed_buf_size;
    static const int s_min_items_to_process = 1;
    const size_t s_max_buf_items; // = s_fixed_buf_size / 2;
    const size_t s_min_buf_items = 1;
    std::map<port_sptr, std::vector<buffer_sptr>> d_block_buffers;

    // make these two go away
    std::map<std::string, buffer_sptr> d_edge_buffers;
    std::map<std::string, edge_sptr> d_edge_catalog;

    std::string _name = "buffer_manager";
    logger_sptr _logger;
    logger_sptr _debug_logger;

public:
    typedef std::shared_ptr<buffer_manager> sptr;
    buffer_manager(const unsigned int default_buffer_size_in_bytes)
        : s_fixed_buf_size(default_buffer_size_in_bytes),
          s_max_buf_items(default_buffer_size_in_bytes - 1)
    {
        _logger = logging::get_logger(_name, "default");
        _debug_logger = logging::get_logger(_name + "_dbg", "debug");
    }
    ~buffer_manager() {}

    buffer_sptr get_input_buffer(port_sptr p)
    {
        return d_block_buffers[p][0];
    }

    std::vector<buffer_sptr>& get_output_buffers(port_sptr p)
    {
        return d_block_buffers[p];
    }   

    void initialize_buffers(flat_graph_sptr fg, buffer_factory_function buf_factory, std::shared_ptr<buffer_properties> buf_props );
    
private:
    int get_buffer_num_items(edge_sptr e, flat_graph_sptr fg);
};


} // namespace schedulers
} // namespace gr