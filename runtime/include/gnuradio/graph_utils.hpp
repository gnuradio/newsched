
#include <gnuradio/graph.hpp>
#include <gnuradio/scheduler.hpp>
#include <gnuradio/domain.hpp>
#include <gnuradio/neighbor_interface.hpp>

namespace gr
{

    struct graph_partition_info
    {
        std::vector<graph_sptr> subgraphs;
        std::map<scheduler_sptr, std::map<nodeid_t, neighbor_interface_info>>
            neighbor_map_per_scheduler;
        std::vector<scheduler_sptr> partition_scheds;
    };

    struct graph_utils
    {
        static graph_partition_info partition(graph_sptr input_graph,
                                               std::vector<scheduler_sptr> scheds,
                                               std::vector<domain_conf>& confs);
        
    };
}