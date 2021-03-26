
#include <gnuradio/domain.hpp>
#include <gnuradio/graph.hpp>
#include <gnuradio/neighbor_interface_info.hpp>
#include <gnuradio/scheduler.hpp>

namespace gr {

/**
 * @brief Struct providing info for particular partition of flowgraph and how it relates
 * to the other partitions
 *
 * scheduler - the sptr to the scheduler for this partition
 * subgraph - the portion of the flowgraph controlled by this scheduler
 * neighbor map - a per block map of neighbor_interface_info
 */
struct graph_partition_info {
    scheduler_sptr scheduler;
    graph_sptr subgraph;
};

typedef std::vector<graph_partition_info> graph_partition_info_vec;

struct graph_utils {
    /**
     * @brief Partition the graph into subgraphs while preserving the neighbor scheduler map
     * 
     * @param input_graph 
     * @param scheds 
     * @param confs 
     * @param neighbor_intf_map 
     * @return graph_partition_info_vec 
     */
    static graph_partition_info_vec
    partition(graph_sptr input_graph,
              std::vector<scheduler_sptr> scheds,
              std::vector<domain_conf>& confs);
};
} // namespace gr
