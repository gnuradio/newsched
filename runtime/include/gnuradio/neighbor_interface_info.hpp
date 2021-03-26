#pragma once

#include <gnuradio/node.hpp>
#include <gnuradio/neighbor_interface.hpp>
#include <map>
#include <vector>

namespace gr {

/**
 * @brief Keep track of upstream and downstream neighbors for a block
 *
 * A block can only have one upstream neighbor - FIXME - not true - a port can only have
 * one upstream neighbor
 *
 */
struct neighbor_interface_info {
    std::shared_ptr<neighbor_interface> upstream_neighbor_intf = nullptr;
    nodeid_t upstream_neighbor_blkid = -1;
    std::vector<std::shared_ptr<neighbor_interface>> downstream_neighbor_intf;
    std::vector<nodeid_t> downstream_neighbor_blkids;

    void set_upstream(std::shared_ptr<neighbor_interface> intf, nodeid_t blkid)
    {
        upstream_neighbor_intf = intf;
        upstream_neighbor_blkid = blkid;
    }

    void add_downstream(std::shared_ptr<neighbor_interface> intf, nodeid_t blkid)
    {
        downstream_neighbor_intf.push_back(intf);
        downstream_neighbor_blkids.push_back(blkid);
    }
};

} // namespace gr