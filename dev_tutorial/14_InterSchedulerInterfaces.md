# Inter-Scheduler Interfaces

Apart from domain adapters, which serve to get data across domain boundaries via the connected ports, schedulers must communicate with each other through the input queues of their threads, notifying each other when work has been done amongst other things

Previously, we had given a scheduler a `push_message` method. Let's generalize this now to a `neighbor_interface` class from which schedulers will derive.  Note that this interface only has one method `push_message` which is the only way to communicate with a scheduler.

```cpp
struct neighbor_interface
{
    neighbor_interface() {}
    virtual ~neighbor_interface() {}
    virtual void push_message(scheduler_message_sptr msg) = 0;
};
```

Inside a scheduler, blocks/ports need to be traced back to the schedulers that they are connected to.  The `neighbor_interface_info` struct holds this information.  **Currently there is a flaw that a block can have only one upstream neighbor**, which is not true - each port can have only one upstream neighbor. This needs to be fixed in the design.

```cpp
struct neighbor_interface_info {
    std::shared_ptr<neighbor_interface> upstream_neighbor_intf = nullptr;
    nodeid_t upstream_neighbor_blkid = -1;
    std::vector<std::shared_ptr<neighbor_interface>> downstream_neighbor_intf;
    std::vector<nodeid_t> downstream_neighbor_blkids;

    void set_upstream(std::shared_ptr<neighbor_interface> intf, nodeid_t blkid);
    void add_downstream(std::shared_ptr<neighbor_interface> intf, nodeid_t blkid);
};

// FIXME - this mapping prevents a block from going to two different schedulers
typedef std::map<nodeid_t, neighbor_interface_info> neighbor_interface_map;
```
Also, in order to incorporate the `neighbor_interface` with the schedulers we will
1. Derive from neighbor interface
2. Pass in a `neighbor_interface_map` upon initialization

(`runtime/include/gnuradio/scheduler.hpp`)
```diff
-class scheduler : public std::enable_shared_from_this<scheduler>
+class scheduler : public std::enable_shared_from_this<scheduler>, public neighbor_interface
```
```diff
-    virtual void initialize(flat_graph_sptr fg, flowgraph_monitor_sptr fgmon) = 0;
+    virtual void initialize(flat_graph_sptr fg, flowgraph_monitor_sptr fgmon,
+               neighbor_interface_map scheduler_adapter_map = neighbor_interface_map()) = 0;
```

## Partitioning the Flowgraph
The burden is not on the user to create all these hidden connections such as Domain Adapters and Inter-Scheduler interfaces, so we take the simply-created flowgraph with blocks and edges, and partition it across the specified schedulers

### User Configuration
Say the user wants to create a simple flowgraph which is `src->block1->block2->snk`, and there is a domain boundary between `block1` and `block2`.  The user needs to provide the following to inform the flowgraph partitioning where the blocks should go

1. Create a `domain adapter configuration` - this is for the specific type of domain adapter desired

```cpp
    auto da_conf = domain_adapter_direct_conf::make(buffer_preference_t::UPSTREAM);
```
2. Create a vector of domain configurations, one for each domain.  The domain configuration specifies the `scheduler`, the `blocks`, and the `domain adapter configuration` 
```cpp
    domain_conf_vec dconf{ domain_conf(sched1, { src, block1 }, da_conf),
                           domain_conf(sched2, { block2, snk }, da_conf) };
```
3. Tell the flowgraph to partition the graph
```cpp
    fg->partition(dconf);
```

Note: the schedulers would need to have been added to the flowgraph object using `fg->add_scheduler(schedN)`

### Partitioning the graph
The flowgraph object is now responsible for breaking up the user flowgraph into smaller subgraphs connected by domain adapters.  Inside `flowgraph::partition`, the bulk of the work to this end is being done by `graph_utils::partition`, and there is a lot going on in this method, which takes in the graph, the schedulers, the domain confs, and optionally the `neighbor_interface_map` and returns an information struct about the partitioned graph

```cpp
static graph_partition_info_vec
partition(graph_sptr input_graph,
            std::vector<scheduler_sptr> scheds,
            std::vector<domain_conf>& confs,
            neighbor_interface_map neighbor_intf_map =
                neighbor_interface_map());
```
The `partition` method performs the following

1. Create the subgraphs from the nodes that are cohesive in each domain
    - In the example above this would create `subgraph[0]` with `src` and `block1`, then `subgraph[1]` with `block2` and `snk`
2. Save the edges that crossed domains
    - Such as the edge between `block1` and `block2`
3. Add any orphan nodes into the appropriate subgraph, even though they are not directly connected but are encapsulated by the domain definition
4. Set up the _Domain Adapters_ and _Inter-Scheduler Interfaces_ from the saved domain crossings from step (2)

Let's look at step (4) in more detail, and use the connection between `block1` and `block2` as the example
![Simple Configuration](images/domain_adapters_3.png)

For each detected crossing, we need to 
1. Find the `src` and `dst` nodes from the stored subgraphs (some brute force searching)
```cpp
    for (auto c : domain_crossings) {
        // Find the subgraph that holds src block
        graph_sptr src_block_graph = nullptr;
        for (auto info : ret) {
            auto g = info.subgraph;
            auto blocks = g->calc_used_nodes();
            if (std::find(blocks.begin(), blocks.end(), c->src().node()) != blocks.end()) {
                src_block_graph = g;
                break;
            }
        }

        // Find the subgraph that holds dst block
        graph_sptr dst_block_graph = nullptr;
        for (auto info : ret) {
            auto g = info.subgraph;
            auto blocks = g->calc_used_nodes();
            if (std::find(blocks.begin(), blocks.end(), c->dst().node()) != blocks.end()) {
                dst_block_graph = g;
                break;
            }
        }
```

2. Create the _Domain Adapter Pair_ using the appropriate conf associated with the domain or the specific edge
```cpp
// use the conf to produce the domain adapters
auto da_pair = da_conf->make_domain_adapter_pair(
    c->src().port(),
    c->dst().port(),
    "da_" + c->src().node()->alias() + "->" + c->dst().node()->alias());

```

3. Attach Domain Adapters to the `src` and `dst` blocks
    - In our example, `block1` is the `src` and `block2` is the `dst`
    - We also propagate any custom buffers that were configured on the edge to the domain adapter connections since the original edge `E1` has been replaced

```cpp
auto da_src = std::get<0>(da_pair);
auto da_dst = std::get<1>(da_pair);

src_block_graph->connect(c->src(),
                            node_endpoint(da_src, da_src->all_ports()[0]))->set_custom_buffer(c->buffer_factory(), c->buf_properties());
dst_block_graph->connect(node_endpoint(da_dst, da_dst->all_ports()[0]),
                            c->dst())->set_custom_buffer(c->buffer_factory(), c->buf_properties());

```
With these new connections in place, at this point we should have:
![Simple Configuration with Domain Adapters](images/domain_adapters_4.png)


4. Configure the Inter-Scheduler Interfaces by giving the `dst` neighbor map a pointer to `src`'s `neighbor_interface` and vice-versa

```cpp

// Set the block id to "other scheduler" maps
auto dst_block_id = c->dst().node()->id();
auto src_block_id = c->src().node()->id();

ret[sched_index_map[block_to_scheduler_map[dst_block_id]]]
    .neighbor_map[dst_block_id]
    .set_upstream(block_to_scheduler_map[src_block_id], src_block_id);

ret[sched_index_map[block_to_scheduler_map[src_block_id]]]
    .neighbor_map[src_block_id]
    .add_downstream(block_to_scheduler_map[dst_block_id], dst_block_id);

```

In the above code, `ret` is a vector of `graph_partition_info`, which is indexed by some map lookups that were set earlier in the code.  In our example, it is essentially (assume `block1` has global id of `2` and `block2` has global id of `3`) doing the following.  They are just getting each other's scheduler pointers.

```cpp
ret[1].neighbor_map[3].set_upstream(sched1)
ret[0].neighbor_map[2].add_downstream(sched2)
```