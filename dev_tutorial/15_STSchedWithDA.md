# Single Threaded Scheduler with Domain Adapters

Now that we have partitioned a flowgraph and configured domain adapters, we can see how the Single Threaded Scheduler can use this information to coordinate with other schedulers.

## Buffer Manager
For a graph with Domain Adapters, figuring out how to set up buffers becomes more complicated as we cycle through the edges and construct a buffer object.  First, the terminology becomes a bit more convoluted, as there are two pairs of blocks involved, and each has a `src` and `dst`

```
//               SRC                   DST
//     +-----------+  DST         SRC  +-----------+       +---
//     |           |  +----+   +----+  |           |       |
//     |   BLK1    +->+ DA +-->+ DA +->+   BLK2    +------>+
//     |           |  +----+   +----+  |           |       |
//     +-----------+                   +-----------+       +---
//        DOMAIN1                               DOMAIN2

```
For each edge in the _subgraph_ assigned to this scheduler, is either end a domain adapter? - let RTTI determine
```cpp
auto src_da_cast = std::dynamic_pointer_cast<domain_adapter>(e->src().node());
auto dst_da_cast = std::dynamic_pointer_cast<domain_adapter>(e->dst().node());
```
If so, (`src_da_cast` being potentially the domain adapter in this case attached to `BLK2`), we establish a buffer **ONLY** if the DA is configured to have a `LOCAL` buffer.  In either case, we store the DA directly in `d_edge_buffers` as the buffer for the edge between the block and the domain adapter (the original edge between `BLK1` and `BLK2` is gone at this point)
```cpp
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
        gr_log_info(_logger, "Edge: {}, Buf: {}", e->identifier(), buf->type());
    } else {
        d_edge_buffers[e->identifier()] =
            std::dynamic_pointer_cast<buffer>(src_da_cast);
        gr_log_info(_logger, "Edge: {}, Buf: SRC_DA", e->identifier());
    }

```
If the edge does not involve a domain adapter, then we just give the edge a buffer as normal and store that buffer in `d_edge_buffers`

There are no changes to the `graph_exector` since Domain Adapters appear as buffers and adhere to the buffer API.

## Thread Wrapper
Now with the awareness of other schedulers through the neighbor interface maps, the thread wrapper now needs to notify the correct schedulers after work has been completed

In the thread, when `run_one_iteration` returns that work has been done (`READY` status for a particular block), we now need to figure out who our upstream and downstream neighbors are for this block ... 

```cpp
neighbor_interface_info info_us, info_ds;
auto has_us = get_neighbors_upstream(elem.first, info_us);
auto has_ds = get_neighbors_downstream(elem.first, info_ds);

if (has_us) {
    sched_to_notify_upstream.push_back(info_us);
}
if (has_ds) {
    sched_to_notify_downstream.push_back(info_ds);
}
```
... and notify them by pushing a scheduler action message into the queue
```cpp
if (!sched_to_notify_upstream.empty()) {
    for (auto& info : sched_to_notify_upstream) {
        notify_upstream(info.upstream_neighbor_intf, info.upstream_neighbor_blkid);
    }
}

if (!sched_to_notify_downstream.empty()) {
    for (auto& info : sched_to_notify_downstream) {
        int idx = 0;
        for (auto& intf : info.downstream_neighbor_intf) {
            notify_downstream(intf, info.downstream_neighbor_blkids[idx]);
            idx++;
        }
    }
}
```

## User Configuration
Let's now set up a QA test which creates a small flowgraph, partitions across 2 schedulers, and runs as normal

As before, we set up our blocks, connect them, and establish what we expect our output to be.
```cpp
TEST(SchedulerSTTest, DomainAdapterBasic)
{
    std::vector<float> input_data{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<float> expected_data;
    for (auto d : input_data) {
        expected_data.push_back(100.0 * 200.0 * d);
    }

    auto src = blocks::vector_source_f::make(input_data, false);
    auto mult1 = blocks::multiply_const_ff::make(100.0);
    auto mult2 = blocks::multiply_const_ff::make(200.0);
    auto snk = blocks::vector_sink_f::make();

    auto fg = flowgraph::make();
    fg->connect(src, 0, mult1, 0);
    fg->connect(mult1, 0, mult2, 0);
    fg->connect(mult2, 0, snk, 0);
```
But now, we have 2 schedulers, which must be added to the flowgraph
```cpp
    auto sched1 = schedulers::scheduler_st::make("sched1");
    auto sched2 = schedulers::scheduler_st::make("sched2");

    fg->add_scheduler(sched1);
    fg->add_scheduler(sched2);
```
The domain adapter configuration and domain configurations specify how the partitioning will take place.  Partition also achieves the finalization step for the flowgraph

```cpp
    auto da_conf = domain_adapter_direct_conf::make(buffer_preference_t::UPSTREAM);

    domain_conf_vec dconf{ domain_conf(sched1, { src, mult1 }, da_conf),
                           domain_conf(sched2, { mult2, snk }, da_conf) };

    fg->partition(dconf);
```
Run and check with `ninja test` and everything should be good
```cpp

    fg->start();
    fg->wait();

    EXPECT_EQ(snk->data(), expected_data);
}
```
```
[0/1] Running all tests.
1/1 Single Threaded Scheduler Tests OK             0.27s
```