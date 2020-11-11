#include "scheduler_st.hpp"

namespace gr {

namespace schedulers {


scheduler_st::scheduler_st(const std::string name, const unsigned int fixed_buf_size)
    : scheduler(name),
      s_fixed_buf_size(fixed_buf_size),
      s_max_buf_items(fixed_buf_size - 1)
{
    _default_buf_factory = simplebuffer::make;
}

int scheduler_st::get_buffer_num_items(edge e, flat_graph_sptr fg)
{
    size_t item_size = e.itemsize();

    // *2 because we're now only filling them 1/2 way in order to
    // increase the available parallelism when using the TPB scheduler.
    // (We're double buffering, where we used to single buffer)
    size_t nitems = s_fixed_buf_size * 2 / item_size;

    auto grblock = std::dynamic_pointer_cast<block>(e.src().node());
    if (grblock == nullptr) // might be a domain adapter, not a block
    {
        grblock = std::dynamic_pointer_cast<block>(e.dst().node());
    }

    // Make sure there are at least twice the output_multiple no. of items
    if (nitems < 2 * grblock->output_multiple()) // Note: this means output_multiple()
        nitems = 2 * grblock->output_multiple(); // can't be changed by block dynamically

    // // limit buffer size if indicated
    // if (grblock->max_output_buffer(port) > 0) {
    //     // std::cout << "constraining output items to " <<
    //     block->max_output_buffer(port)
    //     // << "\n";
    //     nitems = std::min((long)nitems, (long)grblock->max_output_buffer(port));
    //     nitems -= nitems % grblock->output_multiple();
    //     if (nitems < 1)
    //         throw std::runtime_error("problems allocating a buffer with the given
    //         max "
    //                                 "output buffer constraint!");
    // } else if (grblock->min_output_buffer(port) > 0) {
    //     nitems = std::max((long)nitems, (long)grblock->min_output_buffer(port));
    //     nitems -= nitems % grblock->output_multiple();
    //     if (nitems < 1)
    //         throw std::runtime_error("problems allocating a buffer with the given
    //         min "
    //                                 "output buffer constraint!");
    // }

    // FIXME: Downstream block connections get messed up by domain adapters
    //   Need to tag the blocks before they get partitioned
    //   and store the information in the edge objects
    //   also allow for different rates out of different ports

    // // If any downstream blocks are decimators and/or have a large output_multiple,
    // // ensure we have a buffer at least twice their decimation
    // // factor*output_multiple
    // auto blocks = fg->calc_downstream_blocks(grblock, port);

    // for (auto&  p : blocks) {
    //     // block_sptr dgrblock = cast_to_block_sptr(*p);
    //     // if (!dgrblock)
    //     //     throw std::runtime_error("allocate_buffer found non-gr::block");

    //     // double decimation = (1.0 / dgrblock->relative_rate());
    //     int multiple = p->output_multiple();
    //     nitems =
    //         std::max(nitems, static_cast<int>(2 * (multiple)));
    //         // std::max(nitems, static_cast<int>(2 * (decimation * multiple)));
    // }

    return nitems;
}

void scheduler_st::initialize(flat_graph_sptr fg,
                              flowgraph_monitor_sptr fgmon,
                              block_scheduler_map block_sched_map)
{
    d_fg = fg;
    d_fgmon = fgmon;
    d_block_sched_map = block_sched_map;

    // if (fg->is_flat())  // flatten

    buffer_factory_function buf_factory = _default_buf_factory;
    std::shared_ptr<buffer_properties> buf_props = _default_buf_properties;

    // not all edges may be used
    for (auto e : fg->edges()) {
        // every edge needs a buffer
        d_edge_catalog[e.identifier()] = e;

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


        auto src_da_cast = std::dynamic_pointer_cast<domain_adapter>(e.src().node());
        auto dst_da_cast = std::dynamic_pointer_cast<domain_adapter>(e.dst().node());

        if (src_da_cast != nullptr) {
            if (src_da_cast->buffer_location() == buffer_location_t::LOCAL) {
                buffer_sptr buf;

                if (e.has_custom_buffer()) {
                    buf = e.buffer_factory()(num_items, e.itemsize(), e.buf_properties());
                } else {
                    buf = buf_factory(num_items, e.itemsize(), buf_props);
                }

                src_da_cast->set_buffer(buf);
                auto tmp = std::dynamic_pointer_cast<buffer>(src_da_cast);
                d_edge_buffers[e.identifier()] = tmp;
                gr_log_info(_logger, "Edge: {}, Buf: {}", e.identifier(), buf->type());
            } else {
                d_edge_buffers[e.identifier()] =
                    std::dynamic_pointer_cast<buffer>(src_da_cast);
                gr_log_info(_logger, "Edge: {}, Buf: SRC_DA", e.identifier());
            }
        } else if (dst_da_cast != nullptr) {
            if (dst_da_cast->buffer_location() == buffer_location_t::LOCAL) {
                buffer_sptr buf;

                if (e.has_custom_buffer()) {
                    buf = e.buffer_factory()(num_items, e.itemsize(), e.buf_properties());
                } else {
                    buf = buf_factory(num_items, e.itemsize(), buf_props);
                }

                dst_da_cast->set_buffer(buf);
                auto tmp = std::dynamic_pointer_cast<buffer>(dst_da_cast);
                d_edge_buffers[e.identifier()] = tmp;
                gr_log_info(_logger, "Edge: {}, Buf: {}", e.identifier(), buf->type());
            } else {
                d_edge_buffers[e.identifier()] =
                    std::dynamic_pointer_cast<buffer>(dst_da_cast);
                gr_log_info(_logger, "Edge: {}, Buf: DST_DA", e.identifier());
            }

        }
        // If there are no domain adapter involved, then simply give this edge a
        // buffer
        else {
            buffer_sptr buf;
            if (e.has_custom_buffer()) {
                buf = e.buffer_factory()(num_items, e.itemsize(), e.buf_properties());
            } else {
                buf = buf_factory(num_items, e.itemsize(), buf_props);
            }

            d_edge_buffers[e.identifier()] = buf;
            gr_log_info(_logger, "Edge: {}, Buf: {}", e.identifier(), buf->type());
        }
    }

    for (auto& b : fg->calc_used_blocks()) {
        b->set_scheduler(base());
        d_blocks.push_back(b);
        d_block_id_to_block_map[b->id()] = b;

        port_vector_t input_ports = b->input_stream_ports();
        port_vector_t output_ports = b->output_stream_ports();

        for (auto p : input_ports) {
            d_block_buffers[p] = std::vector<buffer_sptr>{};
            edge_vector_t ed = d_fg->find_edge(p);
            for (auto e : ed)
                d_block_buffers[p].push_back(d_edge_buffers[e.identifier()]);
        }

        for (auto p : output_ports) {
            d_block_buffers[p] = std::vector<buffer_sptr>{};
            edge_vector_t ed = d_fg->find_edge(p);
            for (auto e : ed)
                d_block_buffers[p].push_back(d_edge_buffers[e.identifier()]);
        }
    }
}
void scheduler_st::start()
{
    for (auto& b : d_blocks) {
        b->start();
    }
    d_thread = std::thread(thread_body, this);

    push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL));
}
void scheduler_st::stop()
{
    d_thread_stopped = true;
    d_thread.join();
    for (auto& b : d_blocks) {
        b->stop();
    }
}
void scheduler_st::wait()
{
    d_thread.join();
    for (auto& b : d_blocks) {
        b->done();
    }
}
void scheduler_st::run()
{
    start();
    wait();
}

std::map<nodeid_t, scheduler_iteration_status>
scheduler_st::run_one_iteration(std::vector<block_sptr> blocks)
{
    std::map<nodeid_t, scheduler_iteration_status> per_block_status;

    // If no blocks are specified for the iteration, then run over all the blocks in
    // the default ordering
    if (blocks.empty()) {
        blocks = d_blocks;
    }

    for (auto const& b : blocks) { // TODO - order the blocks

        std::vector<block_work_input> work_input;   //(num_input_ports);
        std::vector<block_work_output> work_output; //(num_output_ports);

        // for each input port of the block
        bool ready = true;
        for (auto p : b->input_stream_ports()) {
            auto p_buf = d_block_buffers[p][0];


            buffer_info_t read_info;
            ready = p_buf->read_info(read_info);
            gr_log_debug(
                _debug_logger, "read_info {} - {}", b->alias(), read_info.n_items);

            if (!ready)
                break;

            if (read_info.n_items < s_min_items_to_process) {
                ready = false;
                break;
            }

            auto tags = p_buf->get_tags(read_info.n_items);
            work_input.push_back(block_work_input(
                read_info.n_items, read_info.total_items, read_info.ptr, tags));
        }

        if (!ready) {
            per_block_status[b->id()] = scheduler_iteration_status::BLKD_IN;
            continue;
        }

        // for each output port of the block
        for (auto p : b->output_stream_ports()) {

            // When a block has multiple output buffers, it adds the restriction
            // that the work call can only produce the minimum available across
            // the buffers.

            size_t max_output_buffer = std::numeric_limits<int>::max();

            void* write_ptr = nullptr;
            uint64_t nitems_written;
            for (auto p_buf : d_block_buffers[p]) {
                buffer_info_t write_info;
                ready = p_buf->write_info(write_info);
                gr_log_debug(_debug_logger,
                             "write_info {} - {} @ {} {}",
                             b->alias(),
                             write_info.n_items,
                             write_info.ptr,
                             write_info.item_size);
                if (!ready)
                    break;

                size_t tmp_buf_size = write_info.n_items;
                if (tmp_buf_size < s_min_buf_items) {
                    ready = false;
                    break;
                }
                // while (tmp_buf_size > p_buf->) {
                //     tmp_buf_size >>= 1;
                //     if (tmp_buf_size < s_min_buf_items)
                //     {
                //         ready = false;
                //         break;
                //     }
                // }
                // if (!ready) { break; }

                if (tmp_buf_size < max_output_buffer - 1)
                    max_output_buffer = tmp_buf_size;

                // store the first buffer
                if (!write_ptr) {
                    write_ptr = write_info.ptr;
                    nitems_written = write_info.total_items;
                }
            }

            max_output_buffer = std::min(max_output_buffer, s_max_buf_items);
            std::vector<tag_t> tags; // needs to be associated with edge buffers

            if (b->output_multiple_set()) {
                // quantize to the output multiple
                if (max_output_buffer < b->output_multiple()) {
                    max_output_buffer = b->output_multiple();
                }

                max_output_buffer =
                    b->output_multiple() * (max_output_buffer / b->output_multiple());
                if (max_output_buffer == 0) {
                    ready = false;
                    break;
                }
            }

            work_output.push_back(
                block_work_output(max_output_buffer, nitems_written, write_ptr, tags));
        }

        if (!ready) {
            per_block_status[b->id()] = scheduler_iteration_status::BLKD_OUT;
            continue;
        }

        if (ready) {
            work_return_code_t ret;
            while (true) {

                if (work_output.size() > 0)
                    gr_log_debug(_debug_logger,
                                 "do_work for {}, {}",
                                 b->alias(),
                                 work_output[0].n_items);
                else
                    gr_log_debug(_debug_logger, "do_work for {}", b->alias());


                ret = b->do_work(work_input, work_output);
                gr_log_debug(_debug_logger, "do_work returned {}", ret);


                if (ret == work_return_code_t::WORK_DONE) {
                    per_block_status[b->id()] = scheduler_iteration_status::DONE;
                    break;
                } else if (ret == work_return_code_t::WORK_OK) {
                    per_block_status[b->id()] = scheduler_iteration_status::READY;
                    break;
                } else if (ret == work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS) {
                    work_output[0].n_items >>= 1;
                    if (work_output[0].n_items < 4) // min block size
                    {
                        break;
                    }
                }
            }
            // TODO - handle READY_NO_OUTPUT

            if (ret == work_return_code_t::WORK_OK ||
                ret == work_return_code_t::WORK_DONE) {


                int input_port_index = 0;
                for (auto p : b->input_stream_ports()) {
                    auto p_buf = d_block_buffers[p][0]; // only one buffer per input port

                    // Pass the tags according to TPP
                    if (b->tag_propagation_policy() ==
                        tag_propagation_policy_t::TPP_ALL_TO_ALL) {
                        int output_port_index = 0;
                        for (auto op : b->output_stream_ports()) {
                            for (auto p_out_buf : d_block_buffers[op]) {
                                p_out_buf->add_tags(
                                    work_output[output_port_index].n_produced,
                                    work_input[input_port_index].tags);
                            }
                            output_port_index++;
                        }
                    } else if (b->tag_propagation_policy() ==
                               tag_propagation_policy_t::TPP_ONE_TO_ONE) {
                        int output_port_index = 0;
                        for (auto op : b->output_stream_ports()) {
                            if (output_port_index == input_port_index) {
                                for (auto p_out_buf : d_block_buffers[op]) {
                                    p_out_buf->add_tags(
                                        work_output[output_port_index].n_produced,
                                        work_input[input_port_index].tags);
                                }
                            }
                            output_port_index++;
                        }
                    }

                    gr_log_debug(_debug_logger,
                                 "post_read {} - {}",
                                 b->alias(),
                                 work_input[input_port_index].n_consumed);

                    p_buf->post_read(work_input[input_port_index].n_consumed);
                    gr_log_debug(_debug_logger, ".");
                    input_port_index++;
                }

                int output_port_index = 0;
                for (auto p : b->output_stream_ports()) {
                    int j = 0;
                    for (auto p_buf : d_block_buffers[p]) {
                        if (j > 0) {
                            gr_log_debug(_debug_logger,
                                         "copy_items {} - {}",
                                         b->alias(),
                                         work_output[output_port_index].n_produced);
                            p_buf->copy_items(d_block_buffers[p][0],
                                              work_output[output_port_index].n_produced);
                            gr_log_debug(_debug_logger, ".");
                        }
                        j++;
                    }
                    for (auto p_buf : d_block_buffers[p]) {
                        // Add the tags that were collected in the work() call
                        if (!work_output[output_port_index].tags.empty()) {
                            p_buf->add_tags(work_output[output_port_index].n_produced,
                                            work_output[output_port_index].tags);
                        }

                        gr_log_debug(_debug_logger,
                                     "post_write {} - {}",
                                     b->alias(),
                                     work_output[output_port_index].n_produced);
                        p_buf->post_write(work_output[output_port_index].n_produced);
                        gr_log_debug(_debug_logger, ".");
                    }
                    output_port_index++;
                }
            }
        }
    }

    return per_block_status;
}

void scheduler_st::notify_self()
{
    gr_log_debug(_debug_logger, "notify_self");
    push_message(std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_ALL));
}

std::vector<scheduler_sptr> scheduler_st::get_neighbors_upstream(nodeid_t blkid)
{
    std::vector<scheduler_sptr> ret;
    // Find whether this block has an upstream neighbor
    auto search = d_block_sched_map.find(blkid);
    if (search != d_block_sched_map.end()) {
        // Entry in the map exists, is the ptr real?
        if (search->second.upstream_neighbor_sched) {
            // Notify upstream neighbor
            ret.push_back(search->second.upstream_neighbor_sched);
        }
    }

    return ret;
}

std::vector<scheduler_sptr> scheduler_st::get_neighbors_downstream(nodeid_t blkid)
{
    std::vector<scheduler_sptr> ret;
    // Find whether this block has any downstream neighbors
    auto search = d_block_sched_map.find(blkid);
    if (search != d_block_sched_map.end()) {
        // Entry in the map exists, are there any entries
        if (!search->second.downstream_neighbor_scheds.empty()) {

            for (auto sched : search->second.downstream_neighbor_scheds) {
                ret.push_back(sched);
            }
        }
    }

    return ret;
}

std::vector<scheduler_sptr> scheduler_st::get_neighbors(nodeid_t blkid)
{
    std::vector ret = get_neighbors_upstream(blkid);
    std::vector ds = get_neighbors_downstream(blkid);

    ret.insert(ret.end(), ds.begin(), ds.end());
    return ret;
}

void scheduler_st::notify_upstream(scheduler_sptr upstream_sched)
{
    gr_log_debug(_debug_logger, "notify_upstream");

    upstream_sched->push_message(
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_OUTPUT));
}
void scheduler_st::notify_downstream(scheduler_sptr downstream_sched)
{
    gr_log_debug(_debug_logger, "notify_downstream");
    downstream_sched->push_message(
        std::make_shared<scheduler_action>(scheduler_action_t::NOTIFY_INPUT));
}

void scheduler_st::handle_parameter_query(std::shared_ptr<param_query_action> item)
{
    auto b = d_block_id_to_block_map[item->block_id()];

    gr_log_debug(
        _debug_logger, "handle parameter query {} - {}", item->block_id(), b->alias());

    b->on_parameter_query(item->param_action());

    if (item->cb_fcn() != nullptr)
        item->cb_fcn()(item->param_action());
}

void scheduler_st::handle_parameter_change(std::shared_ptr<param_change_action> item)
{
    auto b = d_block_id_to_block_map[item->block_id()];

    gr_log_debug(
        _debug_logger, "handle parameter change {} - {}", item->block_id(), b->alias());

    b->on_parameter_change(item->param_action());

    if (item->cb_fcn() != nullptr)
        item->cb_fcn()(item->param_action());
}


void scheduler_st::handle_work_notification()
{
    auto s = run_one_iteration();
    std::string dbg_work_done;
    for (auto elem : s) {
        dbg_work_done += "[" + std::to_string(elem.first) + "," +
                         std::to_string((int)elem.second) + "]" + ",";
    }
    gr_log_debug(_debug_logger, dbg_work_done);

    // Based on state of the run_one_iteration, do things
    // If any of the blocks are done, notify the flowgraph monitor
    for (auto elem : s) {
        if (elem.second == scheduler_iteration_status::DONE) {
            gr_log_debug(
                _debug_logger, "Signalling DONE to FGM from block {}", elem.first);
            d_fgmon->push_message(
                fg_monitor_message(fg_monitor_message_t::DONE, id(), elem.first));
            break; // only notify the fgmon once
        }
    }

    bool notify_self_ = false;

    std::vector<scheduler_sptr> sched_to_notify_upstream, sched_to_notify_downstream;

    for (auto elem : s) {

        if (elem.second == scheduler_iteration_status::READY) {
            // top->notify_neighbors(elem.first);
            auto tmp_us = get_neighbors_upstream(elem.first);
            auto tmp_ds = get_neighbors_downstream(elem.first);

            if (!tmp_us.empty()) {
                sched_to_notify_upstream.insert(
                    sched_to_notify_upstream.end(), tmp_us.begin(), tmp_us.end());
            }
            if (!tmp_ds.empty()) {
                sched_to_notify_downstream.insert(
                    sched_to_notify_downstream.end(), tmp_ds.begin(), tmp_ds.end());
            }
            notify_self_ = true;
        }
    }

    if (notify_self_) {
        gr_log_debug(_debug_logger, "notifying self");
        notify_self();
    }

    if (!sched_to_notify_upstream.empty()) {
        // Reduce to the unique schedulers to notify
        std::sort(sched_to_notify_upstream.begin(), sched_to_notify_upstream.end());
        auto last =
            std::unique(sched_to_notify_upstream.begin(), sched_to_notify_upstream.end());
        sched_to_notify_upstream.erase(last, sched_to_notify_upstream.end());
        for (auto sched : sched_to_notify_upstream) {
            notify_upstream(sched);
        }
    }

    if (!sched_to_notify_downstream.empty()) {
        // Reduce to the unique schedulers to notify
        std::sort(sched_to_notify_downstream.begin(), sched_to_notify_downstream.end());
        auto last = std::unique(sched_to_notify_downstream.begin(),
                                sched_to_notify_downstream.end());
        sched_to_notify_downstream.erase(last, sched_to_notify_downstream.end());
        for (auto sched : sched_to_notify_downstream) {
            notify_downstream(sched);
        }
    }
}

void scheduler_st::thread_body(scheduler_st* top)
{
    top->set_state(scheduler_state::WORKING);
    gr_log_info(top->_logger, "starting thread");
    while (!top->d_thread_stopped) {

        // try to pop messages off the queue
        scheduler_message_sptr msg;
        if (top->pop_message(msg)) // this blocks
        {
            switch (msg->type()) {
            case scheduler_message_t::SCHEDULER_ACTION: {
                // Notification that work needs to be done
                // either from runtime or upstream or downstream or from self

                auto action = std::static_pointer_cast<scheduler_action>(msg);
                switch (action->action()) {
                case scheduler_action_t::DONE:
                    // fgmon says that we need to be done, wrap it up
                    // each scheduler could handle this in a different way
                    gr_log_debug(top->_debug_logger,
                                 "fgm signaled DONE, pushing flushed");
                    top->d_fgmon->push_message(
                        fg_monitor_message(fg_monitor_message_t::FLUSHED, top->id()));
                    break;
                case scheduler_action_t::EXIT:
                    gr_log_debug(top->_debug_logger, "fgm signaled EXIT, exiting thread");
                    // fgmon says that we need to be done, wrap it up
                    // each scheduler could handle this in a different way
                    top->d_thread_stopped = true;
                    break;
                case scheduler_action_t::NOTIFY_OUTPUT:
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_OUTPUT from {}", msg->blkid());
                    top->handle_work_notification();
                    break;
                case scheduler_action_t::NOTIFY_INPUT:
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_INPUT from {}", msg->blkid());
                    top->handle_work_notification();
                    break;
                case scheduler_action_t::NOTIFY_ALL: {
                    gr_log_debug(
                        top->_debug_logger, "got NOTIFY_ALL from {}", msg->blkid());
                    top->handle_work_notification();
                    break;
                }
                default:
                    break;
                    break;
                }
                break;
            }
            case scheduler_message_t::PARAMETER_QUERY: {
                // Query the state of a parameter on a block
                top->handle_parameter_query(
                    std::static_pointer_cast<param_query_action>(msg));
            } break;
            case scheduler_message_t::PARAMETER_CHANGE: {
                // Query the state of a parameter on a block
                top->handle_parameter_change(
                    std::static_pointer_cast<param_change_action>(msg));
            } break;
            default:
                break;
            }
        }
    }
}


} // namespace schedulers
} // namespace gr
