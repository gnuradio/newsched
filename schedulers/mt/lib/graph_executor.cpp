#include "graph_executor.hpp"

namespace gr {
namespace schedulers {

inline static unsigned int round_down(unsigned int n, unsigned int multiple)
{
    return (n / multiple) * multiple;
}

std::map<nodeid_t, executor_iteration_status>
graph_executor::run_one_iteration(std::vector<block_sptr> blocks)
{
    std::map<nodeid_t, executor_iteration_status> per_block_status;

    // If no blocks are specified for the iteration, then run over all the blocks
    // in the default ordering
    if (blocks.empty()) {
        blocks = d_blocks;
    }

    for (auto const& b : blocks) { // TODO - order the blocks

        std::vector<block_work_input> work_input;   //(num_input_ports);
        std::vector<block_work_output> work_output; //(num_output_ports);

        // for each input port of the block
        bool ready = true;
        for (auto p : b->input_stream_ports()) {
            auto p_buf = _bufman->get_input_buffer(p);

            buffer_info_t read_info;
            ready = p_buf->read_info(read_info);
            GR_LOG_DEBUG(
                _debug_logger, "read_info {} - {}", b->alias(), read_info.n_items);

            if (!ready)
                break;

            if (read_info.n_items < s_min_items_to_process) {
                ready = false;
                break;
            }

            auto tags = p_buf->get_tags(read_info.n_items);
            work_input.push_back(block_work_input(read_info.n_items, p_buf));
        }

        if (!ready) {
            per_block_status[b->id()] = executor_iteration_status::BLKD_IN;
            continue;
        }

        // for each output port of the block
        for (auto p : b->output_stream_ports()) {

            // When a block has multiple output buffers, it adds the restriction
            // that the work call can only produce the minimum available across
            // the buffers.

            size_t max_output_buffer = std::numeric_limits<int>::max();

            buffer_sptr p_buf = nullptr;
            uint64_t nitems_written = 0;
            for (auto each_buf : _bufman->get_output_buffers(p)) {
                buffer_info_t write_info;
                ready = each_buf->write_info(write_info);
                GR_LOG_DEBUG(_debug_logger,
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

                


                if (tmp_buf_size < max_output_buffer)
                    max_output_buffer = tmp_buf_size;

                if (b->output_multiple_set())
                    max_output_buffer = round_down(max_output_buffer, b->output_multiple());

                // store the first buffer
                if (!p_buf) {
                    p_buf = each_buf;
                }
            }

            if (!ready)
                break;


            std::vector<tag_t> tags; // needs to be associated with edge buffers

            work_output.push_back(block_work_output(max_output_buffer, p_buf));
        }

        if (!ready) {
            per_block_status[b->id()] = executor_iteration_status::BLKD_OUT;
            continue;
        }

        if (ready) {
            work_return_code_t ret;
            while (true) {

                if (work_output.size() > 0) {
                    GR_LOG_DEBUG(_debug_logger,
                                 "do_work for {}, {}",
                                 b->alias(),
                                 work_output[0].n_items);
                } else {
                    GR_LOG_DEBUG(_debug_logger, "do_work for {}", b->alias());
                }


                ret = b->do_work(work_input, work_output);
                GR_LOG_DEBUG(_debug_logger, "do_work returned {}", ret);
                // ret = work_return_code_t::WORK_OK;

                if (ret == work_return_code_t::WORK_DONE) {
                    per_block_status[b->id()] = executor_iteration_status::DONE;
                    break;
                } else if (ret == work_return_code_t::WORK_OK) {
                    per_block_status[b->id()] = executor_iteration_status::READY;
                    break;
                } else if (ret == work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS) {
                    work_output[0].n_items >>= 1;
                    if (work_output[0].n_items < 4) // min block size
                    {
                        break;
                    }
                } else if (ret == work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS) {
                    per_block_status[b->id()] = executor_iteration_status::BLKD_OUT;
                    break;
                }
            }
            // TODO - handle READY_NO_OUTPUT

            if (ret == work_return_code_t::WORK_OK ||
                ret == work_return_code_t::WORK_DONE) {


                int input_port_index = 0;
                for (auto p : b->input_stream_ports()) {
                    auto p_buf = _bufman->get_input_buffer(p);

                    if (!p_buf->tags().empty()) {
                        // Pass the tags according to TPP
                        if (b->tag_propagation_policy() ==
                            tag_propagation_policy_t::TPP_ALL_TO_ALL) {
                            int output_port_index = 0;
                            for (auto op : b->output_stream_ports()) {
                                for (auto p_out_buf : _bufman->get_output_buffers(op)) {
                                    p_out_buf->propagate_tags(
                                        p_buf, work_input[input_port_index].n_consumed);
                                }
                                output_port_index++;
                            }
                        } else if (b->tag_propagation_policy() ==
                                   tag_propagation_policy_t::TPP_ONE_TO_ONE) {
                            int output_port_index = 0;
                            for (auto op : b->output_stream_ports()) {
                                if (output_port_index == input_port_index) {
                                    for (auto p_out_buf :
                                         _bufman->get_output_buffers(op)) {
                                        p_out_buf->propagate_tags(
                                            p_buf,
                                            work_input[input_port_index].n_consumed);
                                    }
                                }
                                output_port_index++;
                            }
                        }
                    }

                    GR_LOG_DEBUG(_debug_logger,
                                 "post_read {} - {}",
                                 b->alias(),
                                 work_input[input_port_index].n_consumed);

                    p_buf->prune_tags(work_input[input_port_index].n_consumed);
                    p_buf->post_read(work_input[input_port_index].n_consumed);
                    
                    p->notify_connected_ports(std::make_shared<scheduler_action>(
                        scheduler_action_t::NOTIFY_OUTPUT));

                    input_port_index++;
                }

                int output_port_index = 0;
                for (auto p : b->output_stream_ports()) {
                    int j = 0;
                    for (auto p_buf : _bufman->get_output_buffers(p)) {
                        if (j > 0) {
                            GR_LOG_DEBUG(_debug_logger,
                                         "copy_items {} - {}",
                                         b->alias(),
                                         work_output[output_port_index].n_produced);
                            p_buf->copy_items(_bufman->get_input_buffer(p),
                                              work_output[output_port_index].n_produced);
                        }
                        j++;
                    }
                    for (auto p_buf : _bufman->get_output_buffers(p)) {

                        GR_LOG_DEBUG(_debug_logger,
                                     "post_write {} - {}",
                                     b->alias(),
                                     work_output[output_port_index].n_produced);
                        p_buf->post_write(work_output[output_port_index].n_produced);
                    }

                    p->notify_connected_ports(std::make_shared<scheduler_action>(
                        scheduler_action_t::NOTIFY_INPUT));

                    output_port_index++;
                }
            }
        }
    }

    return per_block_status;
}

} // namespace schedulers
} // namespace gr
