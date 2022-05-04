#include "graph_executor.h"

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

    // This is only for streaming blocks
    for (auto const& b : blocks) { // TODO - order the blocks
        if (b->work_mode() != block_work_mode_t::DEFAULT) {
            continue;
        }

        std::vector<block_work_input_sptr> work_input;   //(num_input_ports);
        std::vector<block_work_output_sptr> work_output; //(num_output_ports);

        // If a block is a message port only block, it will raise the finished() flag
        // to indicate that the rest of the flowgraph should clean up
        if (b->finished()) {
            per_block_status[b->id()] = executor_iteration_status::DONE;
            d_debug_logger->debug("pbs[{}]: {}", b->id(), per_block_status[b->id()]);
            continue;
        }

        auto input_stream_ports = b->input_stream_ports();
        auto output_stream_ports = b->output_stream_ports();

        if (input_stream_ports.empty() && output_stream_ports.empty()) {
            // There is no streaming work to do for this block
            per_block_status[b->id()] = executor_iteration_status::MSG_ONLY;
            continue;
        }

        // for each input port of the block
        bool ready = true;
        for (auto p : input_stream_ports) {
            auto p_buf = p->buffer_reader();
            auto max_read = p_buf->max_buffer_read();
            auto min_read = p_buf->min_buffer_read();

            buffer_info_t read_info;
            ready = p_buf->read_info(read_info);
            d_debug_logger->debug(
                         "read_info {} - {} - {}",
                         b->alias(),
                         read_info.n_items,
                         read_info.item_size);

            if (!ready)
                break;

            if (read_info.n_items < s_min_items_to_process ||
                (min_read > 0 && read_info.n_items < (int)min_read)) {

                p_buf->input_blocked_callback(s_min_items_to_process);

                ready = false;
                break;
            }

            if (max_read > 0 && read_info.n_items > (int)max_read) {
                read_info.n_items = max_read;
            }


            auto tags = p_buf->get_tags(read_info.n_items);
            work_input.push_back(
                std::make_shared<block_work_input>(read_info.n_items, p_buf));
        }

        if (!ready) {
            per_block_status[b->id()] = executor_iteration_status::BLKD_IN;
            continue;
        }

        // for each output port of the block
        for (auto p : output_stream_ports) {

            // When a block has multiple output buffers, it adds the restriction
            // that the work call can only produce the minimum available across
            // the buffers.

            size_t max_output_buffer = std::numeric_limits<int>::max();

            auto p_buf = p->buffer();
            auto max_fill = p_buf->max_buffer_fill();
            auto min_fill = p_buf->min_buffer_fill();

            buffer_info_t write_info;
            ready = p_buf->write_info(write_info);
            d_debug_logger->debug(
                         "write_info {} - {} @ {} {}",
                         b->alias(),
                         write_info.n_items,
                         write_info.ptr,
                         write_info.item_size);

            size_t tmp_buf_size = write_info.n_items;
            if (tmp_buf_size < s_min_buf_items ||
                (min_fill > 0 && tmp_buf_size < min_fill)) {
                ready = false;
                p_buf->output_blocked_callback(false);
                break;
            }

            if (tmp_buf_size < max_output_buffer)
                max_output_buffer = tmp_buf_size;

            if (max_fill > 0 && max_output_buffer > max_fill) {
                max_output_buffer = max_fill;
            }

            if (b->output_multiple_set()) {
                max_output_buffer = round_down(max_output_buffer, b->output_multiple());
            }

            if (max_output_buffer <= 0) {
                ready = false;
            }

            if (!ready)
                break;


            std::vector<tag_t> tags; // needs to be associated with edge buffers

            work_output.push_back(
                std::make_shared<block_work_output>(max_output_buffer, p_buf));
        }

        if (!ready) {
            per_block_status[b->id()] = executor_iteration_status::BLKD_OUT;
            continue;
        }

        if (ready) {
            work_return_code_t ret;
            while (true) {

                if (!work_output.empty()) {
                    d_debug_logger->debug(
                                 "do_work (output) for {}, {}",
                                 b->alias(),
                                 work_output[0]->n_items);
                }
                else if (!work_input.empty())
                {
                    d_debug_logger->debug(
                                 "do_work (input) for {}, {}",
                                 b->alias(),
                                 work_input[0]->n_items);
                }
                else {
                    d_debug_logger->debug("do_work for {}", b->alias());
                }


                ret = b->do_work(work_input, work_output);
                d_debug_logger->debug("do_work returned {}", ret);
                // ret = work_return_code_t::WORK_OK;

                if (ret == work_return_code_t::WORK_DONE) {
                    per_block_status[b->id()] = executor_iteration_status::DONE;
                    d_debug_logger->debug("pbs[{}]: {}", b->id(), per_block_status[b->id()]);
                    break;
                }
                else if (ret == work_return_code_t::WORK_OK) {
                    per_block_status[b->id()] = executor_iteration_status::READY;
                    d_debug_logger->debug("pbs[{}]: {}", b->id(), per_block_status[b->id()]);

                    // If a source block, and no outputs were produced, mark as BLKD_IN
                    if (work_input.empty() && !work_output.empty()) {
                        size_t max_output = 0;
                        for (auto& w : work_output) {
                            max_output = std::max(w->n_produced, max_output);
                        }
                        if (max_output <= 0) {
                            per_block_status[b->id()] =
                                executor_iteration_status::BLKD_IN;
                            d_debug_logger->debug(
                                         "pbs[{}]: {}",
                                         b->id(),
                                         per_block_status[b->id()]);
                        }
                    }


                    break;
                }
                else if (ret == work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS) {
                    if (b->output_multiple_set()) {
                        work_output[0]->n_items -= b->output_multiple();
                    }
                    else {
                        work_output[0]->n_items >>= 1;
                    }
                    if (work_output[0]->n_items < b->output_multiple()) // min block size
                    {
                        per_block_status[b->id()] = executor_iteration_status::BLKD_IN;
                        d_debug_logger->debug(
                                     "pbs[{}]: {}",
                                     b->id(),
                                     per_block_status[b->id()]);
                        // call the input blocked callback
                        break;
                    }
                }
                else if (ret == work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS) {
                    per_block_status[b->id()] = executor_iteration_status::BLKD_OUT;
                    d_debug_logger->debug("pbs[{}]: {}", b->id(), per_block_status[b->id()]);
                    // call the output blocked callback
                    break;
                }
            }
            // TODO - handle READY_NO_OUTPUT

            if (ret == work_return_code_t::WORK_OK ||
                ret == work_return_code_t::WORK_DONE) {


                int input_port_index = 0;
                for (auto p : b->input_stream_ports()) {
                    auto p_buf = p->buffer_reader();

                    if (!p_buf->tags().empty()) {
                        // Pass the tags according to TPP
                        if (b->tag_propagation_policy() ==
                            tag_propagation_policy_t::TPP_ALL_TO_ALL) {
                            int output_port_index = 0;
                            for (auto op : b->output_stream_ports()) {
                                auto p_out_buf = op->buffer();
                                p_out_buf->propagate_tags(
                                    p_buf, work_input[input_port_index]->n_consumed);

                                output_port_index++;
                            }
                        }
                        else if (b->tag_propagation_policy() ==
                                 tag_propagation_policy_t::TPP_ONE_TO_ONE) {
                            int output_port_index = 0;
                            for (auto op : b->output_stream_ports()) {
                                if (output_port_index == input_port_index) {
                                    auto p_out_buf = op->buffer();
                                    p_out_buf->propagate_tags(
                                        p_buf, work_input[input_port_index]->n_consumed);
                                }
                                output_port_index++;
                            }
                        }
                    }

                    d_debug_logger->debug(
                                 "post_read {} - {}",
                                 b->alias(),
                                 work_input[input_port_index]->n_consumed);

                    p_buf->post_read(work_input[input_port_index]->n_consumed);
                    p->notify_connected_ports(std::make_shared<scheduler_action>(
                        scheduler_action_t::NOTIFY_OUTPUT));

                    input_port_index++;
                }

                int output_port_index = 0;
                for (auto p : b->output_stream_ports()) {
                    auto p_buf = p->buffer();

                    d_debug_logger->debug(
                                 "post_write {} - {}",
                                 b->alias(),
                                 work_output[output_port_index]->n_produced);
                    p_buf->post_write(work_output[output_port_index]->n_produced);

                    p->notify_connected_ports(std::make_shared<scheduler_action>(
                        scheduler_action_t::NOTIFY_INPUT));

                    output_port_index++;

                    p_buf->prune_tags();
                }
            }
        }
    }


    return per_block_status;
}

} // namespace schedulers
} // namespace gr
