#include "graph_executor.hpp"

namespace gr {
namespace schedulers {

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
            per_block_status[b->id()] = executor_iteration_status::BLKD_IN;
            continue;
        }

        // for each output port of the block
        for (auto p : b->output_stream_ports()) {

            // When a block has multiple output buffers, it adds the restriction
            // that the work call can only produce the minimum available across
            // the buffers.

            size_t max_output_buffer = std::numeric_limits<int>::max();

            void* write_ptr = nullptr;
            uint64_t nitems_written = 0;
            for (auto p_buf : _bufman->get_output_buffers(p)) {
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

                if (tmp_buf_size < max_output_buffer)
                    max_output_buffer = tmp_buf_size;

                // store the first buffer
                if (!write_ptr) {
                    write_ptr = write_info.ptr;
                    nitems_written = write_info.total_items;
                }
            }

            if (!ready)
                break;


            std::vector<tag_t> tags; // needs to be associated with edge buffers

            work_output.push_back(
                block_work_output(max_output_buffer, nitems_written, write_ptr, tags));
        }

        if (!ready) {
            per_block_status[b->id()] = executor_iteration_status::BLKD_OUT;
            continue;
        }

        if (ready) {
            work_return_code_t ret;
            while (true) {

                if (work_output.size() > 0) {
                    gr_log_debug(_debug_logger,
                                 "do_work for {}, {}",
                                 b->alias(),
                                 work_output[0].n_items);
                } else {
                    gr_log_debug(_debug_logger, "do_work for {}", b->alias());
                }


                ret = b->do_work(work_input, work_output);
                gr_log_debug(_debug_logger, "do_work returned {}", ret);
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
                }
            }
            // TODO - handle READY_NO_OUTPUT

            if (ret == work_return_code_t::WORK_OK ||
                ret == work_return_code_t::WORK_DONE) {


                int input_port_index = 0;
                for (auto p : b->input_stream_ports()) {
                    auto p_buf = _bufman->get_input_buffer(p);

                    // Pass the tags according to TPP
                    if (b->tag_propagation_policy() ==
                        tag_propagation_policy_t::TPP_ALL_TO_ALL) {
                        int output_port_index = 0;
                        for (auto op : b->output_stream_ports()) {
                            for (auto p_out_buf : _bufman->get_output_buffers(op)) {
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
                                for (auto p_out_buf : _bufman->get_output_buffers(op)) {
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
                    for (auto p_buf : _bufman->get_output_buffers(p)) {
                        if (j > 0) {
                            gr_log_debug(_debug_logger,
                                         "copy_items {} - {}",
                                         b->alias(),
                                         work_output[output_port_index].n_produced);
                            p_buf->copy_items(_bufman->get_input_buffer(p),
                                              work_output[output_port_index].n_produced);
                            gr_log_debug(_debug_logger, ".");
                        }
                        j++;
                    }
                    for (auto p_buf : _bufman->get_output_buffers(p)) {
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

} // namespace schedulers
} // namespace gr
