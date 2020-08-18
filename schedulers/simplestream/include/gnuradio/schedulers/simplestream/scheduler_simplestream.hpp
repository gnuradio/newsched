//#include <gnuradio/scheduler.hpp>
#include <gnuradio/scheduler.hpp>
// #include <boost/circular_buffer.hpp>
#include <gnuradio/domain_adapter.hpp>
#include <gnuradio/simplebuffer.hpp>
#include <thread> // std::thread
namespace gr {

namespace schedulers {

// This is a terrible scheduler just to pass data through the blocks
// and demonstrate the pluggable scheduler concept

class scheduler_simplestream : public scheduler
{
private:
    std::string _name;

public:
    scheduler_sync* sched_sync;
    const int s_fixed_buf_size;
    static const int s_min_items_to_process = 1;
    const int s_max_buf_items; // = s_fixed_buf_size / 2;

    typedef std::shared_ptr<scheduler_simplestream> sptr;

    scheduler_simplestream(const std::string name = "simplestream",
                           const unsigned int fixed_buf_size = 8192)
        : scheduler(name),
          s_fixed_buf_size(fixed_buf_size),
          s_max_buf_items(fixed_buf_size / 2)
    {
    }
    ~scheduler_simplestream(){

    };

    void initialize(flat_graph_sptr fg)
    {
        d_fg = fg;

        // if (fg->is_flat())  // flatten

        // not all edges may be used
        for (auto e : fg->edges()) {
            // every edge needs a buffer
            d_edge_catalog[e.identifier()] = e;

            auto src_da_cast = std::dynamic_pointer_cast<domain_adapter>(e.src().node());
            auto dst_da_cast = std::dynamic_pointer_cast<domain_adapter>(e.dst().node());

            if (src_da_cast != nullptr) {
                if (src_da_cast->buffer_location() == buffer_location_t::LOCAL) {
                    auto buf = simplebuffer::make(s_fixed_buf_size, e.itemsize());
                    src_da_cast->set_buffer(buf);
                    auto tmp = std::dynamic_pointer_cast<buffer>(src_da_cast);
                    d_edge_buffers[e.identifier()] = tmp;
                } else {
                    d_edge_buffers[e.identifier()] =
                        std::dynamic_pointer_cast<buffer>(src_da_cast);
                }
            } else if (dst_da_cast != nullptr) {
                if (dst_da_cast->buffer_location() == buffer_location_t::LOCAL) {
                    auto buf = simplebuffer::make(s_fixed_buf_size, e.itemsize());
                    dst_da_cast->set_buffer(buf);
                    auto tmp = std::dynamic_pointer_cast<buffer>(dst_da_cast);
                    d_edge_buffers[e.identifier()] = tmp;
                } else {
                    d_edge_buffers[e.identifier()] =
                        std::dynamic_pointer_cast<buffer>(dst_da_cast);
                }

            } else {
                d_edge_buffers[e.identifier()] =
                    simplebuffer::make(s_fixed_buf_size, e.itemsize());
            }

            d_edge_buffers[e.identifier()]->set_name(e.identifier());
        }

        for (auto& b : fg->calc_used_blocks()) {
            b->set_scheduler(base());
            d_blocks.push_back(b);

            port_vector_t input_ports = b->input_stream_ports();
            port_vector_t output_ports = b->output_stream_ports();

            unsigned int num_input_ports = input_ports.size();
            unsigned int num_output_ports = output_ports.size();

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

    void start(scheduler_sync* sync)
    {
        sched_sync = sync;
        for (auto& b : d_blocks) {
            b->start();
        }
        d_thread = std::thread(thread_body, this);
    }

    void stop()
    {
        d_thread_stopped = true;
        d_thread.join();
        for (auto& b : d_blocks) {
            b->stop();
        }
    }

    void wait()
    {
        d_thread.join();
        for (auto& b : d_blocks) {
            b->done();
        }
    }

    void run(scheduler_sync* sync)
    {
        start(sync);
        wait();
    }

private:
    flat_graph_sptr d_fg;
    std::vector<block_sptr> d_blocks;
    std::map<std::string, edge> d_edge_catalog;
    std::map<std::string, buffer_sptr> d_edge_buffers;
    // std::map<block_sptr, std::map<block::io, std::map<int, simplebuffer::sptr>>>
    //     d_block_buffers; // store the block buffers
    std::map<port_sptr, std::vector<buffer_sptr>> d_block_buffers;
    // std::map<port_sptr, simplebuffer::sptr> d_block_buffers; // store the block buffers
    std::thread d_thread;
    bool d_thread_stopped = false;

    static void thread_body(scheduler_simplestream* top)
    {
        int num_empty = 0;
        bool work_done = false;
        top->set_state(scheduler_state::WORKING);
        gr_log_info(top->_logger, "starting thread");
        while (!top->d_thread_stopped) {
            // std::cout << top->name() << ":while" << std::endl;

            // do stuff with the blocks
            bool did_work = false;
            bool go_on_ahead = false;
            // TODO - line up at_sample numbers with work functions
            for (auto const& b : top->d_blocks) {
                // handle parameter changes - queues need to be made thread safe
                while (!top->param_change_queue.empty()) {
                    auto item = top->param_change_queue.front();
                    gr_log_debug(top->_debug_logger,
                                 "param_change_queue - dequeue {} - {}",
                                 item.block_id,
                                 b->alias());
                    if (item.block_id == b->alias()) {
                        b->on_parameter_change(item.param_action);

                        if (item.cb_fcn != nullptr)
                            item.cb_fcn(item.param_action);

                        top->param_change_queue.pop();
                    } else {
                        // no parameter changes for this block
                        go_on_ahead = true;
                        break;
                    }
                }

                // if (go_on_ahead)
                // continue;

                // handle parameter queries
                while (!top->param_query_queue.empty()) {
                    auto item = top->param_query_queue.front();
                    gr_log_debug(top->_debug_logger,
                                 "param_query_queue - dequeue {} - {}",
                                 item.block_id,
                                 b->alias());
                    if (item.block_id == b->alias()) {
                        b->on_parameter_query(item.param_action);

                        if (item.cb_fcn != nullptr)
                            item.cb_fcn(item.param_action);

                        top->param_query_queue.pop();
                    } else {
                        // no parameter changes for this block
                        go_on_ahead = true;
                        break;
                    }
                }

                // handle general callbacks
                while (!top->callback_queue.empty()) {
                    auto item = top->callback_queue.front();
                    gr_log_debug(top->_debug_logger,
                                 "callback_queue - dequeue {} - {}",
                                 item.block_id,
                                 b->alias());
                    if (item.block_id == b->alias()) {
                        auto cbs = item.cb_struct;
                        auto ret = b->callbacks()[cbs.callback_name](cbs.args);

                        cbs.return_val = ret;
                        if (item.cb_fcn != nullptr)
                            item.cb_fcn(cbs);

                        top->callback_queue.pop();
                    } else {
                        // no parameter changes for this block
                        go_on_ahead = true;
                        break;
                    }
                }
                    
                std::vector<block_work_input> work_input;   //(num_input_ports);
                std::vector<block_work_output> work_output; //(num_output_ports);

                std::vector<buffer_sptr> bufs;
                // for each input port of the block
                bool ready = true;
                for (auto p : b->input_stream_ports()) {
                    auto p_buf = top->d_block_buffers[p][0];


                    buffer_info_t read_info;
                    ready = p_buf->read_info(read_info);
                    gr_log_debug(top->_debug_logger,
                                 "read_info {} - {}",
                                 b->name(),
                                 read_info.n_items);

                    // std::cout << top->name() << ":" << b->name() << ":read_info:" <<
                    // ready << "-" << read_info.n_items << std::endl;
                    if (!ready)
                        break;

                    bufs.push_back(p_buf);

                    if (read_info.n_items < s_min_items_to_process) {
                        ready = false;
                        break;
                    }

                    std::vector<tag_t> tags; // needs to be associated with edge buffers
                    work_input.push_back(
                        block_work_input(read_info.n_items, 0, read_info.ptr, tags));
                }

                if (!ready) {
                    // clean up the buffers that we now won't be using
                    gr_log_debug(top->_debug_logger, "cancel");
                    for (auto buf : bufs) {
                        buf->cancel();
                    }
                    std::this_thread::yield();
                    continue;
                }

                // for each output port of the block
                for (auto p : b->output_stream_ports()) {

                    // When a block has multiple output buffers, it adds the restriction
                    // that the work call can only produce the minimum available across
                    // the buffers.

                    int max_output_buffer = std::numeric_limits<int>::max();

                    void* write_ptr = nullptr;
                    for (auto p_buf : top->d_block_buffers[p]) {
                        buffer_info_t write_info;
                        ready = p_buf->write_info(write_info);
                        gr_log_debug(top->_debug_logger,
                                     "write_info {} - {} @ {} {}",
                                     b->name(),
                                     write_info.n_items,
                                     write_info.ptr,
                                     write_info.item_size);
                        if (!ready)
                            break;
                        bufs.push_back(p_buf);

                        if (write_info.n_items <= top->s_max_buf_items) {
                            ready = false;
                            break;
                        }

                        int tmp_buf_size = write_info.n_items;
                        if (tmp_buf_size < max_output_buffer - 1)
                            max_output_buffer = tmp_buf_size;

                        // store the first buffer
                        if (!write_ptr)
                            write_ptr = write_info.ptr;
                    }

                    max_output_buffer = std::min(max_output_buffer, top->s_max_buf_items);
                    std::vector<tag_t> tags; // needs to be associated with edge buffers

                    work_output.push_back(
                        block_work_output(max_output_buffer, 0, write_ptr, tags));
                }

                if (!ready) {
                    // clean up the buffers that we now won't be using
                    for (auto buf : bufs) {
                        buf->cancel();
                    }
                    continue;
                }

                if (ready) {
                    gr_log_debug(top->_debug_logger, "do_work for {}", b->alias());
                    work_return_code_t ret = b->do_work(work_input, work_output);
                    gr_log_debug(top->_debug_logger, "do_work returned {}", ret);


                    if (ret == work_return_code_t::WORK_DONE) {
                        work_done = true;
                        {
                            // Signal to the monitor thread that we are done working
                            std::lock_guard<std::mutex> lk(top->sched_sync->sync_mutex);
                            top->sched_sync->id = top->id();
                            top->sched_sync->state = scheduler_state::DONE;

                            if (top->state() != scheduler_state::EXIT)
                                top->set_state(scheduler_state::DONE);
                        }
                        top->sched_sync->ready = true;
                        top->sched_sync->sync_cv.notify_one();
                    }
                    if (ret == work_return_code_t::WORK_OK ||
                        ret == work_return_code_t::WORK_DONE) {

                        int i = 0;
                        for (auto p : b->input_stream_ports()) {
                            auto p_buf =
                                top->d_block_buffers[p]
                                                    [0]; // only one buffer per input port

                            gr_log_debug(top->_debug_logger,
                                         "post_read {} - {}",
                                         b->name(),
                                         work_input[i].n_consumed);
                            p_buf->post_read(work_input[i].n_consumed);
                            gr_log_debug(top->_debug_logger,".");
                            i++;
                        }

                        i = 0;
                        for (auto p : b->output_stream_ports()) {
                            int j = 0;
                            for (auto p_buf : top->d_block_buffers[p]) {
                                if (j > 0) {
                                    gr_log_debug(top->_debug_logger,
                                                 "copy_items {} - {}",
                                                 b->name(),
                                                 work_output[i].n_produced);
                                    p_buf->copy_items(top->d_block_buffers[p][0],
                                                      work_output[i].n_produced);
                                    gr_log_debug(top->_debug_logger,".");
                                }
                                j++;
                            }
                            for (auto p_buf : top->d_block_buffers[p]) {
                                gr_log_debug(top->_debug_logger,
                                             "post_write {} - {}",
                                             b->name(),
                                             work_output[i].n_produced);
                                p_buf->post_write(work_output[i].n_produced);
                                gr_log_debug(top->_debug_logger,".");
                            }
                            i++;
                        }
                        // update the buffers according to the items produced
                        if (ret != work_return_code_t::WORK_DONE)
                            did_work = true;
                    } else {
                        for (auto buf : bufs) {
                            buf->cancel();
                        }
                        std::this_thread::yield();
                    }
                }
            }


            if (!did_work) {
                // std::this_thread::yield();
                gr_log_debug(top->_debug_logger, "no work in this iteration");
                std::this_thread::sleep_for(std::chrono::microseconds(2));
                // No blocks did work in this iteration

                if (top->state() == scheduler_state::DONE) {
                    // num_empty++;
                    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    gr_log_debug(top->_debug_logger, "flushing ..");

                    // if (num_empty >= 10) {
                    top->set_state(scheduler_state::FLUSHED);
                    // }
                }
            }

            if (top->state() == scheduler_state::EXIT) {
                gr_log_debug(top->_debug_logger,"scheduler state has been set to exit");
                top->d_thread_stopped = true;
                break;
            }
        }

        gr_log_debug(top->_debug_logger,"exiting");
        gr_log_info(top->_logger, "exiting");
    }
};
} // namespace schedulers
} // namespace gr
