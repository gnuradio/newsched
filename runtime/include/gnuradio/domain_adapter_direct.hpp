#pragma once

#include <atomic>

#include <gnuradio/buffer.hpp>
#include <gnuradio/domain_adapter.hpp>
namespace gr {


class direct_sync
{
public:
    std::condition_variable cv;
    std::mutex mtx;
    std::atomic<int> ready = 0;

    da_request_t request;
    da_response_t response;

    buffer_sptr buffer;

    direct_sync() {}
    static std::shared_ptr<direct_sync> make()
    {
        return std::shared_ptr<direct_sync>(new direct_sync());
    }
};
typedef std::shared_ptr<direct_sync> direct_sync_sptr;

/**
 * @brief Uses a shared memory and a condition variable to communicate buffer pointers
 * between domains
 *
 */
class domain_adapter_direct_svr : public domain_adapter
{
private:
    bool d_connected = false;
    std::thread d_thread;

public:
    direct_sync_sptr p_sync;
    typedef std::shared_ptr<domain_adapter_direct_svr> sptr;
    static sptr make(direct_sync_sptr sync, port_sptr other_port)
    {
        auto ptr =
            std::make_shared<domain_adapter_direct_svr>(domain_adapter_direct_svr(sync));

        ptr->add_port(port_base::make("output",
                                      port_direction_t::OUTPUT,
                                      other_port->data_type(),
                                      port_type_t::STREAM,
                                      other_port->dims()));

        ptr->start_thread(ptr); // start thread with reference to shared pointer

        return ptr;
    }

    domain_adapter_direct_svr(direct_sync_sptr sync)
        : domain_adapter(buffer_location_t::LOCAL), p_sync(sync)
    {
    }

    void start_thread(sptr ptr) { d_thread = std::thread(run_thread, ptr); }

    static void run_thread(sptr top)
    {
        while (true) {
            {
                std::unique_lock<std::mutex> l(top->p_sync->mtx);
                // std::cout << "svr unlock" << std::endl;
                top->p_sync->cv.wait(l, [top] { return top->p_sync->ready == 1; });
                // std::cout << "svr out of wait" << std::endl;

                switch (top->p_sync->request) {

                case da_request_t::GET_REMOTE_BUFFER:
                    // std::cout << "svr WRITE_INFO" << std::endl;

                    top->p_sync->buffer = top->buffer();
                    top->p_sync->response = da_response_t::OK;
                }

                // std::cout << "svr out of switch" << std::endl;
                // l.unlock();
                // std::cout << "svr notify_one" << std::endl;
                top->p_sync->ready = 2;
                top->p_sync->cv.notify_one();
            }
        }
    }

    virtual void* read_ptr() { return nullptr; }
    virtual void* write_ptr() { return nullptr; }

    virtual bool read_info(buffer_info_t& info) { return _buffer->read_info(info); }
    virtual bool write_info(buffer_info_t& info) { return _buffer->write_info(info); }
    virtual void cancel() { _buffer->cancel(); }

    virtual void post_read(int num_items) { return _buffer->post_read(num_items); }
    virtual void post_write(int num_items) { return _buffer->post_write(num_items); }

    // This is not valid for all buffers, e.g. domain adapters
    virtual void copy_items(buffer_sptr from, int nitems) {}
};


class domain_adapter_direct_cli : public domain_adapter
{
private:
    direct_sync_sptr p_sync;
    buffer_sptr remote_buffer = nullptr;

public:
    typedef std::shared_ptr<domain_adapter_direct_cli> sptr;
    static sptr make(direct_sync_sptr sync, port_sptr other_port)
    {
        auto ptr =
            std::make_shared<domain_adapter_direct_cli>(domain_adapter_direct_cli(sync));

        // Type of port is not known at compile time
        ptr->add_port(port_base::make("input",
                                      port_direction_t::INPUT,
                                      other_port->data_type(),
                                      port_type_t::STREAM,
                                      other_port->dims()));

        return ptr;
    }
    domain_adapter_direct_cli(direct_sync_sptr sync)
        : domain_adapter(buffer_location_t::LOCAL), p_sync(sync)
    {
    }

    virtual void* read_ptr() { return nullptr; }
    virtual void* write_ptr() { return nullptr; }

    // virtual int capacity() = 0;
    // virtual int size() = 0;

    void get_remote_buffer()
    {
        {
            // std::lock_guard<std::mutex> l(p_sync->mtx);
            // std::cout << "read_info unlock" << std::endl;
            p_sync->request = da_request_t::GET_REMOTE_BUFFER;
            p_sync->ready = 1;
        }

        // std::cout << "read_info notify_one" << std::endl;
        p_sync->cv.notify_one();

        {
            std::unique_lock<std::mutex> l(p_sync->mtx);
            // std::cout << "read_info wait" << std::endl;
            p_sync->cv.wait(l, [this] { return p_sync->ready == 2; });

            remote_buffer = p_sync->buffer;
            p_sync->ready = 0;
        }
    }

    virtual bool read_info(buffer_info_t& info)
    {
        if (!remote_buffer)
            get_remote_buffer();

        return remote_buffer->read_info(info);
    }
    virtual bool write_info(buffer_info_t& info)
    {
        if (!remote_buffer)
            get_remote_buffer();

        return remote_buffer->write_info(info);
    }
    virtual void cancel()
    {
        if (!remote_buffer)
            get_remote_buffer();
        return remote_buffer->cancel();
    }

    virtual void post_read(int num_items)
    {
        if (!remote_buffer)
            get_remote_buffer();
        return remote_buffer->post_read(num_items);
    }
    virtual void post_write(int num_items)
    {
        if (!remote_buffer)
            get_remote_buffer();
        return remote_buffer->post_write(num_items);
    }

    // This is not valid for all buffers, e.g. domain adapters
    // Currently domain adapters require fanout, and cannot copy from a shared output
    // across multiple domains
    virtual void copy_items(buffer_sptr from, int nitems) {}
};


class domain_adapter_direct_conf : public domain_adapter_conf
{
public:
    typedef std::shared_ptr<domain_adapter_direct_conf> sptr;
    static sptr make(buffer_preference_t buf_pref = buffer_preference_t::DOWNSTREAM)
    {
        return std::make_shared<domain_adapter_direct_conf>(
            domain_adapter_direct_conf(buf_pref));
    }

    domain_adapter_direct_conf(buffer_preference_t buf_pref)
        : domain_adapter_conf(buf_pref)
    {
    }

    virtual std::pair<domain_adapter_sptr, domain_adapter_sptr>
    make_domain_adapter_pair(port_sptr upstream_port, port_sptr downstream_port, const std::string& name="")
    {
        auto direct_sync = direct_sync::make();

        if (_buf_pref == buffer_preference_t::DOWNSTREAM) {
            auto upstream_adapter =
                domain_adapter_direct_cli::make(direct_sync, upstream_port);
            auto downstream_adapter =
                domain_adapter_direct_svr::make(direct_sync, downstream_port);

            return std::make_pair(upstream_adapter, downstream_adapter);
        } else {
            auto downstream_adapter =
                domain_adapter_direct_cli::make(direct_sync, upstream_port);
            auto upstream_adapter =
                domain_adapter_direct_svr::make(direct_sync, downstream_port);

            return std::make_pair(upstream_adapter, downstream_adapter);
        }
    }
};


} // namespace gr