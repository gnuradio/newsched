#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include <gnuradio/buffer.hpp>
#include <gnuradio/flat_graph.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/logging.hpp>
#include <gnuradio/scheduler_message.hpp>
#include <gnuradio/neighbor_interface_info.hpp>
namespace gr {

/**
 * @brief Base class for GNU Radio Scheduler
 * 
 */
class scheduler : public std::enable_shared_from_this<scheduler>, public neighbor_interface
{
public:
    scheduler(const std::string& name)
    {
        _name = name;
        _logger = logging::get_logger(name, "default");
        _debug_logger = logging::get_logger(name + "_dbg", "debug");
    };
    virtual ~scheduler() {}
    std::shared_ptr<scheduler> base() { return shared_from_this(); }
    virtual void initialize(flat_graph_sptr fg, flowgraph_monitor_sptr fgmon) = 0;
    virtual void push_message(scheduler_message_sptr msg) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;

    std::string name() { return _name; }
    int id() { return _id; }
    void set_id(int id) { _id = id; }

    virtual void
    set_default_buffer_factory(
                               std::shared_ptr<buffer_properties> bp = nullptr)
    {
        _default_buf_properties = bp;
    }

protected:
    logger_sptr _logger;
    logger_sptr _debug_logger;

    std::shared_ptr<buffer_properties> _default_buf_properties = nullptr;

private:
    std::string _name;
    int _id;
};

typedef std::shared_ptr<scheduler> scheduler_sptr;

} // namespace gr
