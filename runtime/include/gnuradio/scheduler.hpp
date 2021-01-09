#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include <gnuradio/buffer.hpp>
#include <gnuradio/callback.hpp>
#include <gnuradio/flat_graph.hpp>
#include <gnuradio/flowgraph_monitor.hpp>
#include <gnuradio/logging.hpp>
#include <gnuradio/scheduler_message.hpp>
#include <gnuradio/neighbor_interface.hpp>
namespace gr {

class scheduler : public std::enable_shared_from_this<scheduler>, public neighbor_interface
{

public:

    scheduler(const std::string& name)
    {
        _name = name;
        _logger = logging::get_logger(name, "default");
        _debug_logger = logging::get_logger(name + "_dbg", "debug");
    };
    virtual ~scheduler();
    std::shared_ptr<scheduler> base() { return shared_from_this(); }
    virtual void
    initialize(flat_graph_sptr fg,
               flowgraph_monitor_sptr fgmon,
               neighbor_interface_map scheduler_adapter_map = neighbor_interface_map()) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;

    virtual void request_parameter_query(const nodeid_t blkid,
                                         param_action_sptr param_action,
                                         param_action_complete_fcn cb_when_complete)
    {
        gr_log_debug(_debug_logger, "request_parameter_query: {}", blkid);

        push_message(
            std::make_shared<param_query_action>(blkid, param_action, cb_when_complete));
    }

    virtual void request_parameter_change(const nodeid_t blkid,
                                          param_action_sptr param_action,
                                          param_action_complete_fcn cb_when_complete)
    {
        push_message(
            std::make_shared<param_change_action>(blkid, param_action, cb_when_complete));
    }

    virtual void request_callback(const std::string& block_alias,  // FIXME: change to nodeid
                                  const callback_args& args,
                                  block_callback_complete_fcn cb_when_complete)
    {
        push_message(std::make_shared<callback_args_with_callback>(
            block_alias, args, cb_when_complete));
    }

    std::string name() { return _name; }
    int id() { return _id; }
    void set_id(int id) { _id = id; }

    virtual void set_default_buffer_factory(const buffer_factory_function& bff,
                                            std::shared_ptr<buffer_properties> bp = nullptr)
    {
        _default_buf_factory = bff;
        _default_buf_properties = bp;
    }

protected:
    logger_sptr _logger;
    logger_sptr _debug_logger;

    buffer_factory_function _default_buf_factory = nullptr;
    std::shared_ptr<buffer_properties> _default_buf_properties = nullptr;

private:
    std::string _name;
    int _id;
};

typedef std::shared_ptr<scheduler> scheduler_sptr;

} // namespace gr
