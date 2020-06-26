#pragma once

#include <gnuradio/flat_graph.hpp>
#include <gnuradio/blocklib/callback.hpp>
#include <memory>
namespace gr {
class scheduler : public std::enable_shared_from_this<scheduler>
{

public:
    scheduler(){};
    virtual ~scheduler();
    std::shared_ptr<scheduler> base() { return shared_from_this(); }
    virtual void initialize(flat_graph_sptr fg) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;

    virtual void request_parameter_query(std::string block_alias,
                                         param_query_base param_change,
                                         std::function<void(param_query_base)> cb_when_complete)
    {
    }

    virtual void request_parameter_change(std::string block_alias,
                                          param_change_base param_change,
                                          std::function<void(param_change_base)> fn)
    {
    }

    virtual void request_callback(std::string block_alias,
                                          callback_base callback,
                                          std::function<void(callback_base)> fn)
    {
    }
};

typedef std::shared_ptr<scheduler> scheduler_sptr;
} // namespace gr