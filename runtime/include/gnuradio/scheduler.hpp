#pragma once

#include <functional>
#include <memory>
#include <queue>

#include <gnuradio/blocklib/callback.hpp>
#include <gnuradio/flat_graph.hpp>

namespace gr {
class scheduler : public std::enable_shared_from_this<scheduler>
{

public:
    param_action_queue param_change_queue;
    param_action_queue param_query_queue;
    block_callback_queue callback_queue;

    scheduler(){};
    virtual ~scheduler();
    std::shared_ptr<scheduler> base() { return shared_from_this(); }
    virtual void initialize(flat_graph_sptr fg) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;

    virtual void request_parameter_query(const std::string& block_alias,
                                         param_action_base param_action,
                                         param_action_complete_fcn cb_when_complete)
    {
        param_query_queue.emplace(param_action_base_with_callback{
            block_alias, param_action, cb_when_complete });
    }

    virtual void request_parameter_change(const std::string& block_alias,
                                          param_action_base param_action,
                                          param_action_complete_fcn cb_when_complete)
    {
        param_change_queue.emplace(param_action_base_with_callback{
            block_alias, param_action, cb_when_complete });
    }

    virtual void request_callback(const std::string& block_alias,
                                  const callback_args& args,
                                  block_callback_complete_fcn cb_when_complete)
    {
        callback_queue.emplace(callback_args_with_callback{
            block_alias, args, cb_when_complete } );
    }

    // std::queue<std::tuple<std::string, param_action_base>> param_action_queue()
    // {
    //     return _param_action_queue;
    // }
};

typedef std::shared_ptr<scheduler> scheduler_sptr;
} // namespace gr