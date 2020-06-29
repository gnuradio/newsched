#pragma once

#include <gnuradio/blocklib/callback.hpp>
#include <gnuradio/flat_graph.hpp>
#include <functional>
#include <memory>
#include <queue>
namespace gr {
class scheduler : public std::enable_shared_from_this<scheduler>
{


public:
    typedef std::function<void(param_change_base)> param_change_complete_fcn;

    std::queue<std::tuple<std::string, param_change_base>> param_change_queue;
    // separate queue because I can't get the std::function to be part of the tuple
    std::queue<param_change_complete_fcn> param_change_cb_queue;

    scheduler(){};
    virtual ~scheduler();
    std::shared_ptr<scheduler> base() { return shared_from_this(); }
    virtual void initialize(flat_graph_sptr fg) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;

    virtual void request_parameter_query(const std::string& block_alias,
                                         param_query_base param_change,
                                         param_change_complete_fcn cb_when_complete)
    {
    }

    virtual void request_parameter_change(const std::string& block_alias,
                                          param_change_base param_change,
                                          param_change_complete_fcn fn)
    {
        param_change_queue.emplace(
            std::tuple<std::string, param_change_base>(block_alias, param_change));
        param_change_cb_queue.push(fn);


        std::cout << param_change_queue.size();
    }

    virtual void request_callback(const std::string& block_alias,
                                  callback_base callback,
                                  std::function<void(callback_base)> fn)
    {
    }

    // std::queue<std::tuple<std::string, param_change_base>> param_change_queue()
    // {
    //     return _param_change_queue;
    // }
};

typedef std::shared_ptr<scheduler> scheduler_sptr;
} // namespace gr