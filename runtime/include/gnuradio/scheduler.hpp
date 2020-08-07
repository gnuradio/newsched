#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

#include <gnuradio/blocklib/callback.hpp>
#include <gnuradio/flat_graph.hpp>

namespace gr {


enum class scheduler_state { WORKING, DONE, FLUSHED, EXIT};

struct scheduler_sync {
    std::mutex sync_mutex;
    std::condition_variable sync_cv;

    // These are the things to signal back to the main thread
    scheduler_state state; 
    int id;
};

class scheduler : public std::enable_shared_from_this<scheduler>
{

public:
    param_action_queue param_change_queue;
    param_action_queue param_query_queue;
    block_callback_queue callback_queue;

    scheduler(const std::string& name) { _name = name; };
    virtual ~scheduler();
    std::shared_ptr<scheduler> base() { return shared_from_this(); }
    virtual void initialize(flat_graph_sptr fg) = 0;
    virtual void start(scheduler_sync* sync) = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;

    virtual void request_parameter_query(const std::string& block_alias,
                                         param_action_sptr param_action,
                                         param_action_complete_fcn cb_when_complete)
    {
        param_query_queue.emplace(param_action_base_with_callback{
            block_alias, param_action, cb_when_complete });
    }

    virtual void request_parameter_change(const std::string& block_alias,
                                          param_action_sptr param_action,
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
    std::string name() { return _name; }
    int id() { return _id; }
    void set_id(int id) { _id = id; }
    scheduler_state state() { return _state; }
    void set_state (scheduler_state state) { _state = state; }

private:
    std::string _name;
    int _id;
    scheduler_state _state;
};

typedef std::shared_ptr<scheduler> scheduler_sptr;
} // namespace gr
