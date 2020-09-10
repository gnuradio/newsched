#pragma once

#include <any>
#include <functional>
#include <queue>
#include <vector>

#include <gnuradio/scheduler_message.hpp>

namespace gr {

struct callback_args {
    std::string callback_name;
    std::vector<std::any> args;
    std::any return_val;
    uint64_t at_sample;
};


typedef std::function<std::any(std::vector<std::any>)> block_callback_fcn;
typedef std::function<void(callback_args)> block_callback_complete_fcn;

class callback_args_with_callback : public scheduler_message
{
public:
    callback_args_with_callback(std::string block_id,
                                callback_args cb_struct,
                                block_callback_complete_fcn cb_fcn)
        : scheduler_message(scheduler_message_t::CALLBACK),
          _block_id(block_id),
          _cb_struct(cb_struct),
          _cb_fcn(cb_fcn)
    {
    }
    std::string _block_id;
    callback_args _cb_struct;
    block_callback_complete_fcn _cb_fcn;
};

// typedef callback_function

} // namespace gr