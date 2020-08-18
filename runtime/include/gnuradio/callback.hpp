#pragma once

#include <functional>
#include <queue>
#include <any>
#include <vector>

namespace gr {

struct callback_args
{
    std::string callback_name;
    std::vector<std::any> args;
    std::any return_val;
    uint64_t at_sample;
};


typedef std::function<std::any(std::vector<std::any>)> block_callback_fcn;
typedef std::function<void(callback_args)> block_callback_complete_fcn;

struct callback_args_with_callback {
    std::string block_id;
    callback_args cb_struct;
    block_callback_complete_fcn cb_fcn;
};



typedef std::queue<block_callback_fcn> block_callback_fcn_queue;
typedef std::queue<callback_args_with_callback> block_callback_queue;



// typedef callback_function

} // namespace gr