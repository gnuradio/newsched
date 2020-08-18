
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_BLOCK_CALLBACKS_HPP
#define INCLUDED_BLOCK_CALLBACKS_HPP

#include <functional>
#include <map>
#include <string>

namespace gr {
// use string for now, pmt replacement later
typedef std::function<void(std::string)> block_callback_function_t;

/**
 * @brief Asynchronous message format for a block
 *
 */
class block_callback
{
private:
    std::string d_callback_id;
    block_callback_function_t d_callback_function;

public:
    block_callback(std::string& id, std::function<void(std::string)> callback_function)
    {
        d_callback_id = id;
        d_callback_function = callback_function;
    }
    ~block_callback() {}

    std::string id() { return d_callback_id; }
    block_callback_function_t callback_function() { return d_callback_function; }
};

/**
 * @brief Container for all the asynchronous message functions for a block
 *
 */
class block_callbacks
{

private:
    std::map<std::string, block_callback_function_t> callback_map;

public:
    block_callbacks() {}
    ~block_callbacks() {}

    void add(block_callback& callback)
    {
        callback_map[callback.id()] = callback.callback_function();
    }
    //   void remove(block_callback &callback);
};

} // namespace gr
#endif
