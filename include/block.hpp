
// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright 2020

#ifndef INCLUDED_BLOCK_HPP
#define INCLUDED_BLOCK_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"
#include "block_callbacks.hpp"
#include "block_work_io.hpp"
#include "io_signature.hpp"

namespace gr {

enum class work_return_code_t {
  WORK_CALLED_PRODUCE = -2,
  WORK_DONE = -1,
  WORK_OK = 0,
};

class block
{

protected:
    std::string d_name;
    io_signature d_input_signature;
    io_signature d_output_signature;

    // These are overridden by the derived class
    static const io_signature_capability d_input_signature_capability;
    static const io_signature_capability d_output_signature_capability;


    virtual work_return_code_t work(std::vector<block_work_io>& work_input,
                                    std::vector<block_work_io>& work_output);

    virtual int validate(); // ??
    virtual bool start();
    virtual bool stop();

    void set_relative_rate(double relative_rate);
    void set_relative_rate(unsigned int numerator, unsigned int denominator);

    //   tag_propagation_policy_t tag_propagation_policy();
    //   void set_tag_propagation_policy(tag_propagation_policy_t p);

    std::vector<block_callback> d_block_callbacks;
    block(const std::string& name,
          io_signature& input_signature,
          io_signature& output_signature)
        : d_name(name),
          d_input_signature(input_signature),
          d_output_signature(output_signature)
    {
    }

public:
    virtual ~block() {}

    static const io_signature_capability& input_signature_capability()
    {
        return d_input_signature_capability;
    }
    static const io_signature_capability& output_signature_capability()
    {
        return d_output_signature_capability;
    }

    io_signature& input_signature();
    io_signature& output_signature();

    std::string& name() { return d_name; };
};

} // namespace gr
#endif