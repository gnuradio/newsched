/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_BLOCK_HPP
#define INCLUDED_BLOCK_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <gnuradio/blocklib/block_work_io.hpp>
#include <gnuradio/blocklib/io_signature.hpp>
#include <gnuradio/blocklib/types.hpp>
#include <gnuradio/blocklib/block_callbacks.hpp>
#include <memory>

namespace gr {

enum class work_return_code_t {
    WORK_INSUFFICIENT_OUTPUT_ITEMS = -4,
    WORK_INSUFFICIENT_INPUT_ITEMS = -3,
    WORK_CALLED_PRODUCE = -2,
    WORK_DONE = -1,
    WORK_OK = 0,
};

class block : public std::enable_shared_from_this<block>
{
public:
    enum vcolor { WHITE, GREY, BLACK };
    enum io { INPUT, OUTPUT };

private:
    bool d_output_multiple_set = false;
    unsigned int d_output_multiple;

protected:
    std::string d_name;
    std::string d_alias;
    io_signature d_input_signature;
    io_signature d_output_signature;

    vcolor d_color;

    // These are overridden by the derived class
    static const io_signature_capability d_input_signature_capability;
    static const io_signature_capability d_output_signature_capability;

    virtual int validate() { return 0; }; // ??
    virtual bool start() { return true; };
    virtual bool stop() { return true; };

    void set_relative_rate(double relative_rate);
    void set_relative_rate(unsigned int numerator, unsigned int denominator);

    //   tag_propagation_policy_t tag_propagation_policy();
    //   void set_tag_propagation_policy(tag_propagation_policy_t p);

    std::vector<block_callback> d_block_callbacks;


public:
    // block(void) {} // allows pure virtual interface sub-classes
    // block() = delete;
    block(const std::string& name,
          const io_signature& input_signature,
          const io_signature& output_signature);

    ~block();
    typedef std::shared_ptr<block> sptr;
    sptr base() { return shared_from_this(); }

    void set_output_multiple(unsigned int multiple)
    {
        if (multiple < 1)
            throw std::invalid_argument("block::set_output_multiple");

        d_output_multiple_set = true;
        d_output_multiple = multiple;
    }
    void set_alignment(unsigned int multiple) { set_output_multiple(multiple); }
    static const io_signature_capability& input_signature_capability()
    {
        return d_input_signature_capability;
    }
    static const io_signature_capability& output_signature_capability()
    {
        return d_output_signature_capability;
    }

    io_signature& input_signature() { return d_input_signature; };
    io_signature& output_signature() { return d_output_signature; };

    std::string& name() { return d_name; };
    std::string& alias() { return d_alias; }
    void set_alias(std::string alias) { d_alias = alias; }
    vcolor color() const { return d_color; }
    void set_color(vcolor color) { d_color = color; }

    // Must be overridden in the derived class
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) = 0;
    
    // Only called on the base class
    virtual work_return_code_t do_work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
                                    {
                                        return work(work_input, work_output);
                                    };
};

typedef block::sptr block_sptr;
typedef std::vector<block_sptr> block_vector_t;
typedef std::vector<block_sptr>::iterator block_viter_t;

} // namespace gr
#endif