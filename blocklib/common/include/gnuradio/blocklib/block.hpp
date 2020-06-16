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

#include <gnuradio/blocklib/node.hpp>
#include <gnuradio/blocklib/block_callbacks.hpp>
#include <gnuradio/blocklib/block_work_io.hpp>
#include <gnuradio/blocklib/io_signature.hpp>
#include <gnuradio/blocklib/parameter.hpp>
#include <memory>

namespace gr {

/**
 * @brief Enum for return codes from calls to block::work
 *
 */
enum class work_return_code_t {
    WORK_INSUFFICIENT_OUTPUT_ITEMS = -4,
    WORK_INSUFFICIENT_INPUT_ITEMS = -3,
    WORK_CALLED_PRODUCE = -2,
    WORK_DONE = -1,
    WORK_OK = 0,
};

/**
 * @brief The abstract base class for all signal processing blocks in the GR Block Library
 *
 * Blocks are the bare abstraction of an entity that has a
 * name and a set of inputs and outputs  These
 * are never instantiated directly; rather, this is the abstract
 * parent class of blocks that implement actual signal
 * processing functions.
 *
 */


class block : public gr::node, public std::enable_shared_from_this<block>
{
public:
    enum vcolor { WHITE, GREY, BLACK };
    enum io { INPUT, OUTPUT };

private:
    bool d_output_multiple_set = false;
    unsigned int d_output_multiple;

protected:

    vcolor d_color;

    // These are overridden by the derived class
    static const io_signature_capability d_input_signature_capability;
    static const io_signature_capability d_output_signature_capability;

    virtual int validate() { return 0; }; // ??
    virtual bool start() { return true; };
    virtual bool stop() { return true; };

    void set_relative_rate(double relative_rate) {};
    void set_relative_rate(unsigned int numerator, unsigned int denominator) {};

    //   tag_propagation_policy_t tag_propagation_policy();
    //   void set_tag_propagation_policy(tag_propagation_policy_t p);

    std::vector<block_callback> d_block_callbacks;

    parameter_config parameters;

    void add_param(param_base p) { parameters.add(p); }



public:
    /**
     * @brief Construct a new block object
     *
     * @param name The non-unique name of this block representing the block type
     */
    block(const std::string& name);

    virtual ~block() {};
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

    vcolor color() const { return d_color; }
    void set_color(vcolor color) { d_color = color; }

    /**
     * @brief Abstract method to call signal processing work from a derived block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) = 0;

    /**
     * @brief Wrapper for work to perform special checks and take care of special
     * cases for certain types of blocks, e.g. sync_block, decim_block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t do_work(std::vector<block_work_input>& work_input,
                                       std::vector<block_work_output>& work_output)
    {
        return work(work_input, work_output);
    };

    /**
     * @brief handler called when a parameter is changed:
     *   1. From a message port (automatically created message ports for callbacks)
     *   2. From a callback function (e.g. set_k())
     *   3. RPC call
     *
     * @param params
     */

    virtual void on_parameter_change(std::vector<param_change_base> params)
    {
        throw std::runtime_error("parameter changes not defined for this block");
    }
};

typedef block::sptr block_sptr;
typedef std::vector<block_sptr> block_vector_t;
typedef std::vector<block_sptr>::iterator block_viter_t;

} // namespace gr
#endif