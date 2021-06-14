#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gnuradio/block_work_io.hh>
#include <gnuradio/node.hh>
#include <gnuradio/gpdict.hh>

namespace gr {

class scheduler; // Forward declaration to scheduler class

/**
 * @brief The abstract base class for all signal processing blocks in the GR Block Library
 *
 * Blocks are the bare abstraction of an entity that has a name and a set of inputs and
 * outputs  These are never instantiated directly; rather, this is the abstract parent
 * class of blocks that implement actual signal processing functions.
 *
 */
class block : public gr::node, public std::enable_shared_from_this<block>
{
private:
    bool d_running = false;
    tag_propagation_policy_t d_tag_propagation_policy;
    int d_output_multiple = 1;
    bool d_output_multiple_set = false;
    double d_relative_rate = 1.0;

protected:
    std::shared_ptr<scheduler> p_scheduler = nullptr;

public:
    /**
     * @brief Construct a new block object
     *
     * @param name The non-unique name of this block representing the block type
     */
    block(const std::string& name)
        : node(name), d_tag_propagation_policy(tag_propagation_policy_t::TPP_ALL_TO_ALL)
    {
    }
    virtual ~block(){};

    virtual bool start()
    {
        d_running = true;
        return true;
    }
    virtual bool stop()
    {
        d_running = false;
        return true;
    }

    virtual bool done()
    {
        d_running = false;
        return true;
    }

    typedef std::shared_ptr<block> sptr;
    sptr base() { return shared_from_this(); }

    tag_propagation_policy_t tag_propagation_policy()
    {
        return d_tag_propagation_policy;
    };
    void set_tag_propagation_policy(tag_propagation_policy_t policy)
    {
        d_tag_propagation_policy = policy;
    };

    /**
     * @brief Abstract method to call signal processing work from a derived block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        throw std::runtime_error("work function has been called but not implemented");
    }
    
    // /**
    //  * @brief Basic step function which just calls into the work function. 
    //  * This provides an API into blocks that's more stand-alone.
    //  * 
    //  * @note This does not belong in this header file as part of the exposed API 
    //  * 
    //  * @param buffer 
    //  * @param num_items 
    //  */
    // virtual std::vector<block_work_output>& step(void* data, size_t num_items, size_t item_size) {
    //     // efficiently create a block_work_input
    //     auto buf_props = std::make_shared<buffer_properties>();
    //     auto buf = std::make_shared<buffer>(num_items, item_size, buf_props);
    //     auto buf_reader = std::make_shared<buffer_reader>(buf, buf_props, 0);
    //     block_work_input input = block_work_input(num_items, buf_reader);

    //     // efficiently create a block_work_output
    //     // this requires some instrospection from the block to understand 
    //     // what kind of block_work_output it would need to be able to handle the input

    //     // call the work function using the input/output

    //     // return the block_work_output
    // };

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

    void set_scheduler(std::shared_ptr<scheduler> sched) { p_scheduler = sched; }

    void consume_each(int num, std::vector<block_work_input>& work_input)
    {
        for (auto& input : work_input) {
            input.consume(num);
        }
    }

    void produce_each(int num, std::vector<block_work_output>& work_output)
    {
        for (auto& output : work_output) {
            output.produce(num);
        }
    }

    void set_output_multiple(int multiple)
    {
        if (multiple < 1)
            throw std::invalid_argument("block::set_output_multiple");

        d_output_multiple_set = true;
        d_output_multiple = multiple;
    }
    int output_multiple() const { return d_output_multiple; }
    bool output_multiple_set() const { return d_output_multiple_set; }

    void set_relative_rate(double relative_rate) { d_relative_rate = relative_rate; }
    double relative_rate() const { return d_relative_rate; }

    gpdict attributes; // this is a HACK for storing metadata.  Needs to go.
};

typedef block::sptr block_sptr;
typedef std::vector<block_sptr> block_vector_t;
typedef std::vector<block_sptr>::iterator block_viter_t;

} // namespace gr
