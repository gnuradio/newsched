#pragma once

#include <gnuradio/block.hpp>
#include <gnuradio/block_interface.hpp>
#include <gnuradio/node_interface.hpp>

namespace gr {
class block_context : public block_interface, public node_interface
{
    protected:
        block_sptr block_impl = nullptr;
    public:
        block_context() {}
        block_context(block_sptr b) { block_impl = b; }
    /**
     * @brief Abstract method to call signal processing work from a derived block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) { return block_impl->work(work_input, work_output); }
    /**
     * @brief Wrapper for work to perform special checks and take care of special
     * cases for certain types of blocks, e.g. sync_block, decim_block
     *
     * @param work_input Vector of block_work_input structs
     * @param work_output Vector of block_work_output structs
     * @return work_return_code_t
     */
    virtual work_return_code_t do_work(std::vector<block_work_input>& work_input,
                                       std::vector<block_work_output>& work_output) { return block_impl->do_work(work_input, work_output); }

    virtual void set_scheduler(std::shared_ptr<scheduler> sched) { block_impl->set_scheduler(sched); }


    virtual void add_port(port_sptr p) { block_impl->add_port(p); }
    virtual std::vector<port_sptr>& all_ports() { return block_impl->all_ports(); }
    virtual std::vector<port_sptr>& input_ports() { return block_impl->input_ports(); }
    virtual std::vector<port_sptr>& output_ports() { return block_impl->output_ports(); }
    virtual std::vector<port_sptr> input_stream_ports() { return block_impl->input_stream_ports(); }
    virtual std::vector<port_sptr> output_stream_ports() { return block_impl->output_stream_ports(); }
    virtual std::string& name() { return block_impl->name(); }
    virtual std::string& alias() {return block_impl->alias(); }
    virtual uint32_t id() { return block_impl->id(); }
    virtual void set_alias(std::string alias) { block_impl->set_alias(alias); }
    virtual void set_id(uint32_t id) { block_impl->set_id(id); }
    virtual port_sptr get_port(const std::string& name) { return block_impl->get_port(name); }
    virtual message_port_sptr get_message_port(const std::string& name) { return block_impl->get_message_port(name); }
    virtual port_sptr
    get_port(unsigned int index, port_type_t type, port_direction_t direction) { return block_impl->get_port(index, type, direction); }
};

} // namespace gr