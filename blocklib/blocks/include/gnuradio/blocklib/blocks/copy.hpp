#pragma once

#include <gnuradio/sync_block.hpp>
                                    
namespace gr {
namespace blocks {

work_return_code_t copy_kernel(std::vector<block_work_input>& work_input, std::vector<block_work_output>& work_output){
        auto* iptr = (uint8_t*)work_input[0].items;
        auto* optr = (uint8_t*)work_output[0].items;
        memcpy(optr, iptr, 100);
        work_output[0].n_produced = work_output[0].n_items;
        return work_return_code_t::WORK_OK;
};

class copy : public sync_block
{
public:
    enum params : uint32_t { id_itemsize, id_nports, num_params };
    typedef std::shared_ptr<copy> sptr;
    static sptr make(size_t itemsize)
    {
        auto ptr = std::make_shared<copy>(itemsize);

        ptr->add_port(untyped_port::make(
            "input", port_direction_t::INPUT, itemsize, port_type_t::STREAM));

        ptr->add_port(untyped_port::make(
            "out", port_direction_t::OUTPUT, itemsize, port_type_t::STREAM));


        return ptr;
    }

    copy(size_t itemsize)
        : sync_block("copy"), _itemsize(itemsize)
    {
        block_kernel = &copy_kernel;
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        return block_kernel(work_input, work_output);
    }

private:
    size_t _itemsize;
};

} // namespace blocks
} // namespace gr
