#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

class nop : public sync_block
{
public:
    enum params : uint32_t { id_itemsize, id_nports, num_params };

    typedef std::shared_ptr<nop> sptr;
    static sptr make(size_t itemsize)
    {
        auto ptr = std::make_shared<nop>(itemsize);

        ptr->add_port(untyped_port::make(
            "input", port_direction_t::INPUT, itemsize));

        ptr->add_port(untyped_port::make(
            "out", port_direction_t::OUTPUT, itemsize));


        return ptr;
    }

    nop(size_t itemsize)
        : sync_block("nop"), _itemsize(itemsize)
    {
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        // auto* iptr = (uint8_t*)work_input[0].buffer->read_ptr();
        // int size = work_output[0].n_items * _itemsize;
        // auto* optr = (uint8_t*)work_output[0].buffer->write_ptr();
        // std::nop(iptr, iptr + size, optr);
        // memcpy(optr, iptr, size);

        work_output[0].n_produced = work_output[0].n_items;
        return work_return_code_t::WORK_OK;
    }

private:
    size_t _itemsize;
};

} // namespace blocks
} // namespace gr
