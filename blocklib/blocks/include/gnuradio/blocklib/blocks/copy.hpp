#pragma once

#include <gnuradio/kernels/cpu/copy.hpp>
#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

class copy : public sync_block
{
public:
    enum params : uint32_t { id_itemsize, id_nports, num_params };
    typedef std::shared_ptr<copy> sptr;

    static sptr make(size_t itemsize)
    {
        auto ptr = std::make_shared<copy>(itemsize);

        ptr->add_port(untyped_port::make("input", port_direction_t::INPUT, itemsize));

        ptr->add_port(untyped_port::make("out", port_direction_t::OUTPUT, itemsize));

        return ptr;
    }

    copy(size_t itemsize) : sync_block("copy"), _itemsize(itemsize)
    {
        if (itemsize == 1) {
            block_kernel = new gr::kernels::cpu::copy_kernel<uint8_t>;
        } else if (itemsize == 2) {
            block_kernel = new gr::kernels::cpu::copy_kernel<uint16_t>;
        } else if (itemsize == 4) {
            block_kernel = new gr::kernels::cpu::copy_kernel<uint32_t>;
        } else {
            block_kernel = new gr::kernels::cpu::copy_kernel<uint64_t>;
        }
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        auto* iptr = (uint8_t*)work_input[0].buffer->read_ptr();
        int size = work_output[0].n_items * _itemsize;
        auto* optr = (uint8_t*)work_output[0].buffer->write_ptr();
        // std::copy(iptr, iptr + size, optr);
        memcpy(optr, iptr, size);

        return work_return_code_t::WORK_OK;
    }

private:
    size_t _itemsize;
};
} // namespace blocks
} // namespace gr
