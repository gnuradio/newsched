#pragma once
#include <string.h>


#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

class nop_head : public sync_block
{
public:
    enum params : uint32_t { id_itemsize, id_nitems, num_params };

    typedef std::shared_ptr<nop_head> sptr;
    static sptr make(size_t itemsize, size_t nitems)
    {
        auto ptr = std::make_shared<nop_head>(itemsize, nitems);

        ptr->add_port(untyped_port::make("input",
                                    port_direction_t::INPUT,
                                    itemsize));


        ptr->add_port(untyped_port::make("output",
                                    port_direction_t::OUTPUT,
                                    itemsize));


        return ptr;
    }

    nop_head(size_t itemsize, size_t nitems)
        : sync_block("nop_head"), _itemsize(itemsize), _nitems(nitems), _ncopied_items(0)
    {
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        // auto* iptr = (uint8_t*)work_input[0].buffer->read_ptr();
        // auto* optr = (uint8_t*)work_output[0].buffer->write_ptr();

        if (_ncopied_items >= _nitems)
        {
            work_output[0].n_produced = 0;
            return work_return_code_t::WORK_DONE; // Done!
        }

        unsigned n = std::min(_nitems - _ncopied_items, (uint64_t)work_output[0].n_items);

        if (n == 0)
        {
            work_output[0].n_produced = 0;
            return work_return_code_t::WORK_OK;
        }

        // memcpy(optr, iptr, n*_itemsize);
        
        _ncopied_items += n;
        work_output[0].n_produced = n;

        return work_return_code_t::WORK_OK;
    }

    // TODO - add reset() callback

private:
    size_t _itemsize;
    size_t _nitems;

    size_t _ncopied_items;
};


} // namespace blocks
} // namespace gr
