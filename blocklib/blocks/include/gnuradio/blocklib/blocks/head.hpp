#pragma once
#include <string.h>


#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

class head : public sync_block
{
public:
    enum params : uint32_t { id_itemsize, id_nitems, num_params };

    typedef std::shared_ptr<head> sptr;
    static sptr make(size_t itemsize, size_t nitems)
    {
        auto ptr = std::make_shared<head>(head(itemsize, nitems));

        ptr->add_param(param<size_t>::make(
            head::params::id_nitems, "itemsize", itemsize, &(ptr->_itemsize)));

        ptr->add_param(param<size_t>::make(
            head::params::id_nitems, "nitems", nitems, &(ptr->_nitems)));

        ptr->add_port(untyped_port::make("input",
                                    port_direction_t::INPUT,
                                    itemsize,
                                    port_type_t::STREAM));


        ptr->add_port(untyped_port::make("output",
                                    port_direction_t::OUTPUT,
                                    itemsize,
                                    port_type_t::STREAM));


        return ptr;
    }

    head(size_t itemsize, size_t nitems)
        : sync_block("head"), _itemsize(itemsize), _nitems(nitems), _ncopied_items(0)
    {
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        auto* iptr = (uint8_t*)work_input[0].items;
        auto* optr = (uint8_t*)work_output[0].items;

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

        memcpy(optr, iptr, n*_itemsize);
        
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