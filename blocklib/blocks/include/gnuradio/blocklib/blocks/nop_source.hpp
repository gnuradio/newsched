#pragma once

#include <gnuradio/sync_block.hpp>
#include <string.h>

namespace gr {
namespace blocks {

class nop_source : virtual public sync_block
{

public:
    enum params : uint32_t { id_itemsize, id_nports, num_params };

    typedef std::shared_ptr<nop_source> sptr;
    static sptr make(size_t itemsize, size_t nports = 1)
    {

        auto ptr = std::make_shared<nop_source>(itemsize, nports);

        // TODO : do this with multiplicity
        for (size_t i = 0; i < nports; i++) {
            ptr->add_port(untyped_port::make("out" + std::to_string(i),
                                        port_direction_t::OUTPUT,
                                        itemsize));
        }

        return ptr;
    }

    nop_source(size_t itemsize, size_t nports) : sync_block("nop_source"), _itemsize(itemsize), _nports(nports) {}
    // ~nop_source() {};

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        // void* optr;

        for (size_t n = 0; n < work_output.size(); n++) {
        //     optr = work_output[n].buffer->write_ptr();
            auto noutput_items = work_output[n].n_items;
        //     memset(optr, 0, noutput_items * _itemsize);
            work_output[n].n_produced = noutput_items;
        }

        return work_return_code_t::WORK_OK;
    }


private:
    size_t _itemsize;
    size_t _nports;
};


} // namespace blocks
} // namespace gr
