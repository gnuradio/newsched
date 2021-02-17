#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

class fanout : public sync_block
{
public:
    enum params : uint32_t { id_itemsize, id_nports, num_params };

    typedef std::shared_ptr<fanout> sptr;
    static sptr make( size_t itemsize, size_t nports = 2)
    {
        auto ptr = std::make_shared<fanout>(itemsize, nports);

        ptr->add_port(untyped_port::make("input",
                                    port_direction_t::INPUT,
                                    itemsize));

        // TODO : do this with multiplicity
        for (size_t i = 0; i < nports; i++) {
            ptr->add_port(untyped_port::make("out" + std::to_string(i),
                                        port_direction_t::OUTPUT,
                                        itemsize));

        }

        return ptr;
    }

    fanout(size_t itemsize, size_t nports)
        : sync_block("fanout"), _itemsize(itemsize), _nports(nports)
    {
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        auto* iptr = (uint8_t*)work_input[0].items;
        for (size_t n = 0; n < _nports; n++) {
            int size = work_output[0].n_items * _itemsize;
            auto* optr = (uint8_t*)work_output[n].items;
            std::copy(iptr, iptr+size, optr);
            work_output[n].n_produced = work_output[n].n_items;
        }

        return work_return_code_t::WORK_OK;
    }

private:
    size_t _itemsize;
    size_t _nports;
};

} // namespace blocks
} // namespace gr
