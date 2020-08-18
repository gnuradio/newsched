#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class fanout : public sync_block
{
public:
    enum params : uint32_t { id_nports, id_vlen, num_params };

    typedef std::shared_ptr<fanout> sptr;
    static sptr make(size_t nports = 2, size_t vlen = 1)
    {
        auto ptr = std::make_shared<fanout>(fanout(nports, vlen));

        ptr->add_param(param<size_t>::make(
            fanout<T>::params::id_nports, "nports", nports, &(ptr->_nports)));
        ptr->add_param(
            param<size_t>::make(fanout<T>::params::id_vlen, "vlen", vlen, &(ptr->_vlen)));

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));

        // TODO : do this with multiplicity
        for (auto i = 0; i < nports; i++) {
            ptr->add_port(port<T>::make("out" + std::to_string(i),
                                        port_direction_t::OUTPUT,
                                        port_type_t::STREAM,
                                        std::vector<size_t>{ vlen }));
        }

        return ptr;
    }

    fanout(size_t nports, size_t vlen)
        : sync_block("fanout"), _nports(nports), _vlen(vlen)
    {
    }

    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        auto* iptr = (T*)work_input[0].items;
        for (auto n = 0; n < _nports; n++) {
            int size = work_output[0].n_items * _vlen;
            auto* optr = (T*)work_output[n].items;
            std::copy(iptr, iptr+size, optr);
            work_output[n].n_produced = work_output[n].n_items;
        }

        return work_return_code_t::WORK_OK;
    }

private:
    size_t _nports;
    size_t _vlen;
};

typedef fanout<std::int16_t> fanout_ss;
typedef fanout<std::int32_t> fanout_ii;
typedef fanout<float> fanout_ff;
typedef fanout<gr_complex> fanout_cc;

} // namespace blocks
} // namespace gr