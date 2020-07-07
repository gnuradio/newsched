#pragma once

#include <gnuradio/blocklib/sync_block.hpp>

namespace gr {
namespace blocks {

template <class T>
class dummy : public sync_block
{
public:
    enum params : uint32_t { id_a, id_b, id_vlen, num_params };

    typedef std::shared_ptr<dummy> sptr;
    static sptr make(T a, T b, size_t vlen = 1)
    {
        auto ptr = std::make_shared<dummy>(dummy());

        ptr->add_param(param<T>::make(dummy<T>::params::id_a, "a", a, &(ptr->_a)));
        ptr->add_param(param<T>::make(dummy<T>::params::id_b, "b", b, &(ptr->_b)));
        ptr->add_param(param<size_t>::make(dummy<T>::params::id_vlen, "vlen", vlen, &(ptr->_vlen)));

        return ptr;
    }
    dummy() : sync_block("dummy") {}
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
    {
        int size = work_output[0].n_items * _vlen;

        auto* iptr = (T*)work_input[0].items;
        auto* optr1 = (T*)work_output[0].items;
        auto* optr2 = (T*)work_output[1].items;

        for (auto i = 0; i < size; i++) {
            optr1[i] = _a * iptr[i];
            optr2[i] = _b * iptr[i];
        }

        return work_return_code_t::WORK_OK;
    }

private:
    T _a, _b;
    size_t _vlen;
};



} // namespace blocks
} // namespace gr