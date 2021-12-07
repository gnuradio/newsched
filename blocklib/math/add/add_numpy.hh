#pragma once

#include <gnuradio/math/add.hh>

namespace gr {
namespace math {

template <class T>
class add_numpy : public add<T>
{
public:
    add_numpy(const typename add<T>::block_args& args) : sync_block("add_cuda"), add<T>(args)
    {
        
    }

};


} // namespace math
} // namespace gr
