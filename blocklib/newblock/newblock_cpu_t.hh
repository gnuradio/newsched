#pragma once

#include <gnuradio/newmod/newblock.hh>

namespace gr {
namespace newmod {

template <class T>
class newblock_cpu : public newblock<T>
{
public:
    newblock_cpu(const typename newblock<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    // Declare private variables here
};


} // namespace newmod
} // namespace gr
