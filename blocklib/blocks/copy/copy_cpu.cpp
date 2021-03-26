#include <gnuradio/blocks/copy.hpp>

namespace gr {
namespace blocks {
namespace impl {

class copy_cpu : public copy
{
public:
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;
private:
    size_t _itemsize;
};

work_return_code_t copy_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto* iptr = (uint8_t*)work_input[0].items();
    int size = work_output[0].n_items * _itemsize;
    auto* optr = (uint8_t*)work_output[0].items();
    std::copy(iptr, iptr + size, optr);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

} // namespace impl
} // namespace blocks
} // namespace gr