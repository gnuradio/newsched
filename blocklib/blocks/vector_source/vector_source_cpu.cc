#include "vector_source_cpu.hh"
#include "vector_source_cpu_gen.hh"
#include <algorithm>
#include <cstring> // for memcpy
#include <stdexcept>

using namespace std;

namespace gr {
namespace blocks {

template <class T>
vector_source_cpu<T>::vector_source_cpu(const typename vector_source<T>::block_args& args)
    : sync_block("vector_source"),
      vector_source<T>(args),
      d_data(args.data),
      d_repeat(args.repeat),
      d_offset(0),
      d_vlen(args.vlen),
      d_tags(args.tags)
{
    if ((args.data.size() % args.vlen) != 0)
        throw std::invalid_argument("data length must be a multiple of vlen");
}


template <class T>
work_return_code_t vector_source_cpu<T>::work(std::vector<block_work_input>& work_input,
                                          std::vector<block_work_output>& work_output)
{

    // size_t noutput_ports = work_output.size(); // is 1 for this block
    int noutput_items = work_output[0].n_items;
    auto optr = work_output[0].items<T>();

    if (d_repeat) {
        unsigned int size = d_data.size();
        unsigned int offset = d_offset;
        if (size == 0)
            return work_return_code_t::WORK_DONE;

        for (int i = 0; i < static_cast<int>(noutput_items * d_vlen); i++) {
            optr[i] = d_data[offset++];
            if (offset >= size) {
                offset = 0;
            }
        }

        d_offset = offset;

        work_output[0].n_produced = noutput_items;
        return work_return_code_t::WORK_OK;

    } else {
        if (d_offset >= d_data.size()) {
            work_output[0].n_produced = 0;
            return work_return_code_t::WORK_DONE; // Done!
        }

        unsigned n = std::min(d_data.size() - d_offset, noutput_items * d_vlen);
        for (unsigned i = 0; i < n; i++) {
            optr[i] = d_data[d_offset + i];
        }
        d_offset += n;

        work_output[0].n_produced = n / d_vlen;
        return work_return_code_t::WORK_OK;
    }
}

} /* namespace blocks */
} /* namespace gr */
