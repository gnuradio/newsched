#include <gnuradio/buffer_sm.hpp>

namespace gr
{
    std::shared_ptr<buffer_reader> buffer_sm::add_reader()
    {
        std::shared_ptr<buffer_sm_reader> r(
            new buffer_sm_reader(std::dynamic_pointer_cast<buffer_sm>(shared_from_this()), _write_index));
        _readers.push_back(r.get());
        return r;
    }
}