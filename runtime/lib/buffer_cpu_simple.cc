#include <gnuradio/buffer_cpu_simple.hh>

namespace gr
{
    std::shared_ptr<buffer_reader> buffer_cpu_simple::add_reader(std::shared_ptr<buffer_properties> buf_props)
    {
        std::shared_ptr<buffer_cpu_simple_reader> r(
            new buffer_cpu_simple_reader(shared_from_this(), buf_props, _write_index));
        _readers.push_back(r.get());
        return r;
    }
}