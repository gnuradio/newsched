#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hh>

namespace gr {


class buffer_cuda_pinned : public buffer
{
private:
    uint8_t* _pinned_buffer;

public:
    typedef std::shared_ptr<buffer_cuda_pinned> sptr;
    buffer_cuda_pinned(size_t num_items,
                       size_t item_size,
                       std::shared_ptr<buffer_properties> buf_properties);
    ~buffer_cuda_pinned();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);
    void* read_ptr(size_t index);
    void* write_ptr();
    virtual void post_write(int num_items);

    virtual std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props, size_t itemsize);
};
class buffer_cuda_pinned_reader : public buffer_reader
{
public:
    buffer_cuda_pinned_reader(buffer_sptr buffer,
                              std::shared_ptr<buffer_properties> buf_props,
                              size_t itemsize,
                              size_t read_index)
        : buffer_reader(buffer, buf_props, itemsize, read_index)
    {
    }

    virtual void post_read(int num_items);
};

class buffer_cuda_pinned_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    buffer_cuda_pinned_properties() : buffer_properties()
    {
        _bff = buffer_cuda_pinned::make;
    }
    static std::shared_ptr<buffer_properties> make()
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<buffer_cuda_pinned_properties>());
    }
};


} // namespace gr

#define CUDA_BUFFER_PINNED_ARGS buffer_cuda_pinned_properties::make()
