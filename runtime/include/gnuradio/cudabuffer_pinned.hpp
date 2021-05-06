#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {

class cuda_buffer_pinned_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    cuda_buffer_pinned_properties()
        : buffer_properties()
    {
    }
    static std::shared_ptr<buffer_properties> make()
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<cuda_buffer_pinned_properties>());
    }
};


class cuda_buffer_pinned : public buffer
{
private:
    uint8_t* _pinned_buffer;

public:
    typedef std::shared_ptr<cuda_buffer_pinned> sptr;
    cuda_buffer_pinned(size_t num_items,
                size_t item_size);
    ~cuda_buffer_pinned();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);
    void* read_ptr(size_t index);
    void* write_ptr();
    virtual void post_write(int num_items);

    virtual std::shared_ptr<buffer_reader> add_reader(const std::string& name="");
};
class cuda_buffer_pinned_reader : public buffer_reader
{
public:
    cuda_buffer_pinned_reader(buffer_sptr buffer, size_t read_index = 0)
        : buffer_reader(buffer, read_index)
    {
    }

    virtual void post_read(int num_items);
};

} // namespace gr

#define CUDA_BUFFER_PINNED_ARGS cuda_buffer_pinned::make, cuda_buffer_pinned_properties::make()