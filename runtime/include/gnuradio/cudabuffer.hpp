#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {
enum class cuda_buffer_type { D2D, H2D, D2H };


class cuda_buffer_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    cuda_buffer_properties(cuda_buffer_type buffer_type_,
                           size_t buf_size = 0,
                           size_t max_buffer_size = 0,
                           size_t min_buffer_size = 0,
                           size_t max_buffer_fill = 0,
                           size_t min_buffer_fill = 0)
        : buffer_properties(buf_size,
                            max_buffer_size,
                            min_buffer_size,
                            max_buffer_fill,
                            min_buffer_fill),
          _buffer_type(buffer_type_)
    {
    }
    cuda_buffer_type buffer_type() { return _buffer_type; }
    static std::shared_ptr<buffer_properties> make(cuda_buffer_type buffer_type_,
                                                   size_t buf_size = 0,
                                                   size_t max_buffer_size = 0,
                                                   size_t min_buffer_size = 0,
                                                   size_t max_buffer_fill = 0,
                                                   size_t min_buffer_fill = 0)
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<cuda_buffer_properties>(buffer_type_,
                                                     buf_size,
                                                     max_buffer_size,
                                                     min_buffer_size,
                                                     max_buffer_fill,
                                                     min_buffer_fill));
    }

private:
    cuda_buffer_type _buffer_type;
};


class cuda_buffer : public buffer
{
private:
    uint8_t* _host_buffer;
    uint8_t* _device_buffer;
    cuda_buffer_type _type;
    cudaStream_t stream;

public:
    typedef std::shared_ptr<cuda_buffer> sptr;
    cuda_buffer(size_t num_items,
                size_t item_size,
                cuda_buffer_type type = cuda_buffer_type::D2D);
    ~cuda_buffer();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);

    void* read_ptr(size_t read_index);
    void* write_ptr();

    virtual void post_write(int num_items);

    virtual std::shared_ptr<buffer_reader> add_reader();
};

class cuda_buffer_reader : public buffer_reader
{
public:
    cuda_buffer_reader(buffer_sptr buffer, size_t read_index = 0)
        : buffer_reader(buffer, read_index)
    {
    }

    virtual void post_read(int num_items);
};


} // namespace gr

#define CUDA_BUFFER_ARGS_H2D cuda_buffer::make, cuda_buffer_properties::make(cuda_buffer_type::H2D)
#define CUDA_BUFFER_ARGS_D2H cuda_buffer::make, cuda_buffer_properties::make(cuda_buffer_type::D2H)
#define CUDA_BUFFER_ARGS_D2D cuda_buffer::make, cuda_buffer_properties::make(cuda_buffer_type::D2D)
