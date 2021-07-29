#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer_sm.hh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
enum class cuda_buffer_sm_type { D2D, H2D, D2H, UNKNOWN };

class cuda_buffer_sm : public buffer_sm
{
private:
    uint8_t* _host_buffer;
    uint8_t* _device_buffer;
    cuda_buffer_sm_type _type = cuda_buffer_sm_type::UNKNOWN;
    cudaStream_t stream;

public:
    typedef std::shared_ptr<cuda_buffer_sm> sptr;
    cuda_buffer_sm(size_t num_items,
                   size_t item_size,
                   cuda_buffer_sm_type type,
                   std::shared_ptr<buffer_properties> buf_properties);
    ~cuda_buffer_sm();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);

    void* read_ptr(size_t read_index);
    void* write_ptr();
    cuda_buffer_sm_type type() { return _type; }

    virtual void post_write(int num_items);

    virtual std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props);

    static void* cuda_memcpy(void* dest, const void* src, std::size_t count);
    static void* cuda_memmove(void* dest, const void* src, std::size_t count);

    virtual bool output_blocked_callback(bool force = false) override
    {
        switch(_type)
        {
            case cuda_buffer_sm_type::H2D:
                return output_blocked_callback_logic(force, std::memmove);
            case cuda_buffer_sm_type::D2D:
            case cuda_buffer_sm_type::D2H:
                return output_blocked_callback_logic(force, cuda_memmove);
            default:
                return false;
        }
    }
};

class cuda_buffer_sm_reader : public buffer_sm_reader
{
private:
    // logger_sptr _logger;
    // logger_sptr _debug_logger;

    std::shared_ptr<cuda_buffer_sm> _cuda_buffer_sm;

public:
    cuda_buffer_sm_reader(std::shared_ptr<cuda_buffer_sm> buffer,
                          std::shared_ptr<buffer_properties> buf_props,
                          size_t read_index)
        : buffer_sm_reader(buffer, buf_props, read_index)
    {
        _cuda_buffer_sm = buffer;
        // _logger = logging::get_logger("cudabuffer_sm_reader", "default");
        // _debug_logger = logging::get_logger("cudabuffer_sm_reader_dbg", "debug");
    }

    // virtual void post_read(int num_items);

    virtual bool input_blocked_callback(size_t items_required) override
    {
        // Only singly mapped buffers need to do anything with this callback
        // std::scoped_lock guard(*(_buffer->mutex()));
        std::lock_guard<std::mutex> guard(*(_buffer->mutex()));

        auto items_avail = items_available();

        // GR_LOG_DEBUG(_debug_logger,
        //              "input_blocked_callback: items_avail {}, _read_index {}, "
        //              "_write_index {}, items_required {}",
        //              items_avail,
        //              _read_index,
        //              _buffer->write_index(),
        //              items_required);

        // GR_LOG_DEBUG(_debug_logger,
        //              "input_blocked_callback: total_written {}, total_read {}",
        //              _buffer->total_written(),
        //              total_read());


        // Maybe adjust read pointers from min read index?
        // This would mean that *all* readers must be > (passed) the write index
        if (items_avail < items_required && _buffer->write_index() < read_index()) {
            // GR_LOG_DEBUG(_debug_logger, "Calling adjust_buffer_data ");

            switch (_cuda_buffer_sm->type()) {
            case cuda_buffer_sm_type::H2D:
            case cuda_buffer_sm_type::D2D:
                return _buffer_sm->adjust_buffer_data(cuda_buffer_sm::cuda_memcpy,
                                                      cuda_buffer_sm::cuda_memmove);
            case cuda_buffer_sm_type::D2H:
                return _buffer_sm->adjust_buffer_data(std::memcpy, std::memmove);
            default:
                return false;
            }
        }

        return false;
    }
};

class cuda_buffer_sm_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    cuda_buffer_sm_properties(cuda_buffer_sm_type buffer_type_)
        : buffer_properties(), _buffer_type(buffer_type_)
    {
        _bff = cuda_buffer_sm::make;
    }
    cuda_buffer_sm_type buffer_type() { return _buffer_type; }
    static std::shared_ptr<buffer_properties>
    make(cuda_buffer_sm_type buffer_type_ = cuda_buffer_sm_type::D2D)
    {
        return std::static_pointer_cast<buffer_properties>(
            std::make_shared<cuda_buffer_sm_properties>(buffer_type_));
    }

private:
    cuda_buffer_sm_type _buffer_type;
};


} // namespace gr

#define CUDA_BUFFER_SM_ARGS_H2D cuda_buffer_sm_properties::make(cuda_buffer_sm_type::H2D)
#define CUDA_BUFFER_SM_ARGS_D2H cuda_buffer_sm_properties::make(cuda_buffer_sm_type::D2H)
#define CUDA_BUFFER_SM_ARGS_D2D cuda_buffer_sm_properties::make(cuda_buffer_sm_type::D2D)
