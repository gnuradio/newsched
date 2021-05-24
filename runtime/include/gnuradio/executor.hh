#include <gnuradio/logging.hh>

namespace gr {

enum class executor_state { WORKING, DONE, FLUSHED, EXIT };
enum class executor_iteration_status {
    READY,           // We made progress; everything's cool.
    READY_NO_OUTPUT, // We consumed some input, but produced no output.
    BLKD_IN,         // no progress; we're blocked waiting for input data.
    BLKD_OUT,        // no progress; we're blocked waiting for output buffer space.
    DONE,            // we're done; don't call me again.
};

class executor
{
protected:
    std::string _name;
    logger_sptr _logger;
    logger_sptr _debug_logger;

public:
    executor(const std::string& name) : _name(name)
    {
        _logger = logging::get_logger(_name, "default");
        _debug_logger = logging::get_logger(_name + "_dbg", "debug");
    }
};

} // namespace gr