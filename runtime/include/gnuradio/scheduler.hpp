#include <gnuradio/flowgraph.hpp>
namespace gr {
class scheduler
{

public:
    scheduler(flowgraph_sptr fg) {};
    virtual ~scheduler();
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;
};
} // namespace gr