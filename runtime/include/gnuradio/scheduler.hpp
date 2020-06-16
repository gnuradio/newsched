#pragma once

#include <gnuradio/flat_graph.hpp>
#include <memory>
namespace gr {
class scheduler : public std::enable_shared_from_this<scheduler>
{

public:
    scheduler() {};
    virtual ~scheduler();
    std::shared_ptr<scheduler> base() {return shared_from_this();}
    virtual void initialize(flat_graph_sptr fg) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void wait() = 0;
};

typedef std::shared_ptr<scheduler> scheduler_sptr;
} // namespace gr