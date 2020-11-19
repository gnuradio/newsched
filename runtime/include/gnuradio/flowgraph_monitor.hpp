#pragma once


#include <gnuradio/concurrent_queue.hpp>
#include <gnuradio/logging.hpp>
#include <thread>
#include <vector>

namespace gr {


enum class fg_monitor_message_t { UNKNOWN, DONE, FLUSHED, KILL };

class scheduler;

class fg_monitor_message
{
public:
    fg_monitor_message(fg_monitor_message_t type = fg_monitor_message_t::UNKNOWN,
                       int64_t schedid = -1,
                       int64_t blkid = -1)
        : _type(type), _blkid(blkid), _schedid(schedid)
    {
    }
    fg_monitor_message_t type() { return _type; }
    int64_t schedid() { return _schedid; }
    int64_t blkid() { return _blkid; }

private:
    fg_monitor_message_t _type;
    int64_t _blkid;
    int64_t _schedid;
};

/**
 * @brief The flowgraph_monitor is responsible for tracking the start/stop status of
 * execution threads in the flowgraph
 *
 */
class flowgraph_monitor
{

public:
    flowgraph_monitor(std::vector<std::shared_ptr<scheduler>>& sched_ptrs)
        : d_schedulers(sched_ptrs)
    {

    } // TODO: bound the queue size
    virtual ~flowgraph_monitor() {}

    virtual void push_message(fg_monitor_message msg) { msgq.push(msg); }
    void start();
    void stop() { push_message(fg_monitor_message(fg_monitor_message_t::KILL, 0, 0)); }

    /**
     * @brief Replace the specified scheduler pointer with a group of other scheduler
     * pointers
     *
     * For Hierarchical schedulers, notify the Flowgraph Monitor that sub-schedulers will
     * be doing the work This could be the case for a multi-threaded scheduler that breaks
     * its graph into single threaded scheduler graphs.  In this case, the flowgraph
     * monitor needs to be notified to monitor those threads instead
     *
     * @param original the scheduler pointer that was originally given to the flowgraph
     * monitor when the flowgraph was first partitioned
     * @param replacements the set of replacement scheduler pointers to be tracked instead
     * @return true if the original scheduler was found
     * @return false if the original scheduler was not found
     */
    bool replace_scheduler(std::shared_ptr<scheduler> original,
                           const std::vector<std::shared_ptr<scheduler>> replacements);

private:
    bool _monitor_thread_stopped = false;
    std::vector<std::shared_ptr<scheduler>> d_schedulers;

protected:
    concurrent_queue<fg_monitor_message> msgq;
    virtual bool pop_message(fg_monitor_message& msg) { return msgq.pop(msg); }
    virtual void empty_queue() { msgq.clear(); }
};

typedef std::shared_ptr<flowgraph_monitor> flowgraph_monitor_sptr;

} // namespace gr
