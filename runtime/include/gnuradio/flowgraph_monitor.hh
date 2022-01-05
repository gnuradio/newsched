#pragma once


#include <gnuradio/concurrent_queue.hh>
#include <gnuradio/logging.hh>
#include <gnuradio/neighbor_interface.hh>
#include <string>
#include <thread>
#include <vector>
#include <gnuradio/fgm_proxy.hh>

namespace gr {

enum class fg_monitor_message_t { UNKNOWN, DONE, FLUSHED, KILL, START };

class scheduler;

/**
 * @brief Messages received by flowgraph_monitor
 *
 */
class fg_monitor_message
{
public:
    typedef std::shared_ptr<fg_monitor_message> sptr;
    static std::map<fg_monitor_message_t, std::string> string_map;
    static std::map<std::string, fg_monitor_message_t> rev_string_map;
    fg_monitor_message(fg_monitor_message_t type = fg_monitor_message_t::UNKNOWN,
                       int64_t schedid = -1,
                       int64_t blkid = -1)
        : _type(type), _blkid(blkid), _schedid(schedid)
    {
    }
    fg_monitor_message_t type() { return _type; }
    int64_t schedid() { return _schedid; }
    void set_schedid(int64_t id) { _schedid = id; }
    int64_t blkid() { return _blkid; }

    std::string to_string()
    {
        return fmt::format(
            "{{ type: {}, schedid: {}, blkid: {} }}", string_map[type()], schedid(), blkid());
    }

    static sptr from_string(const std::string& str)
    {
        auto opt_yaml = YAML::Load(str);

        auto typestr = opt_yaml["type"].as<std::string>("UNKNOWN");
        auto type = rev_string_map[typestr];
        auto schedid = opt_yaml["schedid"].as<int>(-1);
        auto blockid = opt_yaml["blkid"].as<int>(-1);

        return make(type, schedid, blockid);
    }

    static sptr make(fg_monitor_message_t type = fg_monitor_message_t::UNKNOWN,
                     int64_t schedid = -1,
                     int64_t blkid = -1)
    {
        return std::make_shared<fg_monitor_message>(type, schedid, blkid);
    }

private:
    fg_monitor_message_t _type;
    int64_t _blkid;
    int64_t _schedid;
};

typedef fg_monitor_message::sptr fg_monitor_message_sptr;

/**
 * @brief The flowgraph_monitor is responsible for tracking the start/stop status of
 * execution threads in the flowgraph
 *
 */
class flowgraph_monitor
{

public:
    flowgraph_monitor(std::vector<std::shared_ptr<scheduler>>& sched_ptrs,
                      std::vector<std::shared_ptr<fgm_proxy>>& proxy_ptrs)
        : d_schedulers(sched_ptrs),
          d_fgm_proxies(proxy_ptrs)
    {

    } // TODO: bound the queue size
    virtual ~flowgraph_monitor() {}

    virtual void push_message(fg_monitor_message_sptr msg) { msgq.push(msg); }
    void start();
    void stop() { push_message(fg_monitor_message::make(fg_monitor_message_t::KILL, 0, 0)); }

private:
    bool _monitor_thread_stopped = false;
    std::vector<std::shared_ptr<scheduler>> d_schedulers;
    std::vector<std::shared_ptr<fgm_proxy>> d_fgm_proxies;

protected:
    concurrent_queue<fg_monitor_message_sptr> msgq;
    virtual bool pop_message(fg_monitor_message_sptr& msg) { return msgq.pop(msg); }
    virtual void empty_queue() { msgq.clear(); }
};

typedef std::shared_ptr<flowgraph_monitor> flowgraph_monitor_sptr;

} // namespace gr
