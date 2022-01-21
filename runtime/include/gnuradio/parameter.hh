#pragma once

#include <pmtf/wrap.hpp>
#include <pmtf/scalar.hpp>
#include <pmtf/vector.hpp>
#include <pmtf/string.hpp>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include <gnuradio/scheduler_message.hh>

namespace gr {

class param_action
{
protected:
    uint32_t _id;
    pmtf::pmt _pmt_value;
    uint64_t _at_sample;

public:
    typedef std::shared_ptr<param_action> sptr;
    static sptr
    make(uint32_t id, pmtf::pmt pmt_value = pmtf::pmt(), uint64_t at_sample = 0)
    {
        return std::make_shared<param_action>(id, pmt_value, at_sample);
    }
    param_action(uint32_t id, pmtf::pmt pmt_value, uint64_t at_sample)
        : _id(id), _pmt_value(pmt_value), _at_sample(at_sample)
    {
    }
    uint32_t id() const { return _id; }
    pmtf::pmt pmt_value() {
         return _pmt_value; 
    }
    void set_pmt_value(pmtf::pmt val) { _pmt_value = val; }
    uint64_t at_sample() { return _at_sample; }
    void set_at_sample(uint64_t val) { _at_sample = val; }
};

typedef std::shared_ptr<param_action> param_action_sptr;

typedef std::function<void(param_action_sptr)> param_action_complete_fcn;
class param_action_with_callback : public scheduler_message
{
public:
    param_action_with_callback(scheduler_message_t action_type,
                               nodeid_t block_id,
                               param_action_sptr param_action,
                               param_action_complete_fcn cb_fcn)
        : scheduler_message(action_type), _param_action(param_action), _cb_fcn(cb_fcn)
    {
        set_blkid(block_id);
    }
    param_action_sptr param_action() { return _param_action; }
    param_action_complete_fcn cb_fcn() { return _cb_fcn; }

private:
    param_action_sptr _param_action;
    param_action_complete_fcn _cb_fcn;
};

typedef std::queue<param_action_with_callback> param_action_queue;

class param_query_action : public param_action_with_callback
{
public:
    param_query_action(nodeid_t block_id,
                       param_action_sptr param_action,
                       param_action_complete_fcn cb_fcn)
        : param_action_with_callback(
              scheduler_message_t::PARAMETER_QUERY, block_id, param_action, cb_fcn)
    {
    }
};

class param_change_action : public param_action_with_callback
{
public:
    param_change_action(nodeid_t block_id,
                        param_action_sptr param_action,
                        param_action_complete_fcn cb_fcn)
        : param_action_with_callback(
              scheduler_message_t::PARAMETER_CHANGE, block_id, param_action, cb_fcn)
    {
    }
};

struct parameter_config
{
    std::map<std::string, pmtf::pmt> param_map;
    std::map<int, pmtf::pmt> param_map_int;
    size_t num() { return param_map.size(); }
    void add(const std::string& name, int id, const pmtf::pmt& b) { 
        param_map[name] = b; 
        param_map_int[id] = b;
        }
    pmtf::pmt& get(const std::string& name)
    {
        return param_map[name];
    }
    pmtf::pmt& get(int id)
    {
        return param_map_int[id];
    }
    void clear() { param_map.clear(); }
};

} // namespace gr
