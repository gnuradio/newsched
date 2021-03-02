#pragma once

#include <pmt/pmtf.hpp>
#include <pmt/pmtf_scalar.hpp>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include <gnuradio/scheduler_message.hpp>

namespace gr {

enum class range_type_t { MIN_MAX, ACCEPTABLE, BLACKLIST };
enum class param_flags_t {
    NO_FLAGS = 0,
    MANDATORY = 1 << 0, // used to indicate that this parameter must be set (if params are
                        // removed from constructor)
    CONST =
        1 << 1, // set at initialization, but cannot be set once a flowgraph is running
};

class value_check
{
    param_type_t _type;
    int _dim = 0;

public:
};

template <class T>
class continuous_range : value_check
{
protected:
    T _min_value;
    T _max_value;

public:
    void set_min_value(T);
    void set_max_value(T);
    T min_value();
    T max_value();
};

template <class T>
class fixed_set : value_check
{
    std::vector<T> acceptable_values;
};

template <class T>
class blacklist_set : value_check
{
    std::vector<T> blacklist_values;
};

class param
{
public:
    typedef std::shared_ptr<param> sptr;
    static sptr
    make(const uint32_t id, const std::string name, pmtf::pmt_sptr pmt_value = nullptr)
    {
        return std::make_shared<param>(id, name, pmt_value);
    }
    param(const uint32_t id, const std::string name, pmtf::pmt_sptr pmt_value)
        : _id(id), _name(name), _pmt_value(pmt_value)
    {
    }
    virtual ~param() {}
    std::string to_string() { return ""; };

    const auto id() { return _id; }
    const auto name() { return _name; }
    const auto pmt_value() { return _pmt_value; }

    void set_pmt_value(pmtf::pmt_sptr val) { _pmt_value = val; };


protected:
    const uint32_t _id;
    const std::string _name;
    param_type_t _type; // should be some sort of typeinfo, but worst case enum or string
    pmtf::pmt_sptr _pmt_value;
};

template <class T>
class scalar_param : public param
{
public:
    typedef std::shared_ptr<scalar_param> sptr;
    static sptr make(const uint32_t id, const std::string name, T value)
    {
        return std::make_shared<scalar_param<T>>(id, name, value);
    }
    scalar_param<T>(const uint32_t id, const std::string name, T value)
        : param(id, name, pmtf::pmt_scalar<T>::make(value)), _value(value)
    {
    }
    virtual ~scalar_param<T>() {}

    void set_value(T val)
    {
        std::static_pointer_cast<pmtf::pmt_scalar<T>>(pmt_value())->set_pmt_value(val);
    }
    T value()
    {
        return std::static_pointer_cast<pmtf::pmt_scalar<T>>(pmt_value())->value();
    }

protected:
    T _value;
};

#define DECLARE_SCALAR_PARAM(type, name)                                     \
public:                                                                      \
    void set_##name(type name)                                               \
    {                                                                        \
        return request_parameter_change(params::id_##name,                   \
                                        pmtf::pmt_scalar<type>::make(name)); \
    }                                                                        \
    type name()                                                              \
    {                                                                        \
        return std::static_pointer_cast<pmtf::pmt_scalar<type>>(             \
                   request_parameter_query(params::id_##name))               \
            ->value();                                                       \
    }                                                                        \
                                                                             \
private:                                                                     \
    typename scalar_param<type>::sptr d_##name;
    // typename scalar_param<type> d_##name;

class param_action
{
protected:
    uint32_t _id;
    pmtf::pmt_sptr _pmt_value;
    uint64_t _at_sample;

public:
    typedef std::shared_ptr<param_action> sptr;
    static sptr
    make(uint32_t id, pmtf::pmt_sptr pmt_value = nullptr, uint64_t at_sample = 0)
    {
        return std::make_shared<param_action>(id, pmt_value, at_sample);
    }
    param_action(uint32_t id, pmtf::pmt_sptr pmt_value, uint64_t at_sample)
        : _id(id), _pmt_value(pmt_value), _at_sample(at_sample)
    {
    }
    uint32_t id() const { return _id; }
    pmtf::pmt_sptr pmt_value() { return _pmt_value; }
    void set_pmt_value(pmtf::pmt_sptr val) { _pmt_value = val; }
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

typedef std::shared_ptr<param> param_sptr;

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

class parameter_config
{
private:
    std::vector<param_sptr> params;

public:
    size_t num() { return params.size(); }
    void add(param_sptr b) { params.push_back(b); }
    param_sptr get(const uint32_t id)
    {
        auto pred = [id](param_sptr item) { return item->id() == id; };
        std::vector<param_sptr>::iterator it =
            std::find_if(std::begin(params), std::end(params), pred);

        if (it == std::end(params))
            throw std::runtime_error(
                "parameter not defined for this block"); // TODO logging

        return *it;
    }
    param_sptr get(const std::string& name)
    {
        auto pred = [name](param_sptr item) { return item->name() == name; };
        std::vector<param_sptr>::iterator it =
            std::find_if(std::begin(params), std::end(params), pred);

        if (it == std::end(params))
            throw std::runtime_error(
                "parameter not defined for this block"); // TODO logging

        return *it;
    }
    void clear() { params.clear(); }
};

} // namespace gr
