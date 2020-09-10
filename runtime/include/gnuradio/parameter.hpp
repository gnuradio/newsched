#pragma once

#include <gnuradio/parameter_types.hpp>
#include <any>
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


class param_base
{
public:
    param_base(const uint32_t id,
               const std::string name,
               const param_type_t& type,
               const std::vector<size_t> dims,
               std::any any_value)
        : _id(id), _name(name), _type(type), _dims(dims), _any_value(any_value)
    {
    }

    std::string to_string() { return ""; };
    // std::any to_any() { return std::make_any<T>(value()); }
    const uint32_t id() { return _id; }
    const std::string name() { return _name; }
    const std::any any_value() { return _any_value; }
    const param_type_t type() { return _type; }

    virtual void set_value(const std::any& val) = 0;


protected:
    const uint32_t _id;
    const std::string _name;
    param_type_t _type; // should be some sort of typeinfo, but worst case enum or string
    std::vector<size_t> _dims;
    bool _param_set = false;
    std::any _any_value;
};

template <class T>
class param : public param_base
{
public:
    typedef std::shared_ptr<param> sptr;
    static sptr make(const uint32_t id,
                     const std::string name,
                     const T default_value,
                     T* value_ptr,
                     const std::vector<size_t> dims = std::vector<size_t>{ 1 })
    {
        return std::make_shared<param<T>>(
            param<T>(id, name, default_value, value_ptr, dims));
    }
    param(const uint32_t id,
          const std::string name,
          const T default_value,
          T* value_ptr,
          const std::vector<size_t> dims = std::vector<size_t>{ 1 })
        : param_base(id,
                     name,
                     parameter_functions::get_param_type_from_typeinfo(
                         std::type_index(typeid(T))),
                     dims,
                     std::make_any<T>(default_value)),
          _default_value(default_value),
          _value_ptr(value_ptr)
    {
        set_value(default_value);
    }

    param(param_base& b) : param_base(b) { _value = std::any_cast<T>(b.any_value()); }

    void set_value(T val)
    {
        // TODO: do range checking
        // FIXME - don't use raw pointers
        _param_set = true;
        *_value_ptr = val;
    }
    void set_value(const std::any& val)
    {
        _any_value = val;
        set_value(std::any_cast<T>(val));
    }
    T value() { return *_value_ptr; };

protected:
    T* _value_ptr; // TODO: use smart pointer
    T _default_value;
    T _value;
    value_check _range;
};

class param_action_base
{
protected:
    uint32_t _id;
    std::any _any_value;
    uint64_t _at_sample;

public:
    param_action_base(uint32_t id, std::any any_value, uint64_t at_sample)
        : _id(id), _any_value(any_value), _at_sample(at_sample)
    {
    }
    uint32_t id() const { return _id; }
    std::any any_value() { return _any_value; }
    void set_any_value(std::any val) { _any_value = val; }
    uint64_t at_sample() { return _at_sample; }
    void set_at_sample(uint64_t val) { _at_sample = val; }
};

typedef std::shared_ptr<param_action_base> param_action_sptr;

template <class T>
class param_action : public param_action_base
{
protected:
    T _new_value;

public:
    typedef std::shared_ptr<param_action<T>> sptr;

    static sptr make(uint32_t id)
    {
        return std::make_shared<param_action<T>>(param_action<T>(id));
    }

    static sptr make(uint32_t id, T new_value, uint64_t at_sample)
    {
        return std::make_shared<param_action<T>>(
            param_action<T>(id, new_value, at_sample));
    }

    // Constructor where the current value is "don't care"
    param_action(uint32_t id) : param_action_base(id, std::any(), 0) {}

    param_action(uint32_t id, T new_value, uint64_t at_sample)
        : param_action_base(id, std::make_any<T>(new_value), at_sample),
          _new_value(new_value)
    {
    }
    param_action(param_action_base& b)
        : param_action_base(b.id(), b.any_value(), b.at_sample())
    {
        _new_value = std::any_cast<T>(b.any_value());
    }

    T new_value() { return std::any_cast<T>(_any_value); }
};

typedef std::function<void(param_action_sptr)> param_action_complete_fcn;
class param_action_base_with_callback : public scheduler_message
{
public:
    param_action_base_with_callback(scheduler_message_t action_type,
                                    nodeid_t block_id,
                                    param_action_sptr param_action,
                                    param_action_complete_fcn cb_fcn)
        : scheduler_message(action_type), _block_id(block_id), _param_action(param_action), _cb_fcn(cb_fcn)
    {
    }
    nodeid_t block_id() { return _block_id; }
    param_action_sptr param_action() { return _param_action; }
    param_action_complete_fcn cb_fcn() { return _cb_fcn; }
private:
    nodeid_t _block_id;
    param_action_sptr _param_action;
    param_action_complete_fcn _cb_fcn;
};

typedef std::queue<param_action_base_with_callback> param_action_queue;

typedef std::shared_ptr<param_base> param_sptr;

class param_query_action : public param_action_base_with_callback
{
public:
    param_query_action(nodeid_t block_id,
                       param_action_sptr param_action,
                       param_action_complete_fcn cb_fcn) :
          param_action_base_with_callback(scheduler_message_t::PARAMETER_QUERY, block_id, param_action, cb_fcn)
    {
    }
};

class param_change_action : public param_action_base_with_callback
{
public:
    param_change_action(nodeid_t block_id,
                       param_action_sptr param_action,
                       param_action_complete_fcn cb_fcn) :
          param_action_base_with_callback(scheduler_message_t::PARAMETER_CHANGE, block_id, param_action, cb_fcn)
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
    param_sptr get(const std::string name)
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