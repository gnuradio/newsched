#pragma once

#include <gnuradio/blocklib/parameter_types.hpp>
#include <any>
#include <string>
#include <vector>

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
                const std::vector<size_t> dims)
        : _id(id), _name(name), _type(type), _dims(dims)
    {
    }

    std::string to_string() { return ""; };
    // std::any to_any() { return std::make_any<T>(value()); }
    const uint32_t id() { return _id; }
    const std::string name() { return _name; }
    const std::any any_value() { return _any_value; }

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
    param(const uint32_t id,
                const std::string name,
                const T default_value,
                const std::vector<size_t> dims = std::vector<size_t>{ 1 })
        : param_base(id,
                      name,
                      parameter_functions::get_param_type_from_typeinfo(
                          std::type_index(typeid(T))),
                      dims),
          _default_value(default_value)
    {
    }

    param(param_base& b) : param_base(b)
    {
        _value = std::any_cast<T>(b.any_value());
    }

    void set_value(T val)
    {
        // do range checking

        _param_set = true;
        _value = val;
    }
    T value() { return _value; };

protected:
    T _default_value;
    T _value;
    value_check _range;
};

class param_change_base
{
protected:
    uint32_t _id;
    std::any _any_value;
    uint64_t _at_sample;

public:
    param_change_base(uint32_t id, std::any any_value, uint64_t at_sample)
        : _id(id), _any_value(any_value), _at_sample(at_sample)
    {
    }
    uint32_t id() { return _id; }
    std::any any_value() { return _any_value; }
    uint64_t at_sample() { return _at_sample; }
};

template <class T>
class param_change : public param_change_base
{
protected:
    T _new_value;

public:
    param_change(uint32_t id, T new_value, uint64_t at_sample)
        : param_change_base(id, std::make_any<T>(new_value), at_sample),
        _new_value(new_value)
    {
    }
    param_change(param_change_base& b) : param_change_base(b.id(), b.any_value(), b.at_sample())
    {
        _new_value = std::any_cast<T>(b.any_value());

    }

    uint32_t new_value() { return _new_value; }
};

class parameter_config
{
private:
    std::vector<param_base> params;

public:
    size_t num() { return params.size(); }
    void add(param_base b) { params.push_back(b); }
    param_base get(const uint32_t id)
    {
        auto pred = [id](param_base& item) { return item.id() == id; };
        std::vector<param_base>::iterator it =
            std::find_if(std::begin(params), std::end(params), pred);
        return *it;
    }
    param_base get(const std::string name)
    {
        auto pred = [name](param_base& item) { return item.name() == name; };
        std::vector<param_base>::iterator it =
            std::find_if(std::begin(params), std::end(params), pred);
        return *it;
    }
    void clear() { params.clear(); }
};

} // namespace gr