#pragma once

#include <gnuradio/blocklib/parameter_types.hpp>
#include <any>
#include <string>
#include <vector>

namespace gr {

enum class range_type_t { MIN_MAX, ACCEPTABLE, BLACKLIST };
enum class param_flags_t {
    NO_FLAGS = 0,
    MANDATORY = 1 << 0, // used to indicate that this parameter must be set (if params are removed from constructor)
    CONST = 1 << 1, // set at initialization, but cannot be set once a flowgraph is running
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


class block_param
{
public:
    block_param(const uint32_t id,
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
class typed_param : public block_param
{
public:
    typed_param(const uint32_t id,
                const std::string name,
                const T default_value,
                const std::vector<size_t> dims = std::vector<size_t>{ 1 })
        : block_param(id,
                      name,
                      parameter_functions::get_param_type_from_typeinfo(
                          std::type_index(typeid(T))),
                      dims),
                      _default_value(default_value)
    {
    }

    typed_param(block_param& b)
    : block_param(b)
    {
        _value = std::any_cast<T>(b.any_value());
    }

    void set_value(T val)
    {
        // do range checking

        _param_set = true;
        _value = val;
    }
    T value() {return _value; };

protected:
    T _default_value;
    T _value;
    value_check _range;
};

class parameter_config
{
private:
    std::vector<block_param> params;

public:
    size_t num() { return params.size(); }
    void add(block_param b) { params.push_back(b); }
    block_param get(const uint32_t id){
        auto pred = [id](block_param & item) {
            return item.id() == id;
        };
        std::vector<block_param>::iterator it = std::find_if(std::begin(params), std::end(params), pred);
        return *it;
    } 
    block_param get(const std::string name){
        auto pred = [name](block_param & item) {
            return item.name() == name;
        };
        std::vector<block_param>::iterator it = std::find_if(std::begin(params), std::end(params), pred);
        return *it;
    }
    void clear() { params.clear(); }
};

} // namespace gr