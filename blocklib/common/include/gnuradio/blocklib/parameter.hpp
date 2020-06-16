#pragma once

#include <string>
#include <vector>
#include <any>
#include <gnuradio/blocklib/parameter_types.hpp>

namespace gr
{

enum class range_type_t { MIN_MAX, ACCEPTABLE, BLACKLIST };

class value_check
{
    parameter_type_t _type;
    int _dim = 0;

    public:
}   

template <class T>
class continuous_range : value_check
{
    protected:
        T min_value;
        T max_value;

    public:
        void set_min_value(T);
        void set_max_value(T);
        T min_value();
        T max_value();
};

template <class T>
class fixed_set : value_check{
    std::vector<T> acceptable_values;
};

template <class T>
class blacklist_set : value_check{
    std::vector<T> blacklist_values;
};


class block_parameter
{
public:
    block_parameter(uint32_t id,
                    std::string& name,
                    std::string& short_name,
                    param_type_t& type,
                    int dim)
        : _id(id), _name(name), _short_name(short_name), _type(type), _dim(dim)
    {
    }

    std::string to_string()
    {
        return "";
    };
    std::any to_any()
    {
        return std::make_any<T>(value());
    }

protected:
    uint32_t _id;
    std::string _name;
    std::string _short_name;
    param_type_t _type; // should be some sort of typeinfo, but worst case enum or string
    int _dim;
};

template <class T>
class typed_parameter : block_parameter
{
    public:
        set_value(T val);
        T value();

    protected:
        T _value;
        parameter_range<T> _range;
};

class parameter_config
{
private:
    std::vector<std::any> params;

public:
    size_t num_parameters() { return params.size(); }
    void add_parameter(std::any b) { params.push_back(b); }
    std::any get_parameter(uint32_t id){

    }; // by name or index
    std::any get_parameter(std::string name){

    }; // by name or index
    void clear() { params.clear(); }
};

}