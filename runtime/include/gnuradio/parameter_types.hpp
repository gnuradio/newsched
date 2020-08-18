#ifndef GR_PARAMETER_TYPES_HPP
#define GR_PARAMETER_TYPES_HPP

#include <gnuradio/types.hpp>
#include <map>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace gr {

enum class param_type_t {
    UNTYPED,
    FLOAT,
    DOUBLE,
    CFLOAT,
    CDOUBLE,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL,
    ENUM,
    STRING,
    VOID
};


class parameter_functions
{
private:
    static std::map<param_type_t, std::type_index> param_type_index_map;

    static std::map<param_type_t, size_t> param_type_size_map;

    static std::map<std::type_index, param_type_t> param_index_type_map;

    // static const std::type_index
    // param_type_info(param_type_t p)
    // {
    //     return param_type_index_map[p];
    // }

public:
    static size_t param_size_info(param_type_t p);
    static param_type_t get_param_type_from_typeinfo(std::type_index t);
};


} // namespace gr

#endif