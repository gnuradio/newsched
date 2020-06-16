#include <gnuradio/blocklib/types.hpp>
#include <map>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace gr {

enum class param_type_t {
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

std::map<param_type_t, std::type_index>
    param_type_index_map = { { param_type_t::FLOAT, typeid(float) },
                             { param_type_t::DOUBLE, typeid(double) },
                             { param_type_t::CFLOAT, typeid(gr_complex) },
                             { param_type_t::CDOUBLE, typeid(gr_complexd) },
                             { param_type_t::INT8, typeid(int8_t) },
                             { param_type_t::INT16, typeid(int16_t) },
                             { param_type_t::INT32, typeid(int32_t) },
                             { param_type_t::INT64, typeid(int64_t) },
                             { param_type_t::UINT8, typeid(uint8_t) },
                             { param_type_t::UINT16, typeid(uint16_t) },
                             { param_type_t::UINT32, typeid(uint32_t) },
                             { param_type_t::UINT64, typeid(uint64_t) },
                             { param_type_t::BOOL, typeid(bool) },
                             { param_type_t::ENUM, typeid(int) }, //??
                             { param_type_t::STRING, typeid(std::string) },
                             { param_type_t::VOID, typeid(void) } }

std::map<param_type_t, size_t>
    param_type_size_map = { { param_type_t::FLOAT, sizeof(float) },
                             { param_type_t::DOUBLE, sizeof(double) },
                             { param_type_t::CFLOAT, sizeof(gr_complex) },
                             { param_type_t::CDOUBLE, sizeof(gr_complexd) },
                             { param_type_t::INT8, sizeof(int8_t) },
                             { param_type_t::INT16, sizeof(int16_t) },
                             { param_type_t::INT32, sizeof(int32_t) },
                             { param_type_t::INT64, sizeof(int64_t) },
                             { param_type_t::UINT8, sizeof(uint8_t) },
                             { param_type_t::UINT16, sizeof(uint16_t) },
                             { param_type_t::UINT32, sizeof(uint32_t) },
                             { param_type_t::UINT64, sizeof(uint64_t) },
                             { param_type_t::BOOL, sizeof(bool) },
                             { param_type_t::ENUM, sizeof(int) }, //??
                             { param_type_t::STRING, sizeof(std::string) },
                             { param_type_t::VOID, sizeof(void) } }

std::type_index
param_type_info(param_type_t p)
{
    return param_type_index_map(p);
}

size_t
param_type_info(param_type_t p)
{
    return param_type_size_map(p);
}

} // namespace gr