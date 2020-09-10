#include <pmt/pmt.hpp>

namespace pmt
{

std::map<pmt_data_type_t, std::type_index> pmt_functions::pmt_type_index_map = {
    { pmt_data_type_t::FLOAT, typeid(float) },
    { pmt_data_type_t::DOUBLE, typeid(double) },
    { pmt_data_type_t::CFLOAT, typeid(gr_complex) },
    { pmt_data_type_t::CDOUBLE, typeid(gr_complexd) },
    { pmt_data_type_t::INT8, typeid(int8_t) },
    { pmt_data_type_t::INT16, typeid(int16_t) },
    { pmt_data_type_t::INT32, typeid(int32_t) },
    { pmt_data_type_t::INT64, typeid(int64_t) },
    { pmt_data_type_t::UINT8, typeid(uint8_t) },
    { pmt_data_type_t::UINT16, typeid(uint16_t) },
    { pmt_data_type_t::UINT32, typeid(uint32_t) },
    { pmt_data_type_t::UINT64, typeid(uint64_t) },
    { pmt_data_type_t::BOOL, typeid(bool) },
    { pmt_data_type_t::ENUM, typeid(int) }, //??
    { pmt_data_type_t::STRING, typeid(std::string) },
    { pmt_data_type_t::VOID, typeid(void) }
};

std::map<std::type_index, pmt_data_type_t> pmt_functions::pmt_index_type_map = {
    { std::type_index(typeid(float)), pmt_data_type_t::FLOAT },
    { std::type_index(typeid(double)), pmt_data_type_t::DOUBLE },
    { std::type_index(typeid(gr_complex)), pmt_data_type_t::CFLOAT },
    { std::type_index(typeid(gr_complexd)), pmt_data_type_t::CDOUBLE },
    { std::type_index(typeid(int8_t)), pmt_data_type_t::INT8 },
    { std::type_index(typeid(int16_t)), pmt_data_type_t::INT16 },
    { std::type_index(typeid(int32_t)), pmt_data_type_t::INT32 },
    { std::type_index(typeid(int64_t)), pmt_data_type_t::INT64 },
    { std::type_index(typeid(uint8_t)), pmt_data_type_t::UINT8 },
    { std::type_index(typeid(uint16_t)), pmt_data_type_t::UINT16 },
    { std::type_index(typeid(uint32_t)), pmt_data_type_t::UINT32 },
    { std::type_index(typeid(uint64_t)), pmt_data_type_t::UINT64 },
    { std::type_index(typeid(bool)), pmt_data_type_t::BOOL },
    { std::type_index(typeid(int)), pmt_data_type_t::ENUM }, //??
    { std::type_index(typeid(std::string)), pmt_data_type_t::STRING },
    { std::type_index(typeid(void)), pmt_data_type_t::VOID }
};

std::map<pmt_data_type_t, size_t> pmt_functions::pmt_type_size_map = {
    { pmt_data_type_t::FLOAT, sizeof(float) },
    { pmt_data_type_t::DOUBLE, sizeof(double) },
    { pmt_data_type_t::CFLOAT, sizeof(gr_complex) },
    { pmt_data_type_t::CDOUBLE, sizeof(gr_complexd) },
    { pmt_data_type_t::INT8, sizeof(int8_t) },
    { pmt_data_type_t::INT16, sizeof(int16_t) },
    { pmt_data_type_t::INT32, sizeof(int32_t) },
    { pmt_data_type_t::INT64, sizeof(int64_t) },
    { pmt_data_type_t::UINT8, sizeof(uint8_t) },
    { pmt_data_type_t::UINT16, sizeof(uint16_t) },
    { pmt_data_type_t::UINT32, sizeof(uint32_t) },
    { pmt_data_type_t::UINT64, sizeof(uint64_t) },
    { pmt_data_type_t::BOOL, sizeof(bool) },
    { pmt_data_type_t::ENUM, sizeof(int) }, //??
    { pmt_data_type_t::STRING, sizeof(std::string) },
    { pmt_data_type_t::VOID, sizeof(void*) }
};

size_t pmt_functions::pmt_size_info(pmt_data_type_t p)
{
    return pmt_type_size_map[p];
}
pmt_data_type_t pmt_functions::get_pmt_type_from_typeinfo(std::type_index t)
{
    return pmt_index_type_map[t];
}



}