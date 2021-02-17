#include <pmt/pmtf_map.hpp>
#include <map>
#include <flatbuffers/flatbuffers.h>

namespace pmtf {

template <>
flatbuffers::Offset<void> pmt_map<std::string>::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    throw std::runtime_error("This should not get called");
}

template <>
std::map<std::string, pmt_sptr> pmt_map<std::string>::value() const
{
    std::map<std::string, pmt_sptr> ret;
    auto pmt = GetSizePrefixedPmt(_buf);
    auto entries = pmt->data_as_MapString()->entries();
    for (size_t k=0; k<entries->size(); k++)
    {
        ret[entries->Get(k)->key()->str()] = pmt_base::from_pmt(entries->Get(k)->value());       
    }

    return ret;
}

template <>
void pmt_map<std::string>::set_value(const std::map<std::string,pmt_sptr>& map_in)
{
    _fbb.Reset();
    std::vector<flatbuffers::Offset<MapEntryString>> entries;

    for( auto const& [key, val] : map_in )
    {
        auto str = _fbb.CreateString(key.c_str());
        auto pmt_offset = val->build(_fbb);
        entries.push_back(CreateMapEntryString(_fbb, str, pmt_offset ));
    }

    auto vec = _fbb.CreateVectorOfSortedTables(&entries);
    MapStringBuilder mb(_fbb);
    mb.add_entries(vec);
    _data = mb.Finish().Union();
    build();
}

template <>
pmt_map<std::string>::pmt_map(const std::map<std::string,pmt_sptr>& val)
    : pmt_base(Data::MapString)
{
    set_value(val);
}

template <>
pmt_map<std::string>::pmt_map(const uint8_t* buf, size_t size) 
    : pmt_base(Data::MapString)
{
    set_buffer(buf, size);
}

template <>
pmt_sptr pmt_map<std::string>::ref(const std::string& str)
{
    auto pmt = GetSizePrefixedPmt(_buf);
    auto entries = pmt->data_as_MapString()->entries();
    auto ptr = entries->LookupByKey(str.c_str()); // should return flatbuffers::Offset<pmtf::MapEntryString>
    if (ptr == nullptr)
    {
        throw std::runtime_error("Map key not found");
    }

    auto val = ptr->value();

    return pmt_base::from_pmt(val);
}

template <>
void pmt_map<std::string>::set(const std::string& k, pmt_sptr v)
{
    auto entire_map = value();

    // checking has key and such ... TODO

    entire_map[k] = v;

    set_value(entire_map);
}


// template <>
// std::map<std::string,pmt_sptr> pmt_map<std::string>::value() const
// {
//     auto pmt = GetSizePrefixedPmt(_buf);
//     auto fb_vec = pmt->data_as_VectorInt32()->value();
//     // _value.assign(fb_vec->begin(), fb_vec->end());
//     std::vector<int32_t> ret(fb_vec->begin(), fb_vec->end());
//     return ret;
// }


} // namespace pmtf
