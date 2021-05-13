
#include <pmt/pmtf_map.hpp>
#include <map>
#include <flatbuffers/flatbuffers.h>

namespace pmtf {

template <>
pmt_map<std::string>::pmt_map() : 
    pmt_base(Data::MapString)
{
    // Don't need anything here.
}

template <>
pmt_map<std::string>::pmt_map(const std::map<std::string, pmt_wrap>& val) : 
    pmt_base(Data::MapString), 
    _map(val) 
{
    // Don't need anything here.
}

template <>
pmt_map<std::string>::pmt_map(const uint8_t* buf, size_t size):
    pmt_base(Data::MapString)
{
    set_buffer(buf, size);
    // Possibly make this conversion lazy.
    auto pmt = GetSizePrefixedPmt(_buf);
    auto entries = pmt->data_as_MapString()->entries();
    for (size_t k=0; k<entries->size(); k++)
    {
        _map[entries->Get(k)->key()->str()] = pmt_base::from_pmt(entries->Get(k)->value());       
    }
}

template <>
flatbuffers::Offset<void> pmt_map<std::string>::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    throw std::runtime_error("This should not get called");
}

template <>
pmt_wrap& pmt_map<std::string>::operator[](const pmt_map::key_type& key)
{
    return _map[key];
}

template <>
void pmt_map<std::string>::fill_flatbuffer()
{
    _fbb.Reset();
    std::vector<flatbuffers::Offset<MapEntryString>> entries;

    for( auto& [key, val] : _map )
    {
        auto str = _fbb.CreateString(key.c_str());
        auto pmt_offset = val.ptr()->build(_fbb);
        entries.push_back(CreateMapEntryString(_fbb, str, pmt_offset ));
    }

    auto vec = _fbb.CreateVectorOfSortedTables(&entries);
    MapStringBuilder mb(_fbb);
    mb.add_entries(vec);
    _data = mb.Finish().Union();
    build();
}

template <>
void pmt_map<std::string>::serialize_setup()
{
    fill_flatbuffer();
}
}
