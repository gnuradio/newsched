#include <pmt/pmtf_string.hpp>
#include <map>

namespace pmtf {

flatbuffers::Offset<void> pmt_string::rebuild_data(flatbuffers::FlatBufferBuilder& fbb)
{
    // fbb.Reset();
    return CreatePmtStringDirect(fbb, value().c_str()).Union();
}


void pmt_string::set_value(const std::string& val)
{
    _data = CreatePmtStringDirect(_fbb, val.c_str()).Union();
    build();
}

pmt_string::pmt_string(const std::string& val)
    : pmt_base(Data::PmtString)
{
    set_value(val);
}

pmt_string::pmt_string(const uint8_t *buf)
    : pmt_base(Data::PmtString)
{
    auto data = GetPmt(buf)->data_as_PmtString()->value();
    set_value(*((const std::string*)data));
}

pmt_string::pmt_string(const pmtf::Pmt* fb_pmt)
    : pmt_base(Data::PmtString)
{
    auto data = fb_pmt->data_as_PmtString()->value();
    set_value(*((const std::string*)data));
}

std::string pmt_string::value() const
{
    auto pmt = GetSizePrefixedPmt(_fbb.GetBufferPointer());
    return std::string(pmt->data_as_PmtString()->value()->str());
}


} // namespace pmtf
