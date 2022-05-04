#include <gnuradio/pdu.h>
#include <pmtf/string.hpp>
namespace gr {

template <>
void pdu<float>::set_data_type()
{
    _meta["vector_datatype"] = pmtf::string("rf32");
}

template <>
void pdu<int16_t>::set_data_type()
{
    _meta["vector_datatype"] = pmtf::string("ri16");
}

template class pdu<float>;
template class pdu<int16_t>;

} // namespace gr



template <>
pmtf::pmt::pmt<gr::pdu<float>>(const gr::pdu<float>& x)
{
    *this = x.get_pmt_buffer();
}
