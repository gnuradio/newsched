#include <gnuradio/blocklib/parameter_types.hpp>
#include <string>
#include <typeinfo>
#include <typeindex>

namespace gr {

enum class port_type_t {
    STREAM,
    PACKET,
    MESSAGE
}

enum class port_direction_t {
    INPUT,
    OUTPUT,
    BIDIRECTONAL //?? can it be done
}

class port
{
public:
    port(name,
         data_type = parameter_type_t::CFLOAT,
         port_type = port_type_t::STREAM,
         dims = std::vector<int>(),
         multiplicity = 1)
        : _name(name),
        _data_type(data_type),
        _port_type(port_type),
        _dims(dims),
        _multiplicity(multiplicity)
    {
        _type_info = param_type_info(_data_type); // might not be needed
        _data_size = param_size_info(_data_type);
    }

protected:
    std::string _name;
    std::string _short_name;
    port_type_t _port_type;
    parameter_type_t _data_type;
    std::type_index _type_info;
    int _multiplicity;      // port can be replicated as in grc
    std::vector<int> _dims; // allow for matrices to be sent naturally across ports
    // empty dims refers to a scalar, dims=[n] same as vlen=n
    size_t _data_size;
};

} // namespace gr