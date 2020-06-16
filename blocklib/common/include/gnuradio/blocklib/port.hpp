#include <gnuradio/blocklib/parameter_types.hpp>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace gr {

enum class port_type_t { STREAM, PACKET, MESSAGE };

enum class port_direction_t {
    INPUT,
    OUTPUT,
    BIDIRECTONAL //?? can it be done
};

class port_base
{
protected:
    std::string _name;
    port_direction_t _direction;
    std::string _short_name;
    param_type_t _data_type;
    port_type_t _port_type;
    // std::type_index _type_info;
    std::vector<size_t> _dims; // allow for matrices to be sent naturally across ports
    // empty dims refers to a scalar, dims=[n] same as vlen=n
    int _multiplicity; // port can be replicated as in grc
    size_t _data_size;

public:
    port_base(const std::string& name,
         const port_direction_t direction,
         const param_type_t data_type = param_type_t::CFLOAT,
         //  const std::type_index T
         const port_type_t port_type = port_type_t::STREAM,
         const std::vector<size_t>& dims = std::vector<size_t>(),
         const int multiplicity = 1)
        : _name(name),
          _direction(direction),
          _data_type(data_type),
          _port_type(port_type),
          _dims(dims),
          _multiplicity(multiplicity)
    {
        // _type_info = param_type_info(_data_type); // might not be needed
        _data_size = parameter_functions::param_size_info(_data_type);
    }

    std::string name() { return _name; }
    port_type_t port_type() { return _port_type; }
    port_direction_t port_direction() { return _direction; }
};

template <class T>
class port : public port_base
{
public:
    port(const std::string& name,
               const port_direction_t direction,
               const port_type_t port_type = port_type_t::STREAM,
               const std::vector<size_t>& dims = std::vector<size_t>(),
               const int multiplicity = 1)
        : port_base(name,
               direction,
               parameter_functions::get_param_type_from_typeinfo(
                   std::type_index(typeid(T))),
               port_type,
               dims,
               multiplicity)
    {
    }
};

} // namespace gr