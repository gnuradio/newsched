#pragma once

#include <gnuradio/blocklib/parameter_types.hpp>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace gr {

class node;  // forward declaration for storing parent pointer

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
    std::string _alias;
    // std::shared_ptr<node> parent = nullptr;
    port_direction_t _direction;
    std::string _short_name;
    param_type_t _data_type;
    port_type_t _port_type;
    int _index = -1; // how does this get set??
    // std::type_index _type_info;
    std::vector<size_t> _dims; // allow for matrices to be sent naturally across ports
    // empty dims refers to a scalar, dims=[n] same as vlen=n
    int _multiplicity; // port can be replicated as in grc
    size_t _data_size;
    size_t _itemsize; // data size across all dims

public:
    typedef std::shared_ptr<port_base> sptr;
    port_base(const std::string& name,
        //  std::shared_ptr<node> parent,
         const port_direction_t direction,
         const param_type_t data_type = param_type_t::CFLOAT,
         //  const std::type_index T
         const port_type_t port_type = port_type_t::STREAM,
         const std::vector<size_t>& dims = std::vector<size_t>{1},
         const int multiplicity = 1)
        : _name(name),
        //   _parent(parent),
          _direction(direction),
          _data_type(data_type),
          _port_type(port_type),
          _dims(dims),
          _multiplicity(multiplicity)
    {
        // _type_info = param_type_info(_data_type); // might not be needed
        _data_size = parameter_functions::param_size_info(_data_type);
        _itemsize = _data_size;
        for (auto d : _dims)
            _itemsize *= d;
    }

    std::string name() { return _name; }
    std::string alias() { return _alias; }
    void set_alias(const std::string& alias) { _alias = alias; }
    void set_index(int val) { _index = val; }
    int index() { return _index; }
    port_type_t type() { return _port_type; }
    port_direction_t direction() { return _direction; }
    size_t data_size() { return _data_size; }
    size_t itemsize() { return _itemsize; }
};

typedef port_base::sptr port_sptr;

template <class T>
class port : public port_base
{
public:
    static std::shared_ptr<port<T>> make(const std::string& name,
            //    std::shared_ptr<node> parent,
               const port_direction_t direction,
               const port_type_t port_type = port_type_t::STREAM,
               const std::vector<size_t>& dims = std::vector<size_t>(),
               const int multiplicity = 1)
               {
                   return std::shared_ptr<port<T>>(new port<T>(name, direction, port_type, dims, multiplicity) );
               }
    port(const std::string& name,
            //    std::shared_ptr<node> parent,
               const port_direction_t direction,
               const port_type_t port_type = port_type_t::STREAM,
               const std::vector<size_t>& dims = std::vector<size_t>(),
               const int multiplicity = 1)
        : port_base(name,
            //    parent,
               direction,
               parameter_functions::get_param_type_from_typeinfo(
                   std::type_index(typeid(T))),
               port_type,
               dims,
               multiplicity)
    {
    }
};

typedef std::vector<port_sptr> port_vector_t;

} // namespace gr