#pragma once

#include <gnuradio/parameter_types.hpp>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace gr {

enum class port_type_t { STREAM, MESSAGE };

enum class port_direction_t {
    INPUT,
    OUTPUT,
    BIDIRECTONAL //?? can it be done
};

/**
 * @brief Base class for all ports
 *
 * Holds the necessary information to describe the port to the runtime
 *
 */
class port_base
{
protected:
    std::string _name;
    std::string _alias;
    port_direction_t _direction;
    param_type_t _data_type;
    port_type_t _port_type;
    int _index = -1;           // how does this get set??
    std::vector<size_t> _dims; // allow for matrices to be sent naturally across ports
    // empty dims refers to a scalar, dims=[n] same as vlen=n
    int _multiplicity; // port can be replicated as in grc
    size_t _datasize;
    size_t _itemsize; // data size across all dims

public:
    typedef std::shared_ptr<port_base> sptr;
    static sptr make(const std::string& name,
                     const port_direction_t direction,
                     const param_type_t data_type = param_type_t::CFLOAT,
                     const port_type_t port_type = port_type_t::STREAM,
                     const std::vector<size_t>& dims = std::vector<size_t>{ 1 },
                     const int multiplicity = 1)
    {
        return std::make_shared<port_base>(
            name, direction, data_type, port_type, dims, multiplicity);
    }

    port_base(const std::string& name,
              const port_direction_t direction,
              const param_type_t data_type = param_type_t::CFLOAT,
              const port_type_t port_type = port_type_t::STREAM,
              const std::vector<size_t>& dims = std::vector<size_t>{ 1 },
              const int multiplicity = 1)
        : _name(name),
          _direction(direction),
          _data_type(data_type),
          _port_type(port_type),
          _dims(dims),
          _multiplicity(multiplicity)
    {
        // _type_info = param_type_info(_data_type); // might not be needed
        _datasize = parameter_functions::param_size_info(_data_type);
        _itemsize = _datasize;

        // If dims is empty or [1], then the port type is a scalar value
        // If dims has values, then the total itemsize is the product of the dimensions *
        // the scalar itemsize
        for (auto d : _dims)
            _itemsize *= d;
    }

    port_base(const std::string& name,
              const port_direction_t direction,
              const size_t itemsize,
              const port_type_t port_type = port_type_t::STREAM,
              const int multiplicity = 1)
        : _name(name),
          _direction(direction),
          _data_type(param_type_t::UNTYPED),
          _port_type(port_type),
          _multiplicity(multiplicity),
          _datasize(itemsize),
          _itemsize(itemsize)
    {
    }

    std::string name() { return _name; }
    std::string alias() { return _alias; }
    void set_alias(const std::string& alias) { _alias = alias; }
    void set_index(int val) { _index = val; }
    int index() { return _index; }
    port_type_t type() { return _port_type; }
    param_type_t data_type() { return _data_type; }
    port_direction_t direction() { return _direction; }
    size_t data_size() { return _datasize; }
    size_t itemsize() { return _itemsize; }
    std::vector<size_t> dims() { return _dims; }
};

typedef port_base::sptr port_sptr;
typedef std::vector<port_sptr> port_vector_t;


/**
 * @brief Typed port class
 *
 * Wraps the port_base class with a type to take care of all the sizing and lower level
 * properties
 *
 * @tparam T datatype to instantiate the base port class
 */
template <class T>
class port : public port_base
{
public:
    static std::shared_ptr<port<T>>
    make(const std::string& name,
         const port_direction_t direction,
         const port_type_t port_type = port_type_t::STREAM,
         const std::vector<size_t>& dims = std::vector<size_t>(),
         const int multiplicity = 1)
    {
        return std::shared_ptr<port<T>>(
            new port<T>(name, direction, port_type, dims, multiplicity));
    }
    port(const std::string& name,
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


/**
 * @brief Untyped port class
 *
 * Wraps the port base class but only populates stream size info.  To be used in case of
 * e.g. head block where the underlying datatype is not used, just copied byte for byte
 *
 */
class untyped_port : public port_base
{
public:
    static std::shared_ptr<untyped_port>
    make(const std::string& name,
         const port_direction_t direction,
         const size_t itemsize,
         const port_type_t port_type = port_type_t::STREAM,
         const int multiplicity = 1)
    {
        return std::shared_ptr<untyped_port>(
            new untyped_port(name, direction, itemsize, port_type, multiplicity));
    }
    untyped_port(const std::string& name,
                 const port_direction_t direction,
                 const size_t itemsize,
                 const port_type_t port_type = port_type_t::STREAM,
                 const int multiplicity = 1)
        : port_base(name, direction, itemsize, port_type, multiplicity)
    {
    }
};

} // namespace gr
