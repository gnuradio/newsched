#include <any>
#include <complex>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

typedef std::complex<float> gr_complex;
typedef std::complex<double> gr_complexd;

namespace pmt {

enum class pmt_data_type_t {
    UNKNOWN = 0,
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
    PMT,
    ANY,
    VOID
};

enum class pmt_container_type_t { NONE, VECTOR, MAP, DICT, PAIR, TUPLE };

class pmt_functions
{
private:
    static std::map<pmt_data_type_t, std::type_index> pmt_type_index_map;
    static std::map<pmt_data_type_t, size_t> pmt_type_size_map;
    static std::map<std::type_index, pmt_data_type_t> pmt_index_type_map;

public:
    static size_t pmt_size_info(pmt_data_type_t p);
    static pmt_data_type_t get_pmt_type_from_typeinfo(std::type_index t);
};


class pmt_base
{
public:
    pmt_base(const std::any& object,
             const pmt_data_type_t data_type,
             pmt_container_type_t container_type = pmt_container_type_t::NONE)
        : _object(object), _data_type(data_type), _container_type(container_type){};
    pmt_data_type_t data_type() const { return _data_type; };
    pmt_container_type_t container_type() const { return _container_type; };
    std::any object() { return _object; }

    bool serialize(std::streambuf& sink);
    bool deserialize(std::streambuf& source);

protected:
    std::any _object;
    pmt_data_type_t _data_type;
    pmt_container_type_t _container_type;
};

typedef std::shared_ptr<pmt_base> pmt_sptr;

template <class T>
class pmt_scalar : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_scalar> sptr;
    static sptr make(const T value)
    {
        return std::make_shared<pmt_scalar<T>>(pmt_scalar<T>(value));
    }

    pmt_scalar(const T& value)
        : pmt_base(std::make_any<T>(value),
                   pmt_functions::get_pmt_type_from_typeinfo(std::type_index(typeid(T))),
                   pmt_container_type_t::NONE),
          _value(value)
    {
    }

    pmt_scalar(pmt_base& b) : pmt_base(b) { _value = std::any_cast<T>(b.object()); }

    void set_value(T val)
    {
        _value = val;
        _object = std::make_any<T>(val);
    }
    T value() { return _value; };

    void operator=(const T& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const T& other) { return other == _value; }
    bool operator!=(const T& other) { return other != _value; }

protected:
    T _value;
};

template <class T>
class pmt_vector : public pmt_base
{
public:
    typedef std::shared_ptr<pmt_vector> sptr;
    static sptr make(const T value)
    {
        return std::make_shared<pmt_vector<T>>(pmt_vector<T>(value));
    }

    pmt_vector(const std::vector<T>& value)
        : pmt_base(std::make_any<std::vector<T>>(value),
                   pmt_functions::get_pmt_type_from_typeinfo(std::type_index(typeid(T))),
                   pmt_container_type_t::VECTOR) //,
        //   _value(std::move(value))
    {
        _value = value;
    }

    pmt_vector(pmt_base& b) : pmt_base(b)
    {
        _value = std::any_cast<std::vector<T>>(b.object());
    }

    void set_value(T val)
    {
        _value = val;
        _object = std::make_any<std::vector<T>>(val);
    }
    std::vector<T> value() const { return _value; };

    void operator=(const std::vector<T>& other) // copy assignment
    {
        set_value(other);
    }

    bool operator==(const std::vector<T>& other) { return other == _value; }
    bool operator!=(const std::vector<T>& other) { return other != _value; }


protected:
    std::vector<T> _value;
};

} // namespace pmt