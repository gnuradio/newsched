#pragma once

#include <map>
#include <string>


namespace gr {
/**
 * @brief General Purpose Dictionary
 *
 * The gpdict allows storage of values to attribute to the block by
 * use by mechanisms unknown to the block itself.  For example, a scheduler may have a
 * concept of cpu affinity, but that doesn't really matter to the block itself.
 * Instead of building in specific requirements to the block, allow a generic storage
 * container to hold such values
 */
class gpdict
{
    std::map<std::string, int> _int_dict;
    std::map<std::string, double> _real_dict;
    std::map<std::string, bool> _bool_dict;
    std::map<std::string, std::string> _str_dict;

public:
    void set_int_value(const std::string& key, const int val) { _int_dict[key] = val; }
    void set_real_value(const std::string& key, const double val)
    {
        _real_dict[key] = val;
    }
    void set_string_value(const std::string& key, const std::string& val)
    {
        _str_dict[key] = val;
    }
    void set_bool_value(const std::string& key, const bool val) { _bool_dict[key] = val; }

    int get_int_value(const std::string& key) { return _int_dict[key]; }
    double get_real_value(const std::string& key) { return _real_dict[key]; }
    std::string get_string_value(const std::string& key) { return _str_dict[key]; }
    bool get_bool_value(const std::string& key) { return _bool_dict[key]; }
};

} // namespace gr