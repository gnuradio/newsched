// Define interface for a block module: a module of blocks

#include <string>
#include <vector>

namespace gr {
namespace plugins {


class block_info
{
public:
    std::string name;
    std::vector<std::string> parameters;
};

/**
 * @brief Interface definition for a component in a block library
 *
 */
class plugin_block_module
{
public:
    plugin_block_module(){};
    virtual ~plugin_block_module(){};

    // void register_plugin(ComponentManager &CM);
    std::vector<uint64_t> get_supported_runtime_versions();
    std::vector<block_info> list_blocks();
    std::string name();
    std::vector<std::string> categories();


private:
};
} // namespace plugins
} // namespace gr