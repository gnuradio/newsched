// Define interface for a blocklib: a module of blocks

namespace gr {
namespace components {
/**
 * @brief Interface definition for a component in a block library
 * 
 */
class IBlockLibComponent {
public:
  IBlockLibComponent(){};
  virtual ~IBlockLibComponent(){};

    void register_component(ComponentManager &CM);
    void get_expected_runtime_version();

private:
};
} // namespace components
} // namespace gr