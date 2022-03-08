#include <gnuradio/rpc_client_interface.h>

namespace gr {
namespace rpc {
namespace rest {

class client
{
public:
    void block_method(const std::string& block_name,
                      const std::string& method,
                      const std::string& payload) override;

    pmtf::pmt block_parameter_query(const std::string& block_name,
                                    const std::string& parameter) override;

    void block_parameter_change(const std::string& block_name,
                                const std::string& parameter,
                                const std::string& payload) override;
};

} // namespace rest
} // namespace rpc
} // namespace gr