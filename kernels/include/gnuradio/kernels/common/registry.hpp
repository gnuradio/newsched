#pragma once
#include <gnuradio/kernels/cpu/kernel.hpp>
#include <map>

namespace gr {
namespace kernels {

class kernel_registry_interface
{
public:
    /**
     * @brief Create a kernel registration object
     *
     * @param kernel
     */
    virtual void create_kernel_registration(kernel_interface* kernel) = 0;

    /**
     * @brief Read a kernel registration object
     *
     * @param kernel
     */
    virtual void read_kernel_registration(kernel_interface* kernel) = 0;

    /**
     * @brief Update a kernel registration object
     *
     * @param kernel
     */
    virtual void update_kernel_registration(kernel_interface* kernel) = 0;

    /**
     * @brief Delete a kernel registration object
     *
     * @param kernel
     */
    virtual void delete_kernel_registration(kernel_interface* kernel) = 0;

    /**
     * @brief Write Kernel registration to a cache
     *
     */
    virtual void cache_write_registry() = 0;

    /**
     * @brief Read Kernel registration from a cache
     *
     */
    virtual void cache_read_registry() = 0;


    /**
     * @brief Force there to be a single globally available kernel_registry
     *
     * @return kernel_registry_interface*
     */
    static kernel_registry_interface* Global()
    {
        static kernel_registry_interface* global_kernel_registry =
            new kernel_registry_interface;
        return global_kernel_registry;
    };
};

class map_kernel_registry : kernel_registry_interface
{
    void create_kernel_registration(kernel_interface* kernel);
    void read_kernel_registration(kernel_interface* kernel);
    void update_kernel_registration(kernel_interface* kernel);
    void delete_kernel_registration(kernel_interface* kernel);

    void cache_write_registry();
    void cache_read_registry();

private:
    std::map<std::string, kernel_interface*> _registry;
}

} // namespace kernels
} // namespace gr
