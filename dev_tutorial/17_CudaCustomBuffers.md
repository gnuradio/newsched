# CUDA Custom Buffers

We want to explore creating our own custom buffers, and the CUDA programming model (assuming
you have an nVidia GPU) makes for convenient way to show zero-copy in and out of the GPU

## Installation Caveats

If the CUDA drivers and toolkit have been installed using `apt-get`, meson needs a bit of help to find the modules.  When calling `meson setup ...`, set the `CUDA_ROOT` environment variable to `\usr`, e.g. `CUDA_ROOT=\usr meson setup ...`

If you want the latest CUDA toolkit, and install it from tar files, there is probably an easier way to get this working, but ... on Ubuntu 20.04, in order to get CUDA installed locally on my machine, I had to:

1. Download the CUDA `runfile` installer [here](https://developer.nvidia.com/cuda-11.0-update1-download-archive)
1. Uninstall all nvidia-* packages using `apt remove`
2. [Blacklist the noveau driver](https://linuxconfig.org/how-to-disable-blacklist-nouveau-nvidia-driver-on-ubuntu-20-04-focal-fossa-linux)
3. Reboot the machine - now that CUDA and noveau are uninstalled, it can't boot into GNOME
4. In GRUB, boot to the 5.4 kernel with Recovery Mode
5. Enable Networking
6. Install the runfile
7. Reboot back into GRUB 5.4 kernel

This was all done with CUDA toolkit 11.0 and the 450 driver, so it might be the case that another combination works with the newer kernel, but this is what I had to go throug.

## Build Options

First, we must add an option to enable/disable CUDA related code from compilation at the command line, with an option in `meson_options.txt`

Next, we add the detection of the CUDA compiler
```meson
cuda_available = add_languages('cuda', required : false)
```
and a dependency that will be used throughout the rest of the `meson.build` files

```
cuda_dep = dependency('cuda', version : '>=10.1', required : cuda_available and get_option('enable_cuda'))
```

## Custom Buffers
We implement 2 types of CUDA custom buffers
1. `buffer_cuda.cu` - Establishes separate host and device memory, and `post_write` method initiates the transfer between where applicable.  
2. `buffer_cuda_pinned.cu` - Creates pinned host memory, and performs no extra H2D or D2H memcpys.  This is for the type of machine that has GPU integrated with CPU, such as Jetson.

**Note: Both of these are doing simple double copy circular buffers but this logic should be changed eventually**

The `buffer_cuda_sm.cu` buffer class uses the single mapped buffer abstraction, but still working out some of the kinks

`buffer_cuda_pinned.cu` is almost exactly the same as `simplebuffer.hpp` just with a CUDA host allocated pinned buffer instead of normal CPU memory.

`buffer_cuda.cu` is more interesting, especially in its `post_write` method.

In `buffer_cuda.hh`, we have defined some convenience macros to wrap the buffer creation arguments:
```c++
#define CUDA_BUFFER_ARGS_H2D buffer_cuda_properties::make(buffer_cuda_type::H2D)
#define CUDA_BUFFER_ARGS_D2H buffer_cuda_properties::make(buffer_cuda_type::D2H)
#define CUDA_BUFFER_ARGS_D2D buffer_cuda_properties::make(buffer_cuda_type::D2D)

```
which will make it easier when calling `set_custom_buffer` on the connected edge.  Now, taking a look at the post_write method, we see that after we have written to the host or device buffer (whichever was returned from `write_info` depending on where the buffer sits in the chain), we take the action to initiate an H2D, D2D, or D2H transfer (additional complication from the double circular buffer)

```c++
    if (_buffer_type == cuda_buffer_type::H2D) {
        cudaMemcpy(&_device_buffer[wi1],
                   &_host_buffer[wi1],
                   bytes_written,
                   cudaMemcpyHostToDevice);

        // memcpy(&_host_buffer[wi2], &_host_buffer[wi1], num_bytes_1);
        cudaMemcpy(&_device_buffer[wi2],
                   &_device_buffer[wi1],
                   num_bytes_1,
                   cudaMemcpyDeviceToDevice);
        if (num_bytes_2) {
            // memcpy(&_host_buffer[0], &_host_buffer[_buf_size], num_bytes_2);
            cudaMemcpy(&_device_buffer[0],
                       &_device_buffer[_buf_size],
                       num_bytes_2,
                       cudaMemcpyDeviceToDevice);
        }
    } else if (_buffer_type == cuda_buffer_type::D2H) {
        cudaMemcpy(&_host_buffer[wi1],
                   &_device_buffer[wi1],
                   bytes_written,
                   cudaMemcpyDeviceToHost);

        memcpy(&_host_buffer[wi2], &_host_buffer[wi1], num_bytes_1);

        if (num_bytes_2) {
            memcpy(&_host_buffer[0], &_host_buffer[_buf_size], num_bytes_2);
        }
    } else // D2D
    {
        cudaMemcpy(&_device_buffer[wi2],
                   &_device_buffer[wi1],
                   num_bytes_1,
                   cudaMemcpyDeviceToDevice);
        if (num_bytes_2)
            cudaMemcpy(&_device_buffer[0],
                       &_device_buffer[_buf_size],
                       num_bytes_2,
                       cudaMemcpyDeviceToDevice);
    }
    // advance the write pointer
    _write_index += bytes_written;
    if (_write_index >= _buf_size) {
        _write_index -= _buf_size;
    }
}
```

If the buffer is instantiated as H2D:
* `write_info` will return a pointer to the **host** buffer
* upstream block will write into the **host** buffer
* `post_write` will perform H2D transfer into **device** memory
* `read_info` will return a pointer to the **device** buffer
* downstream block will read from **device** memory

If the buffer is instantiated as D2H:
* `write_info` will return a pointer to the **device** buffer
* upstream block will write into the **device** buffer
* `post_write` will perform D2H transfer into **host** memory
* `read_info` will return a pointer to the **host** buffer
* downstream block will read from **host** memory

If the buffer is instantiate as D2D:
* Just like a normal buffer, but uses `cudaMemcpy(... DeviceToDevice)` instead

In any of these cases, the block implementation assumes that the sample buffers provided in the `work_input` and `work_output` structs are **device memory** and are able to launch kernels on it directly.


## QA Test

A single QA test for the CUDA copy block is implemented and simply checks whether the samples going in and out of the copy block get the samples across.

The only things of note are:
1. QA tests for CUDA are put into a separate folder to allow easy enabling/disabling from the option/dependency
2. Blocks are instantiated from the `cuda` block module
```c++
    auto copy1 = cuda::copy::make(1024);
    auto copy2 = cuda::copy::make(1024);
```
2. CUDA buffers are manually specified using the `set_custom_buffer` method

```c++
    fg->connect(src, 0, copy1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_H2D);
    fg->connect(copy1, 0, copy2, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2D);
    fg->connect(copy2, 0, snk1, 0)->set_custom_buffer(CUDA_BUFFER_ARGS_D2H);
```