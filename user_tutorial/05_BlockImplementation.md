# Block Implementation

Let's take a look at what might be in our `foo_cpu.h` and `foo_cpu.cc` file

## Implementation Header (`foo_cpu.h`)
```cpp
#pragma once

#include <gnuradio/myoot/foo.h>

namespace gr {
namespace myoot {

class foo_cpu : public foo
{
public:
    foo_cpu(block_args args) : sync_block("foo"), foo(args) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

private:
    // private variables here
};

} // namespace myoot
} // namespace gr
```
### The Constructor
Rather than drag around a list of constructor parameters, these were generated automatically
from the yaml file and clubbed into `block_args args`.  We can use `args.k`, for example, in
the constructor to initialize some local variable

But since `k` was defined as a parameter, we don't need to - we have access to a `param` object
in the work function that is thread-safe and kept up to date with changes from any access mechanism
(e.g. tags, rpc, message ports)

Just like a regular GR block, we can specify any private variables in here

You'll notice that the work function looks different than GR 3.x - this is because we have
more flexibilty by passing in structs than a complicated function signature.  More on this is 
in the developer tutorial

## Implementation Source (`foo_cpu.cc`)

At its simplest, the implementation source can be __just a work function__ :open_mouth: which gets
us to the original design goal - `<insert signal processing here>`

```cpp
#include "foo_cpu.h"
#include "foo_cpu_gen.h"

namespace gr {
namespace josh {

work_return_code_t foo_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    return work_return_code_t::WORK_OK;
}


} // namespace josh
} // namespace gr
```

We of course include `foo_cpu.h`, but we also need to include `foo_cpu_gen.h` - which is some
boiler plate code necessary per implementation.  This includes the `make_cpu` method which 
just wraps the constructor, and for templated blocks, the explicit instantiation for types
that we specified in the yaml file

### The work function

The work function always returns a `work_return_code_t` code, which is usually `WORK_OK`, but in 
the case where we would normally do forecasting - i.e. the work function has set requirements on
the number of samples it needs - we can return `WORK_INSUFFICIENT_INPUT` or `WORK_INSUFFICIENT_OUTPUT`.
This simplifies things for us by not requiring a `forecast` method

#### work_input/output
These vectors of structs represent the buffers passed in and out per streaming port.

To get access to a pointer to the samples on input port 0 as floats:
```cpp
auto in = work_input[0].items<float>();
```
#### Number of items
```cpp
auto noutput_items = work_output[0].n_items;
auto ninput_items = work_input[0].n_items;
```

#### Consuming/Producing

Since the number of items produced is not the return value, we must consume inside the work function
```cpp
consume_each(noutput_items, work_input);
produce_each(noutput_items, work_output);
```
For sync_blocks, we only need to produce, but for `block` we must do both