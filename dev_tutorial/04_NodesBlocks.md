# Nodes and Blocks


## Nodes

Everything that can be connected together in newsched derives from ***node***.  A node _has_ a name, and alias, an global id, and ports, and some methods to search through the port vectors.  Node is a pretty thin base class.

### Name (string)
Upon instantiation, each node is given a name.  If the node is an `add` block for example, it might be given the name `"add"`

### Alias (string)
A unique name that combines the name with another identifier.  Currently the identifier is the global id

### ID (int)
The id represents a unique value for this node across all the nodes (and potentially other things) in an entire flowgraph.  This is generated using the `nodeid_generator`.  

#### Nodeid_generator
The Nodeid generator is a singleton class that spits out incrementing integers so each node can have a unique ID.  This could be done in a more complex manner if desired.  It is called through the static method `nodeid_generator::get_id();`

### Ports (vector of port_base_sptr)
The list of ports associated with this node.  There are additional convenience wrappers to access e.g. input ports, or message ports, or just the output stream ports.

## Block

The block class is much more interesting than its base class because it is closer to where signal processing will take place.  Much of the functionality of GR has been stripped out in the above commit in order to provide a basic set of methods for running a simple flowgraph.  Just enough to get to a work function, and a reference to tags, though we won't be doing anything with them just yet.

But the overall goal is for the block API to be

1. Simpler
2. Detached from the Scheduler State

We should be able to make block work calls without any scheduler involved at all to do general signal processing in a simulation environment.

### Block Work I/O

Rather than have a large and restrictive function signature for the `work()` function, blocks will take in a vector of `block_work_input` and `block_work_output` structs

```cpp
struct block_work_input {
    int n_items;
    buffer_reader_sptr buffer;
    int n_consumed =
        -1; // output the number of items that were consumed on the work() call
...
struct block_work_output {
    int n_items;
    buffer_sptr buffer;
    int n_produced =
        -1; // output the number of items that were produced on the work() call
...
```

`block_work_input` and `block_work_output` are separated in the class definitions just because of convenient field naming (e.g. consumed vs produced).  These structs may change over time to handle things such as blocking I/O.

Rather than passing in a raw pointer to the samples in the buffer object, a pointer to the actual buffer object
is passed in.  This allows more complex operations to be done some particular buffer type since some buffers
(e.g. OpenCL) don't have unrestricted raw memory access.  The buffer also holds the tags, so these can be 
accessed via convenience methods within `block_work_input` and `block_work_output`

Inside the `work` function, the raw pointers can be obtained by calling the templated `items()` method on the `block_work_input` or `block_work_output` item.  For examples

```cpp
auto in = work_input[0]->items<float>();
```

The `void *` pointer from the buffer can also be accessed using `->raw_items()`

### Work function

```cpp
virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) = 0;
```
Some notable changes on the work function from GR

* Call it `work` and not `general_work`
* Returns an enum indicating what happened instead of `noutput_items`
* Work replaces forecast
    - an enum value of `INSUFFICIENT_INPUT/OUTPUT` is used from the work function to handle forecasting
    - Additionally a pseudo forecasting could be added as another `block_work_io` struct member to return
      the required number of inputs or outputs 
* Removes the restriction that all output ports must produce the same number of samples, though this could be a challenging scheduler bookkeeping problem
* No history.  
    - Blocks will be responsible for saving samples for their own history

### Tags
`runtime/include/gnuradio/tag.h` creates a tag class that takes advantage of the new `pmtf` library (https://github.com/gnuradio/pmt).  This structure might change once tags are defined explicitly in the `pmtf` library

The tags are associated with the buffer class, not the work I/O function

[--> Next: Graphs and Buffers](04_GraphsBuffers)
