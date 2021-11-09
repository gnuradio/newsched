# Nodes and Blocks


## Nodes

Reference [commit](https://github.com/gnuradio/newsched/commit/b18d2fe2e2e4fbdee7d99f55da0b4a1c4c46df38)


Everything that can be connected together in newsched derives from ***node***.  A node _has_ a name, and alias, an global id, and ports, and some methods to search through the port vectors.  Node is a pretty thin base class.

### Name (string)
Upon instantiation, each node is given a name.  If the node is a block for example, it might be given the name `"add"`

### Alias (string)
A unique name that combines the name with another identifier.  Currently the identifier is the global id

### ID (int)
The id represents a unique value for this node across all the nodes (and potentially other things) in an entire flowgraph.  This is generated using the `nodeid_generator`.  

#### Nodeid_generator
The Nodeid generator is a singleton class that spits out incrementing integers so each node can have a unique ID.  This could be done in a more complex manner if desired.  It is called through the static method `nodeid_generator::get_id();`

### Ports (vector of port_base_sptr)
The list of ports associated with this node


## Block

Reference [commit](https://github.com/gnuradio/newsched/commit/23a4a829fa8762ad1defa0bd3a9594753e1d0977)

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
    uint64_t n_items_read = 0; // Name TBD. Replacement for _read and _written because I/O
    void* items;           // cannot be const for output items
    std::vector<tag_t> tags;
    int n_consumed; // output the number of items that were consumed on the work
...
struct block_work_output {
    int n_items;
    uint64_t n_items_written = 0; // Name TBD. Replacement for _read and _written because I/O
    void* items;              // cannot be const for output items
    std::vector<tag_t> tags;
    int n_produced; // output the number of items that were consumed on the work() call
...
```

`block_work_input` and `block_work_output` are separated in the class definitions just because of convenient field naming (e.g. consumed vs produced).  These structs may change over time to handle things such as blocking I/O.

### Work function

```cpp
virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) = 0;
```
Some notable changes on the work function from GR

* Call it `work` and not `general_work`
* Returns an enum indicating what happened instead of `noutput_items`
* Work replaces forecast
    - an enum value of `INSUFFICIENT_INPUT/OUTPUT` is used from the work function to handle forecasting
* Removes the restriction that all output ports must produce the same number of samples, though this could be a challenging scheduler bookkeeping problem
* No history.  
    - Blocks will be responsible for saving samples for their own history


### Tags
`runtime/include/gnuradio/tag.hpp` creates a tag class that looks very similar to GR, and it is intended that tags are used to attach metadata.

The only real difference here is that the tags do not include PMTs as we would like to replace those and have not included them yet in this implementation sequence.