# Graphs and Buffers

Reference [commit](https://github.com/gnuradio/newsched/commit/5072a3223a92afc0729daf8756c41a46810e34b9)

The basic constructs for a graph, which is a group of nodes connected together, is similar if not exact and just renamed from GNU Radio

## Terminology

| GNU Radio       | newsched    |
| :-----------    | :---------- |
|  basic_block    | node        |
|  block          | block       |
|  hier_block2    | graph       |
|  top_block      | flowgraph   |
|  flat_flowgraph | flat_graph  |

## Graphs

Graphs inherit from node because they can also be connected together via the exposed ports.  Taking a look at `graph.hh` we first notice that most of this code was ripped straight from GNU Radio.  

The graph class is used to connect ports that reside on a node together, and can currently be done via `node_sptr` and port index, or by the `endpoint` which is just a pair with `node_sptr` and `port_sptr`.

```cpp
    edge_sptr connect(const node_endpoint& src,
                 const node_endpoint& dst);
    edge_sptr connect(node_sptr src_node,
                 unsigned int src_port_index,
                 node_sptr dst_node,
                 unsigned int dst_port_index);
```
More overloads of connect can be easily added, such as connecting by port name, defaulting to port 0, etc.

Connecting two ports into a graph does the following
1. Check whether the connection is legal
2. Assign an alias to the blocks involved
3. Create a new edge
4. Return the newly created edge (for further operations such as setting a custom buffer if desired)

### Orphan Nodes
In the case of a graph that includes blocks that are not connected to anything else, or a graph that includes only one block and has no edges, these nodes are tracked as they still need to be handled by the scheduler and potentially attached to other graphs

## Flat Graphs
Reference [commit](https://github.com/gnuradio/newsched/commit/91fe3ef8dfcc69b918f80240981ba739d2cef992)

Flat graphs are simply graphs with all the hierarchical blocks transformed into regular blocks+edges.  

The code for performing operations on flattened graphs is mostly taken straight from GNU Radio.  The only exception (which is a hack implementation that needs to be straightened out) is the `gpdict` class that allows outside entities to tag the block with metadata.  Previously graph specific properties such as the `color` simply used to mark blocks as the graph is traversed, were included in the block API.  Since these are not really block properties they have been abstracted into the block as a general purpose dictionary.  **This design should be revisited**

## Buffers

![Buffers between blocks](images/graphs_buffers.png)

A buffer is simply an object that is associated with the edge between two blocks that contains a place to store items and the necessary methods to update read/write pointers to the items stores.  In the above image, the buffer is denoted between each pair of ports, and the coloring represents that some blocks might have requirements or a desire to use different buffers than the standard GNU Radio circular buffer.  These block could be accelerated on GPU, for example.

In reality, the buffer is associated with the output port of the upstream block and any downstream blocks that are
connected have a `buffer_reader` object.  This is the same design as GNU Radio.

Below is pseudocode of how a scheduler might interface with the buffer API:

```c++
run_one_iteration {
    for b in blocks {
        for p in input_ports {
            get read_info, append to work_input
        }
        for p in output_ports 
            get write_info, append to work_output
        }

        code = b->work(work_input, work_output)

        for p in input_ports {
            post_read(nconsumed per input)
        }
        for p in output_ports 
            post_write(nproduced per output)
        }
    }
}

```

For every block (as determined by the scheduler), the buffer pointers are obtained via `buffer_info` structs.  If there is room to read and write, `work()` is called, and the buffer pointers are updated accordingly through the `post_read` and `post_write` calls.

### Simple Buffer
A simple circular buffer implementation is included as `buffer_cpu_simple.hh` as an example to show the implementation of a simple buffer class.  The `read_info` and `write_info` methods simply query the state of the circular buffer, while `post_read` and `post_write` update the pointers and perform the copy to the other half of the circular buffer

The vmcircbuf methods from GNU Radio have been ported over as `buffer_cpu_vmcirc.hh` which wrap the other possible 
vmcircbuf implementations (mmap_shm, sysv_shm)

### Buffer Factories for Custom Buffers

After an edge is created, but before buffer memory is allocated by the scheduler, it can be set with properties that tell the scheduler how to create the buffer.  By calling the `set_custom_buffer` method on a constructed edge, these properties are store for later use.

```cpp
    void set_custom_buffer(std::shared_ptr<buffer_properties> buffer_properties)
    {
        _buffer_properties = buffer_properties;
    }
```

An example of using this method would be as follows (referring to the diagram above):

```cpp
mygraph->connect(src,blk1)->set_custom_buffer(cuda_buffer_properties::make(cuda_buffer_type::H2D));
mygraph->connect(blk1,blk2)->set_custom_buffer(cuda_buffer_properties::make(cuda_buffer_type::D2D));
mygraph->connect(blk2,blk3)->set_custom_buffer(cuda_buffer_properties::make(cuda_buffer_type::D2H));
mygraph->connect(blk3,snk) // uses default buffer
```

In this example there is a derived buffer class `cuda_buffer` that has a factory (`cuda_buffer::make`), and a properties class `cuda_buffer_properties` which just wraps an enum in this case.  

To create a new custom buffer type, do the following:
1. Define the derived buffer and buffer reader classes that adhere to the buffer API
```cpp
class simplebuffer : public buffer
...
class simplebuffer_reader : public buffer_reader

```
2. Create a make function for the buffer that serves as the buffer factory
```cpp
static buffer_sptr make(size_t num_items,
                        size_t item_size,
                        std::shared_ptr<buffer_properties> buffer_properties)
{
    return buffer_sptr(new simplebuffer(num_items, item_size));
}
```
3. Create a derived buffer properties that will be used by the buffer factory (since simplebuffer doesn't have one, here is another example)
```cpp
class cuda_buffer_properties : public buffer_properties
```