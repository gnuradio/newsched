# Ports

Reference [commit](https://github.com/gnuradio/newsched/commit/1b5949c8e1665d8a0daf82f0b0b98f09832b9823)

newsched introduces the concept of ***ports*** to replace io_signature in a more descriptive and portable manner, and simplify the logic for connecting and maintaining flowgraphs

This is more consistent with how blocks are presented in _GNU Radio Companion (GRC)_, and one of the goals of this project is to make GUI interfaces to GNU Radio such as GRC much thinner veneers around a rich core API


***Note: in the block library, the ports are automatically generated from the block .yml, so most of this code will never be touched by an OOT developer***
## Parameter Types

Ports can be typed or untyped, and all the supported types are contained in `runtime/include/gnuradio/parameter_types.h` in the enum

```cpp
enum class param_type_t {
```

Also included in `parameter_types.h` is methods for mapping c++ `type_info` and the sizes of particular types to and from these enums

## Port Base Class

All of the port related code is defined in `runtime/include/gnuradio/port.h`.  Currently, a port can be one of two types - `STREAM` or `MESSAGE`.  Just like in GRC, both message and stream ports can be connected to like ports, but the API more closely resembles what is done for GR Message ports.  The base port class should not be directly instantiated.

The base port class contains the following fields:

### Name
Name given to the port upon instantiation - usually something like `"in"` or `"out"` - such as what would show up in GRC ports

### Direction
Defined by the enum with values `INPUT`, `OUTPUT` and `BIDIRECTIONAL`.  Bidirectional is not implemented, but intended for the case where a block doesn't need to modify the buffer, but performs some other function, e.g. a `head` block.  Not sure exactly how this would work.

### Data Type
From the enum list in `parameter_types.h`, this defines the underlying type of data that will be passed through this port.  The data streams can be vectors of this type as defined by the `_dims` field.  `UNTYPED` refers to a port that doesn't care about the type, and will just process the incoming data byte for byte, e.g. a `copy` or `head` block.

### Port Type
`STREAM` or `MESSAGE` port.  Stream ports will hand their data through buffers associated with graph edges.  Message ports will push the data onto a queue handled by the scheduler threads.

### Index
As ports are instantiated onto a block, an index per port is given - this is similar to the indexing used with io_signature.  TBD whether it is global across message/stream ports

### Dims
For vector or matrix inputs to a block, set the dims accordingly.  For instance, if an FFT block takes in vectors of 1024 samples, `_dims` would get set to `std::vector<int>{1024}`.  

If a block interprets its input as 4x4 matrices, `_dims` would get set to `std::vector<int>{4,4}`

***Note: Should be renamed to `shape` to match the common nomenclature of numpy***

### Multiplicity
Not currently used.  Intended to allow a single port to be configured but replicated similar to bus ports in GRC.  Not sure how this will work yet.

### Datasize
CALCULATED INTERNALLY - The size of an individual element - for instance a `100` length vector input of `uint16_t` would have a `datasize` of `2` (ie it disregards the vector aspect)

### Itemsize
CALCULATED INTERNALLY - The size of items to be processed in the work function - for instance a `100` length vector input of `uint16_t` would have an `itemsize` of `200`

## Typed Ports
Typed ports wrap the base `port` class with a standard c++ type through the template parameter and take care of all the sizing, etc.

Creating a typed port within the block factory would look something like:

```cpp
auto p = port<float>::make("input",
            port_direction_t::INPUT,
            port_type_t::STREAM,
            std::vector<size_t>{ vlen });
```
This creates an input stream port, named as such that expects vectors of `vlen` floats.

## Untyped Ports

Some blocks don't care about the underlying datatype and just process the raw data that passes through.  Untyped ports preserve this behavior that was controlled previously by the io_signature.

Untyped ports are instantiated as 

```cpp
auto p = untyped_port::make(
            "out", port_direction_t::OUTPUT, 
            itemsize, 
            port_type_t::STREAM);

```

Additionally, an itemsize of 0 will be inferred from the port connected to it at flowgraph initialization

[--> Next: Nodes and Blocks](04_NodesBlocks)