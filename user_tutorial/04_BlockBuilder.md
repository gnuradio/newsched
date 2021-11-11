# Block Design Workflow

Creating a custom block in newsched is designed to be as easy as possible and get the
developer right to the "insert signal processing here" part 

## gr_modtool Substitute

Partly out of "getting it done quickly" and partly out of leveraging the simplicity of 
the new block design, there are some simple scripts to create modules and blocks
rather than any `gr_modtool` integration at this point

Integration into a new gr_modtool would be a highly desirable feature for `newsched`

### Create a new module
Let's assume the `newsched` source tree lives at `$NEWSCHED_SRC`, which is likely
`$PREFIX/src/newsched`, and we want to create an out of tree module named `myoot`

```
cd $PREFIX/src
python3 $NEWSCHED_SRC/utils/modtool/create_mod.py myoot
```
This has now created a skeleton module structure at `$NEWSCHED_SRC/ns-myoot/`

Now, we can create a block that we will use to do our signal processing

```
cd ns-myoot
python3 $NEWSCHED_SRC/utils/modtool/create_block.py foo --cpu
```
The `create_block.py` script will generate the block with the desired reference
implementation (`--cpu` and/or `--cuda`)
```
usage: create_block.py [-h] [--cpu] [--cuda] [--templated] block_name
```
Also, with the `--templated` flag, a block that will be templated across a 
variety of possible datatypes (complex, float, etc.) will be generated and 
can be modified accordingly

Now, let's look at the components of the block that was generated
### The YAML File
The files for each block lives in it's own directory, and the main file driving
much of the underlying code generation is the yaml file - in this case `foo.yml`

At the top of the yaml file, we have some basic information
```yaml
module: myoot # <-- the module to which this block belongs
block: foo    # <-- the name of the block - needs to match the dir
label: foo    # <-- the label as it would show up in GRC
blocktype: sync_block  # <-- the type of the block as in c++ class names {sync_block, block }
```

If the block is templated, we have a section for how the templated code will
be generated
```yaml
typekeys:
  - id: T # <-- an id for the template key
    type: class # <-- the type of template parameter
    options:    
      - value: int16_t # <-- the c++ datatype to be explicitly instantiated
        suffix: ss     # <-- suffix given for typedef, e.g. add_ss
      - value: int32_t 
        suffix: ii 
      - value: float
        suffix: ff   
      - value: gr_complex 
        suffix: cc 
```
Next, we have parameters, which define how the block will be instantiated
and accessed.  Unless told otherwise, all parameters become constructor 
arguments

```yaml
parameters:
-   id: k            # <-- the name of the parameter
    label: Constant  # <-- for GRC purposes
    dtype: T         # <-- the datatype - can be any c++ datatype, or one of the typekeys
    settable: true   # <-- if true, will create set_k() and k() methods
-   id: vlen
    label: Vec. Length
    dtype: size_t
    settable: false
    default: 1       # <-- default value, must go last in list (no non-default value parameters after a default)

## additionally, can set
#   cotr: false      #< -- this will make the parameter not appear in the constructor, but can be set or queried via accessors
```

The ports reference how blocks will be connected together and with what corresponding types
```yaml
ports:
-   domain: stream  # <-- {stream, message}
    id: in          # <-- give the port a name
    direction: input  #<-- {input, output}
    type: typekeys/T  # <-- lookup in the typekeys section, type T
    dims: parameters/vlen  # <-- lookup in the parameters section, value of vlen

-   domain: stream
    id: out
    direction: output
    type: typekeys/T
    dims: parameters/vlen
```

The implementations section is super-important. This will determine which implementations you will 
be specifying in the c++ files that live with the yaml file

In the example below, we will be including a `foo_cpu.cc` file, but have decided to not compile a 
`foo_cuda.cc`

In the top level `meson.build`, it is important that for each implementation, an `IMPLEMENT_IMPL` 
variable is set.  In this case, we would need `IMPLEMENT_CPU=true` set somewhere in that `meson.build`

```yaml
implementations:
-   id: cpu
# -   id: cuda
```

[Next: Implementing the Block](05_BlockImplementation)