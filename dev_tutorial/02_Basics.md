# Basic Structure

Starting with [commit](https://github.com/gnuradio/newsched/commit/8827f3aff37ae994b77d75322f626342939dc744) 

## Licensing

Because newsched will be borrowing heavily from GNU radio, we stick with the GPLv3 license to ensure no license incompatibility issues

## Caveats

You will see in the implementation that many many things are constructed as `shared_ptr`s due to their simplicity.  This can be cleaned up.  There are other non-optimal implementations, as the current state is to establish the core concepts.  Please consider optimizing and modernizing as the project matures.

You will also see that much of the code is copied or the ideas copied from GNU Radio - not always built up from scratch.

## Folders

Since newsched is designed with modularity in mind, the key components that will be modular are pulled away from the runtime directory

- runtime - core components used by other modules
- blocks - block library - already modular in GNU radio but pulled away from runtime
- schedulers - implementations of in-tree cpu and common domain schedulers
- docs - all general documentation related things

```
├── blocklib
├── docs
├── runtime
└── schedulers
```

## Logging

Reference [commit](https://github.com/gnuradio/newsched/commit/1f74c1c0b22e312a415cbfe4b2a175c76ee64b43)

Logging is pervasive throughout the various modules, so we go ahead and include the commit that provides the logging mechanisms.  This is very similar to GNU Radio logging, e.g. `GR_LOG_DEBUG()`, but based on [spdlog](https://github.com/gabime/spdlog), which is a well supported library for all sorts of logging

The logging here was developed before the `spdlog` updates to GNU Radio were integrated so it can be expected
that the version which is in GR 3.10 will eventually become the common logging standard

### Instantiating a Logger

A logger is instantiated by trying to gain access by a string name, e.g. 

```cpp
_logger = logging::get_logger(_name, "default");
```

Most constructs, e.g. anything that derives from `node` or `scheduler` have `_logger` and `_debug_logger` objects defined in the base class.

### Using a Logger

To log a message into the logging stream (which via preferences can be through a file or console), the global logging functions are used, such as 

```cpp
template<typename... Args>
inline void gr_log_debug(logger_sptr logger, const Args &... args)
{
    if (logger) {
        logger->debug(args...);
    }
}
```

SPDlog has a convenient arguments list interface to its logging objects which are wrapped into these global functions to be more compatible with legacy GNU Radio.  

## Preferences

Reference [commit](https://github.com/gnuradio/newsched/commit/a94aed43bab39f8eca0faff0eeaeb8478b4ebeb2)

The Logging module is configured using a preferences file, which currently is searched for as `$HOME/.gnuradio/config.yml`

The yml file currently has a single section for configuring logging properties

### Logging 

`config.yml` defines a logging section such that different _logging configurations_ can be used and point to different log files.  Below is an example

```yaml
logging:
-   id: default
    type: console
    console_type: stdout
    pattern: "%+"
    level: info
    
-   id: debug
    type: basic
    pattern: "%+"
    level: debug
    filename: /tmp/gr_debug_log.txt
    
-   id: trace
    type: none
    pattern: "%+"
    level: trace
    filename: /tmp/gr_trace_log.txt
```

In this case the default logger, which would be obtained by `logging::get_logger(name, "default");`, would log messages sent to it with `INFO` and above directly to the console

The debug logger, which would be obtained with argument `"debug"` would append to the file specified in `/tmp/gr_debug_log.txt`

The trace level logger is turned off in this example

[--> Next: Ports](03_Ports)