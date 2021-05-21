runtime_pybind_sources = files([
    'block_work_io_pybind.cc',
    'tag_pybind.cc',
    'block_pybind.cc',
    'node_pybind.cc',
    'sync_block_pybind.cc',
    'graph_pybind.cc',
    'flowgraph_pybind.cc',
    'runtime_pybind.cc'
 ] )

newsched_runtime_pybind = py3_mod.extension_module('gr_python',
    runtime_pybind_sources, 
    dependencies : [newsched_runtime_dep, python3_dep, pybind11_dep],
    link_language : 'cpp'
)

newsched_runtime_pybind_dep = declare_dependency(
					   link_with : newsched_runtime_pybind)