runtime_pybind_sources = files([
    'block_work_io_pybind.cc',
    'tag_pybind.cc',
    'block_pybind.cc',
    'node_pybind.cc',
    'sync_block_pybind.cc',
    'edge_pybind.cc',
    'graph_pybind.cc',
    'flowgraph_pybind.cc',
    'runtime_pybind.cc',
    'scheduler_pybind.cc',
    'buffer_pybind.cc',
    'utils_pybind.cc'
])

cpp_args = []
runtime_pybind_deps = [newsched_runtime_dep, python3_dep, pybind11_dep, pmtf_dep]
if (cuda_available and get_option('enable_cuda'))
    runtime_pybind_sources += 'cudabuffer_pybind.cc'
    runtime_pybind_deps += newsched_runtime_cu_dep
    cpp_args = '-DHAVE_CUDA'
endif
newsched_runtime_pybind = py3_inst.extension_module('runtime_python',
    runtime_pybind_sources, 
    dependencies : runtime_pybind_deps,
    link_language : 'cpp',
    install : true,
    cpp_args : cpp_args,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','gr')
)

newsched_runtime_pybind_dep = declare_dependency(
					   link_with : newsched_runtime_pybind)