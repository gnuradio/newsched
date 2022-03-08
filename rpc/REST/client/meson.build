# pybind_sources = files([
#     'gr_rest_client.cc'
#  ] )

# cpp_args = []
# runtime_pybind_deps = [newsched_gr_dep, python3_dep, pybind11_dep, pmtf_dep]

# newsched_rpc_rest_client_pybind = py3_inst.extension_module('rpc_rest_client_python',
#     runtime_pybind_sources, 
#     dependencies : runtime_pybind_deps,
#     link_language : 'cpp',
#     install : true,
#     cpp_args : cpp_args,
#     install_dir : join_paths(py3_inst.get_install_dir(),'gnuradio','gr','rpc','rest')
# )

# newsched_rpc_rest_client_pybind_dep = declare_dependency(
# 					   link_with : newsched_rpc_rest_client_pybind)



#### Python Output
srcs = [
    '__init__.py',
    'gr_rest_client.py'
    ]

foreach s: srcs
configure_file(copy: true,
    input: s,
    output: s
)
endforeach

py3_inst.install_sources(files(srcs), subdir : join_paths('gnuradio','rpc','rest'))
