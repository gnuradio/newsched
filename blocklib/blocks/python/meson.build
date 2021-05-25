######################
#  Python Bindings ###
######################

# Generate _python.cc for each block

# Compile target for each _python.cc
# blocks_pybind_sources = [
#     blocks_copy_pybind_files,
# ]

blocks_pybind_sources += files('blocks_pybind.cc')

newsched_blocklib_blocks_pybind = py3_mod.extension_module('blocks_python',
    blocks_pybind_sources, 
    dependencies : [newsched_blocklib_blocks_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','blocks')
)

newsched_blocklib_blocks_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_pybind,
                       dependencies : blocks_deps)

# Generate python_bindings.cc

# Compile target for python_bindings.cc

# Target for pure python
py3_inst.install_sources(files('__init__.py'), subdir : join_paths('newsched','blocks'))
