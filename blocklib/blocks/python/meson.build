######################
#  Python Bindings ###
######################

# Generate _python.cc for each block

# Compile target for each _python.cc
# blocks_pybind_sources = [
#     blocks_copy_pybind_files,
# ]

d = {
  'blocks' : blocks_pybind_names,
  'module' : 'blocks',
  'imports' : ['newsched.gr']
}

message(d)

# blocks_pybind_sources += files('blocks_pybind.cc')

gen_blocks_pybind = custom_target('gen_blocks_pybind',
                        output : ['blocks_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

newsched_blocklib_blocks_pybind = py3_mod.extension_module('blocks_python',
    gen_blocks_pybind, 
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
