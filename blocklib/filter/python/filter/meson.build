######################
#  Python Bindings ###
######################

# Generate _python.cc for each block

srcs = ['__init__.py']

foreach s: srcs
configure_file(copy: true,
    input: s,
    output: s
)
endforeach

d = {
  'blocks' : filter_pybind_names,
  'module' : 'filter',
  'imports' : ['newsched.gr']
}

gen_filter_pybind = custom_target('gen_filter_pybind',
                        output : ['filter_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

filter_pybind_sources += gen_filter_pybind

newsched_blocklib_filter_pybind = py3_inst.extension_module('filter_python',
    filter_pybind_sources, 
    dependencies : [newsched_blocklib_filter_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','filter')
)

newsched_blocklib_filter_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_filter_pybind,
                       dependencies : filter_deps)

# Generate python_bindings.cc

# Compile target for python_bindings.cc

# Target for pure python
py3_inst.install_sources(files('__init__.py'), subdir : join_paths('newsched','filter'))
