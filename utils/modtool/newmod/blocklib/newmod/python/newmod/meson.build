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
  'blocks' : newmod_pybind_names,
  'module' : 'newmod',
  'imports' : ['newsched.gr']
}

gen_newmod_pybind = custom_target('gen_newmod_pybind',
                        output : ['newmod_pybind.cc'],
                        command : ['python3', join_paths(SCRIPTS_DIR,'process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

newmod_pybind_sources += gen_newmod_pybind

newsched_blocklib_newmod_pybind = py3_inst.extension_module('newmod_python',
    newmod_pybind_sources, 
    dependencies : [newsched_blocklib_newmod_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','newmod')
)

newsched_blocklib_newmod_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_newmod_pybind,
                       dependencies : newmod_deps)

# Generate python_bindings.cc

# Compile target for python_bindings.cc

# Target for pure python
py3_inst.install_sources(files('__init__.py'), subdir : join_paths('newsched','newmod'))
