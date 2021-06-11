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
  'blocks' : analog_pybind_names,
  'module' : 'analog',
  'imports' : ['newsched.gr']
}

gen_analog_pybind = custom_target('gen_analog_pybind',
                        output : ['analog_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

analog_pybind_sources += gen_analog_pybind

newsched_blocklib_analog_pybind = py3_inst.extension_module('analog_python',
    analog_pybind_sources, 
    dependencies : [newsched_blocklib_analog_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','analog')
)

newsched_blocklib_analog_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_analog_pybind,
                       dependencies : analog_deps)


# Target for pure python
py3_inst.install_sources(srcs, subdir : join_paths('newsched','analog'))
