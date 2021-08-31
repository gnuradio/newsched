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
  'blocks' : digital_pybind_names,
  'module' : 'digital',
  'imports' : ['newsched.gr']
}

gen_digital_pybind = custom_target('gen_digital_pybind',
                        output : ['digital_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

digital_pybind_sources += gen_digital_pybind

newsched_blocklib_digital_pybind = py3_inst.extension_module('digital_python',
    digital_pybind_sources, 
    dependencies : [newsched_blocklib_digital_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','digital')
)

newsched_blocklib_digital_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_digital_pybind,
                       dependencies : digital_deps)


# Target for pure python
py3_inst.install_sources(srcs, subdir : join_paths('newsched','digital'))
