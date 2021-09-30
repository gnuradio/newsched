######################
#  Python Bindings ###
######################

# Generate _python.cc for each block

srcs = ['__init__.py']
inc = ['../../lib/']
soapy_pybind_sources = ['bindings/block_pybind.cc','bindings/soapy_types_pybind.cc','bindings/soapy_common.cc'] + soapy_pybind_sources
soapy_pybind_names = ['block'] + soapy_pybind_names

foreach s: srcs
configure_file(copy: true,
    input: s,
    output: s
)
endforeach

d = {
  'blocks' : soapy_pybind_names,
  'module' : 'soapy',
  'imports' : ['newsched.gr']
}

gen_soapy_pybind = custom_target('gen_soapy_pybind',
                        output : ['soapy_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

soapy_pybind_sources += gen_soapy_pybind

newsched_blocklib_soapy_pybind = py3_inst.extension_module('soapy_python',
    soapy_pybind_sources, 
    dependencies : [newsched_blocklib_soapy_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','soapy'),
    include_directories : inc
)

newsched_blocklib_soapy_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_soapy_pybind,
                       dependencies : soapy_deps)

# Generate python_bindings.cc

# Compile target for python_bindings.cc

# Target for pure python
py3_inst.install_sources(files('__init__.py'), subdir : join_paths('newsched','soapy'))
