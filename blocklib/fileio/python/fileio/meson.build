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
  'blocks' : fileio_pybind_names,
  'module' : 'fileio',
  'imports' : ['newsched.gr']
}

gen_fileio_pybind = custom_target('gen_fileio_pybind',
                        output : ['fileio_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

fileio_pybind_sources += gen_fileio_pybind

newsched_blocklib_fileio_pybind = py3_inst.extension_module('fileio_python',
    fileio_pybind_sources, 
    dependencies : [newsched_blocklib_fileio_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','fileio')
)

newsched_blocklib_fileio_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_fileio_pybind,
                       dependencies : fileio_deps)


# Target for pure python
py3_inst.install_sources(srcs, subdir : join_paths('newsched','fileio'))
