######################
#  Python Bindings ###
######################

# Manually created noblock sources
fft_pybind_sources = ['bindings/window_pybind.cc'] + fft_pybind_sources
fft_pybind_names = ['window'] + fft_pybind_names

srcs = ['__init__.py']

foreach s: srcs
configure_file(copy: true,
    input: s,
    output: s
)
endforeach

d = {
  'blocks' : fft_pybind_names,
  'module' : 'fft',
  'imports' : ['newsched.gr']
}

gen_fft_pybind = custom_target('gen_fft_pybind',
                        output : ['fft_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_module_pybind.py'),
                            '--blocks', d['blocks'],
                            '--imports', ' '.join(d['imports']),
                            '--module', d['module'],
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)      

fft_pybind_sources += gen_fft_pybind

newsched_blocklib_fft_pybind = py3_inst.extension_module('fft_python',
    fft_pybind_sources, 
    dependencies : [newsched_blocklib_fft_dep, python3_dep, pybind11_dep],
    link_language : 'cpp',
    install : true,
    install_dir : join_paths(py3_inst.get_install_dir(),'newsched','fft')
)

newsched_blocklib_fft_pybind_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_fft_pybind,
                       dependencies : fft_deps)


# Target for pure python
py3_inst.install_sources(srcs, subdir : join_paths('newsched','fft'))
