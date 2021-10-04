yml_file = 'source.yml'

soapy_source_files = files(['source_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     soapy_source_files += files('source_cuda.cc')
#     soapy_cu_sources += files('source_cuda.cu')
# endif

gen_source_h = custom_target('gen_source_cpu_h',
                        input : ['source.yml'],
                        output : ['source.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/soapy')

gen_source_cc = custom_target('gen_source_cpu_cc',
                        input : ['source.yml'],
                        output : ['source.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

soapy_deps += declare_dependency(sources : [gen_source_h])
soapy_sources += [soapy_source_files, gen_source_cc]

if get_option('enable_python')
    gen_source_pybind = custom_target('gen_source_cpu_pybind',
                            input : ['source.yml'],
                            output : ['source_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    soapy_pybind_sources += gen_source_pybind
    soapy_pybind_names += 'source'
endif
