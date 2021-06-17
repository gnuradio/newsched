yml_file = 'atsc_derandomizer.yml'

dtv_atsc_derandomizer_files = files(['atsc_derandomizer_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    dtv_atsc_derandomizer_files += files('atsc_derandomizer_cuda.cc')
    dtv_cu_sources += files('atsc_derandomizer_cuda.cu')
endif

gen_atsc_derandomizer_h = custom_target('gen_atsc_derandomizer_cpu_h',
                        input : ['atsc_derandomizer.yml'],
                        output : ['atsc_derandomizer.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/dtv')

gen_atsc_derandomizer_cc = custom_target('gen_atsc_derandomizer_cpu_cc',
                        input : ['atsc_derandomizer.yml'],
                        output : ['atsc_derandomizer.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

dtv_deps += declare_dependency(sources : [gen_atsc_derandomizer_h])
dtv_sources += [dtv_atsc_derandomizer_files, gen_atsc_derandomizer_cc]

if get_option('enable_python')
    gen_atsc_derandomizer_pybind = custom_target('gen_atsc_derandomizer_cpu_pybind',
                            input : ['atsc_derandomizer.yml'],
                            output : ['atsc_derandomizer_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    dtv_pybind_sources += gen_atsc_derandomizer_pybind
    dtv_pybind_names += 'atsc_derandomizer'
endif
