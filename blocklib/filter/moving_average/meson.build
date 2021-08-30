filter_moving_average_files = files(['moving_average_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    filter_moving_average_files += files('moving_average_cuda.cc')
#     filter_cu_sources += files('moving_average_cuda.cu')
endif

gen_moving_average_h = custom_target('gen_moving_average_h',
                        input : ['moving_average.yml'],
                        output : ['moving_average.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/filter')

gen_moving_average_cc = custom_target('gen_moving_average_cc',
                        input : ['moving_average.yml'],
                        output : ['moving_average.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/filter')

filter_deps += declare_dependency(sources : [gen_moving_average_h] ) 
filter_sources += [filter_moving_average_files, gen_moving_average_cc]

if get_option('enable_python')
    gen_moving_average_pybind = custom_target('gen_moving_average_cpu_pybind',
                            input : ['moving_average.yml'],
                            output : ['moving_average_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    filter_pybind_sources += gen_moving_average_pybind
    filter_pybind_names += 'moving_average'
endif
