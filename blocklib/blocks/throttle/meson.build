yml_file = 'throttle.yml'

blocks_throttle_files = files(['throttle_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_throttle_files += files('throttle_cuda.cc')
#     blocks_cu_sources += files('throttle_cuda.cu')
# endif

gen_throttle_h = custom_target('gen_throttle_cpu_h',
                        input : ['throttle.yml'],
                        output : ['throttle.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_throttle_cc = custom_target('gen_throttle_cpu_cc',
                        input : ['throttle.yml'],
                        output : ['throttle.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

blocks_deps += declare_dependency(sources : [gen_throttle_h])
blocks_sources += [blocks_throttle_files, gen_throttle_cc]

if get_option('enable_python')
    gen_throttle_pybind = custom_target('gen_throttle_cpu_pybind',
                            input : ['throttle.yml'],
                            output : ['throttle_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    blocks_pybind_sources += gen_throttle_pybind
    blocks_pybind_names += 'throttle'
endif
