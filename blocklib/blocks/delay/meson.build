yml_file = 'delay.yml'

blocks_delay_files = files(['delay_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_delay_files += files('delay_cuda.cc')
#     blocks_cu_sources += files('delay_cuda.cu')
# endif

gen_delay_h = custom_target('gen_delay_cpu_h',
                        input : ['delay.yml'],
                        output : ['delay.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_delay_cc = custom_target('gen_delay_cpu_cc',
                        input : ['delay.yml'],
                        output : ['delay.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

blocks_deps += declare_dependency(sources : [gen_delay_h])
blocks_sources += [blocks_delay_files, gen_delay_cc]

if get_option('enable_python')
    gen_delay_pybind = custom_target('gen_delay_cpu_pybind',
                            input : ['delay.yml'],
                            output : ['delay_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    blocks_pybind_sources += gen_delay_pybind
    blocks_pybind_names += 'delay'
endif
