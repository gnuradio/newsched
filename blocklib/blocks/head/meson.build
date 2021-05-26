yml_file = 'head.yml'

blocks_head_files = files(['head_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_head_files += files('head_cuda.cc')
#     blocks_head_cu_files = files('head_cuda.cu')
# endif

gen_head_h = custom_target('gen_head_cpu_h',
                        input : ['head.yml'],
                        output : ['head.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_head_cc = custom_target('gen_head_cpu_cc',
                        input : ['head.yml'],
                        output : ['head.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')                        

blocks_deps += declare_dependency(sources : [gen_head_h])
blocks_sources += [blocks_head_files, gen_head_cc]

if get_option('enable_python')
    gen_head_pybind = custom_target('gen_head_cpu_pybind',
                            input : ['head.yml'],
                            output : ['head_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)    
                            
    blocks_pybind_sources += gen_head_pybind
    blocks_pybind_names += 'head'
endif
