yml_file = 'msg_forward.yml'

blocks_msg_forward_files = files(['msg_forward_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    # blocks_msg_forward_files += files('msg_forward_cuda.cc')
    # blocks_msg_forward_cu_files = files('msg_forward_cuda.cu')
endif

gen_msg_forward_h = custom_target('gen_msg_forward_cpu_h',
                        input : ['msg_forward.yml'],
                        output : ['msg_forward.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_msg_forward_cc = custom_target('gen_msg_forward_cpu_cc',
                        input : ['msg_forward.yml'],
                        output : ['msg_forward.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        



blocks_deps += declare_dependency(sources : [gen_msg_forward_h])
blocks_sources += [blocks_msg_forward_files, gen_msg_forward_cc]

if get_option('enable_python')
    gen_msg_forward_pybind = custom_target('gen_msg_forward_cpu_pybind',
                        input : ['msg_forward.yml'],
                        output : ['msg_forward_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)                    
    blocks_pybind_sources += gen_msg_forward_pybind
endif
