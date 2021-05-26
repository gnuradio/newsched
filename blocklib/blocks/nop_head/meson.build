yml_file = 'nop_head.yml'

blocks_nop_head_files = files(['nop_head_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_nop_head_files += files('nop_head_cuda.cc')
#     blocks_nop_head_cu_files = files('nop_head_cuda.cu')
# endif

gen_nop_head_h = custom_target('gen_nop_head_cpu_h',
                        input : ['nop_head.yml'],
                        output : ['nop_head.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_nop_head_cc = custom_target('gen_nop_head_cpu_cc',
                        input : ['nop_head.yml'],
                        output : ['nop_head.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')                        

blocks_deps += declare_dependency(sources : [gen_nop_head_h])
blocks_sources += [blocks_nop_head_files, gen_nop_head_cc]

              

if get_option('enable_python')
    gen_nop_head_pybind = custom_target('gen_nop_head_cpu_pybind',
                            input : ['nop_head.yml'],
                            output : ['nop_head_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)      
    blocks_pybind_sources += gen_nop_head_pybind
    blocks_pybind_names += 'nop_head'
endif
