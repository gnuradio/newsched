blocks_multiply_const_files = files(['multiply_const_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    blocks_multiply_const_files += files('multiply_const_cuda.cc')
    blocks_cu_sources += files('multiply_const_cuda.cu')
endif

gen_multiply_const_h = custom_target('gen_multiply_const_h',
                        input : ['multiply_const.yml'],
                        output : ['multiply_const.hh'],
                        command : ['python3', join_paths(SCRIPTS_DIR,'process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_multiply_const_cc = custom_target('gen_multiply_const_cc',
                        input : ['multiply_const.yml'],
                        output : ['multiply_const.cc'],
                        command : ['python3', join_paths(SCRIPTS_DIR,'process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

blocks_deps += declare_dependency(sources : [gen_multiply_const_h] ) 
blocks_sources += [blocks_multiply_const_files, gen_multiply_const_cc]

if get_option('enable_python')
    gen_multiply_const_pybind = custom_target('gen_multiply_const_cpu_pybind',
                            input : ['multiply_const.yml'],
                            output : ['multiply_const_pybind.cc'],
                            command : ['python3', join_paths(SCRIPTS_DIR,'process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    blocks_pybind_sources += gen_multiply_const_pybind
    blocks_pybind_names += 'multiply_const'
endif
