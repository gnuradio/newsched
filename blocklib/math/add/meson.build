math_add_files = files(['add_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    math_add_files += files('add_cuda.cc')
    # math_cu_sources += files('add_cuda.cu')
endif

gen_add_h = custom_target('gen_add_h',
                        input : ['add.yml'],
                        output : ['add.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

gen_add_cc = custom_target('gen_add_cc',
                        input : ['add.yml'],
                        output : ['add.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

math_deps += declare_dependency(sources : [gen_add_h] ) 
math_sources += [math_add_files, gen_add_cc]

if get_option('enable_python')
    gen_add_pybind = custom_target('gen_add_cpu_pybind',
                            input : ['add.yml'],
                            output : ['add_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    math_pybind_sources += gen_add_pybind
    math_pybind_names += 'add'
endif
