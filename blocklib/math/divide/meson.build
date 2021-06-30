math_divide_files = files(['divide_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
     math_divide_files += files('divide_cuda.cc')
endif

gen_divide_h = custom_target('gen_divide_h',
                        input : ['divide.yml'],
                        output : ['divide.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

gen_divide_cc = custom_target('gen_divide_cc',
                        input : ['divide.yml'],
                        output : ['divide.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

math_deps += declare_dependency(sources : [gen_divide_h] ) 
math_sources += [math_divide_files, gen_divide_cc]

if get_option('enable_python')
    gen_divide_pybind = custom_target('gen_divide_cpu_pybind',
                            input : ['divide.yml'],
                            output : ['divide_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    math_pybind_sources += gen_divide_pybind
    math_pybind_names += 'divide'
endif
