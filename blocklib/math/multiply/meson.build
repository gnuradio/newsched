math_multiply_files = files(['multiply_cpu.cc'])

# if cuda_dep.found() and get_option('enable_cuda')
#     math_multiply_files += files('multiply_cuda.cc')
#     math_cu_sources += files('multiply_cuda.cu')
# endif

gen_multiply_h = custom_target('gen_multiply_h',
                        input : ['multiply.yml'],
                        output : ['multiply.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

gen_multiply_cc = custom_target('gen_multiply_cc',
                        input : ['multiply.yml'],
                        output : ['multiply.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

math_deps += declare_dependency(sources : [gen_multiply_h] ) 
math_sources += [math_multiply_files, gen_multiply_cc]

if get_option('enable_python')
    gen_multiply_pybind = custom_target('gen_multiply_cpu_pybind',
                            input : ['multiply.yml'],
                            output : ['multiply_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    math_pybind_sources += gen_multiply_pybind
    math_pybind_names += 'multiply'
endif
