yml_file = 'conjugate.yml'

math_conjugate_files = files(['conjugate_cpu.cc'])


# if cuda_dep.found() and get_option('enable_cuda')
#     math_conjugate_files += files('conjugate_cuda.cc')
#     math_cu_sources += files('conjugate_cuda.cu')
# endif

gen_conjugate_h = custom_target('gen_conjugate_cpu_h',
                        input : ['conjugate.yml'],
                        output : ['conjugate.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

gen_conjugate_cc = custom_target('gen_conjugate_cpu_cc',
                        input : ['conjugate.yml'],
                        output : ['conjugate.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

math_deps += declare_dependency(sources : [gen_conjugate_h])
math_sources += [math_conjugate_files, gen_conjugate_cc]

if get_option('enable_python')
    gen_conjugate_pybind = custom_target('gen_conjugate_cpu_pybind',
                            input : ['conjugate.yml'],
                            output : ['conjugate_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    math_pybind_sources += gen_conjugate_pybind
    math_pybind_names += 'conjugate'
endif
