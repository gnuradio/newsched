yml_file = 'complex_to_mag_squared.yml'

math_complex_to_mag_squared_files = files(['complex_to_mag_squared_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
     math_complex_to_mag_squared_files += files('complex_to_mag_squared_cuda.cc')
endif

gen_complex_to_mag_squared_h = custom_target('gen_complex_to_mag_squared_cpu_h',
                        input : ['complex_to_mag_squared.yml'],
                        output : ['complex_to_mag_squared.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/math')

gen_complex_to_mag_squared_cc = custom_target('gen_complex_to_mag_squared_cpu_cc',
                        input : ['complex_to_mag_squared.yml'],
                        output : ['complex_to_mag_squared.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

math_deps += declare_dependency(sources : [gen_complex_to_mag_squared_h])
math_sources += [math_complex_to_mag_squared_files, gen_complex_to_mag_squared_cc]

if get_option('enable_python')
    gen_complex_to_mag_squared_pybind = custom_target('gen_complex_to_mag_squared_cpu_pybind',
                            input : ['complex_to_mag_squared.yml'],
                            output : ['complex_to_mag_squared_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)                   
    math_pybind_sources += gen_complex_to_mag_squared_pybind
    math_pybind_names += 'complex_to_mag_squared'
endif
