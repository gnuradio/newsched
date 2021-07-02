streamops_interleaved_short_to_complex_files = files(['interleaved_short_to_complex_cpu.cc'])

# if cuda_dep.found() and get_option('enable_cuda')
#     streamops_interleaved_short_to_complex_files += files('interleaved_short_to_complex_cuda.cc')
#     streamops_cu_sources += files('interleaved_short_to_complex_cuda.cu')
# endif

gen_interleaved_short_to_complex_h = custom_target('gen_interleaved_short_to_complex_h',
                        input : ['interleaved_short_to_complex.yml'],
                        output : ['interleaved_short_to_complex.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/streamops')

gen_interleaved_short_to_complex_cc = custom_target('gen_interleaved_short_to_complex_cc',
                        input : ['interleaved_short_to_complex.yml'],
                        output : ['interleaved_short_to_complex.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/streamops')

streamops_deps += declare_dependency(sources : [gen_interleaved_short_to_complex_h] ) 
streamops_sources += [streamops_interleaved_short_to_complex_files, gen_interleaved_short_to_complex_cc]

if get_option('enable_python')
    gen_interleaved_short_to_complex_pybind = custom_target('gen_interleaved_short_to_complex_cpu_pybind',
                            input : ['interleaved_short_to_complex.yml'],
                            output : ['interleaved_short_to_complex_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    streamops_pybind_sources += gen_interleaved_short_to_complex_pybind
    streamops_pybind_names += 'interleaved_short_to_complex'
endif
