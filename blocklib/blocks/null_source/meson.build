yml_file = 'null_source.yml'

blocks_null_source_files = files(['null_source_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    # blocks_null_source_files += files('null_source_cuda.cc')
    # blocks_null_source_cu_files = files('null_source_cuda.cu')
endif

gen_null_source_h = custom_target('gen_null_source_cpu_h',
                        input : ['null_source.yml'],
                        output : ['null_source.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_null_source_cc = custom_target('gen_null_source_cpu_cc',
                        input : ['null_source.yml'],
                        output : ['null_source.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)        

gen_null_source_pybind = custom_target('gen_null_source_cpu_pybind',
                        input : ['null_source.yml'],
                        output : ['null_source_pybind.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : false)                    

blocks_deps += declare_dependency(sources : [gen_null_source_h])
blocks_sources += [blocks_null_source_files, gen_null_source_cc]

if get_option('enable_python')
    #blocks_pybind_sources += files('null_source_pybind.cc')
    blocks_pybind_sources += gen_null_source_pybind
endif
