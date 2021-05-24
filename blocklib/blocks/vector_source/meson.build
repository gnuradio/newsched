
blocks_vector_source_files = files(['vector_source_cpu.cc'])

# if cuda_dep.found() and get_option('enable_cuda')
#     blocks_vector_source_files += files('vector_source_cuda.cc')
#     blocks_vector_source_cu_files = files('vector_source_cuda.cu')
# endif

gen_vector_source_h = custom_target('gen_vector_source_h',
                        input : ['vector_source.yml'],
                        output : ['vector_source.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_vector_source_cc = custom_target('gen_vector_source_cc',
                        input : ['vector_source.yml'],
                        output : ['vector_source.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')


blocks_deps += declare_dependency(sources : [gen_vector_source_h] ) 
blocks_sources += [blocks_vector_source_files, gen_vector_source_cc]
