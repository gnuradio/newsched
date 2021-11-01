srcs = files(['null_source_cpu.cc'])

# if USE_CUDA
#     srcs += files('null_source_cuda.cc')
#     blocks_cu_sources += files('null_source_cuda.cu')
# endif

gen_srcs = custom_target('gen_null_source_h',
                        input : ['null_source.yml'],
                        output : ['null_source.cc', 'null_source.hh', 'null_source_pybind.cc', 'blocks_null_source.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/blocks',false,'share/gnuradio/grc/blocks'])

blocks_deps += declare_dependency(sources : [gen_srcs[1]] ) 
blocks_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    blocks_pybind_sources += gen_srcs[2]
    blocks_pybind_names += 'null_source'
endif
