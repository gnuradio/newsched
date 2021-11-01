srcs = files(['stream_to_streams_cpu.cc'])

if USE_CUDA
    srcs += files('stream_to_streams_cuda.cc')
    # streamops_cu_sources += files('stream_to_streams_cuda.cu')
endif

gen_srcs = custom_target('gen_stream_to_streams_h',
                        input : ['stream_to_streams.yml'],
                        output : ['stream_to_streams.cc', 'stream_to_streams.hh', 'stream_to_streams_pybind.cc', 'streamops_stream_to_streams.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/streamops',false,'share/gnuradio/grc/blocks'])

streamops_deps += declare_dependency(sources : [gen_srcs[1]] ) 
streamops_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    streamops_pybind_sources += gen_srcs[2]
    streamops_pybind_names += 'stream_to_streams'
endif
