srcs = files(['file_sink_cpu.cc'])

# if USE_CUDA
#     srcs += files('file_sink_cuda.cc')
#     # fileio_cu_sources += files('file_sink_cuda.cu')
# endif

gen_srcs = custom_target('gen_file_sink_h',
                        input : ['file_sink.yml'],
                        output : ['file_sink.cc', 'file_sink.hh', 'file_sink_pybind.cc', 'fileio_file_sink.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/fileio',false,'share/gnuradio/grc/blocks'])

fileio_deps += declare_dependency(sources : [gen_srcs[1]] ) 
fileio_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    fileio_pybind_sources += gen_srcs[2]
    fileio_pybind_names += 'file_sink'
endif
