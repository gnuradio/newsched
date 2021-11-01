srcs = files(['freq_sink_cpu.cc'])

# if USE_CUDA
#     srcs += files('freq_sink_cuda.cc')
#     # qtgui_cu_sources += files('freq_sink_cuda.cu')
# endif

gen_srcs = custom_target('gen_freq_sink_h',
                        input : ['freq_sink.yml'],
                        output : ['freq_sink.cc', 'freq_sink.hh', 'freq_sink_pybind.cc', 'qtgui_freq_sink.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/qtgui',false,'share/gnuradio/grc/blocks'])

qtgui_deps += declare_dependency(sources : [gen_srcs[1]] ) 
qtgui_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    qtgui_pybind_sources += gen_srcs[2]
    qtgui_pybind_names += 'freq_sink'
endif
