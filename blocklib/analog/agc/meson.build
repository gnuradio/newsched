srcs = files(['agc_cpu.cc'])

# if USE_CUDA
#     srcs += files('agc_cuda.cc')
#     analog_cu_sources += files('agc_cuda.cu')
# endif

gen_srcs = custom_target('gen_agc_h',
                        input : ['agc.yml'],
                        output : ['agc.cc', 'agc.hh', 'agc_pybind.cc', 'analog_agc.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/analog',false,'share/gnuradio/grc/blocks'])

analog_deps += declare_dependency(sources : [gen_srcs[1]] ) 
analog_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    analog_pybind_sources += gen_srcs[2]
    analog_pybind_names += 'agc'
endif
