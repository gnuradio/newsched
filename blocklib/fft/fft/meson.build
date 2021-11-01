srcs = files(['fft_cpu.cc'])

if USE_CUDA
    srcs += files('fft_cuda.cc')
    fft_cu_sources += files('fft_shift.cu')
endif

gen_srcs = custom_target('gen_fft_h',
                        input : ['fft.yml'],
                        output : ['fft.cc', 'fft.hh', 'fft_pybind.cc', 'fft_fft.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/fft',false,'share/gnuradio/grc/blocks'])

fft_deps += declare_dependency(sources : [gen_srcs[1]] ) 
fft_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    fft_pybind_sources += gen_srcs[2]
    fft_pybind_names += 'fft'
endif
