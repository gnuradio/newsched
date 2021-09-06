filter_pfb_channelizer_files = files(['pfb_channelizer_cpu.cc'])

if USE_CUDA
    filter_pfb_channelizer_files += files('pfb_channelizer_cuda.cc')
#     filter_cu_sources += files('pfb_channelizer_cuda.cu')
endif

gen_pfb_channelizer_h = custom_target('gen_pfb_channelizer_h',
                        input : ['pfb_channelizer.yml'],
                        output : ['pfb_channelizer.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/filter')

gen_pfb_channelizer_cc = custom_target('gen_pfb_channelizer_cc',
                        input : ['pfb_channelizer.yml'],
                        output : ['pfb_channelizer.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/filter')

filter_deps += declare_dependency(sources : [gen_pfb_channelizer_h] ) 
filter_sources += [filter_pfb_channelizer_files, gen_pfb_channelizer_cc]

if get_option('enable_python')
    gen_pfb_channelizer_pybind = custom_target('gen_pfb_channelizer_cpu_pybind',
                            input : ['pfb_channelizer.yml'],
                            output : ['pfb_channelizer_pybind.cc'],
                            command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_pybind.py'),
                                '--yaml_file', '@INPUT@', 
                                '--output_file', '@OUTPUT@', 
                                '--build_dir', join_paths(meson.build_root())],
                            install : false)   
                            
    filter_pybind_sources += gen_pfb_channelizer_pybind
    filter_pybind_names += 'pfb_channelizer'
endif
