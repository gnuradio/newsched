srcs = files(['dc_blocker_cpu.cc'])

# if USE_CUDA
#     srcs += files('dc_blocker_cuda.cc')
#     # filter_cu_sources += files('dc_blocker_cuda.cu')
# endif

gen_srcs = custom_target('gen_dc_blocker_h',
                        input : ['dc_blocker.yml'],
                        output : ['dc_blocker.cc', 'dc_blocker.hh', 'dc_blocker_pybind.cc', 'filter_dc_blocker.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/filter',false,'share/gnuradio/grc/blocks'])

filter_deps += declare_dependency(sources : [gen_srcs[1]] ) 
filter_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    filter_pybind_sources += gen_srcs[2]
    filter_pybind_names += 'dc_blocker'
endif
