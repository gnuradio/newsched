srcs = files(['msg_forward_cpu.cc'])

# if USE_CUDA
#     srcs += files('msg_forward_cuda.cc')
#     blocks_cu_sources += files('msg_forward_cuda.cu')
# endif

gen_srcs = custom_target('gen_msg_forward_h',
                        input : ['msg_forward.yml'],
                        output : ['msg_forward.cc', 'msg_forward.hh', 'msg_forward_pybind.cc', 'blocks_msg_forward.block.yml'],
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
    blocks_pybind_names += 'msg_forward'
endif
