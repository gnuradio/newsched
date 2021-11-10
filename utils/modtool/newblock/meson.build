srcs = files(['newblock_cpu.cc'])

# if USE_CUDA
#     srcs += files('newblock_cuda.cc')
#     # newmod_cu_sources += files('newblock_cuda.cu')
# endif

gen_srcs = custom_target('gen_newblock_h',
                        input : ['newblock.yml'],
                        output : ['newblock.cc', 'newblock.hh', 'newblock_pybind.cc', 'newmod_newblock.block.yml'],
                        command : ['python3', join_paths(SCRIPTS_DIR,'process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.project_build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/newmod',false,'share/gnuradio/grc/blocks'])

newmod_deps += declare_dependency(sources : [gen_srcs[1]] ) 
newmod_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    newmod_pybind_sources += gen_srcs[2]
    newmod_pybind_names += 'newblock'
endif
