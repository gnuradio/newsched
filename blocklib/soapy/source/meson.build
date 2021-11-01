srcs = files(['source_cpu.cc'])

# if USE_CUDA
#     srcs += files('source_cuda.cc')
#     # soapy_cu_sources += files('source_cuda.cu')
# endif

gen_srcs = custom_target('gen_source_h',
                        input : ['source.yml'],
                        output : ['source.cc', 'source.hh', 'source_pybind.cc', 'soapy_source.block.yml'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output-cc', '@OUTPUT0@', 
                            '--output-hh', '@OUTPUT1@', 
                            '--output-pybind', '@OUTPUT2@', 
                            '--output-grc', '@OUTPUT3@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : [false, 'include/gnuradio/soapy',false,'share/gnuradio/grc/blocks'])

soapy_deps += declare_dependency(sources : [gen_srcs[1]] ) 
soapy_sources += [srcs, gen_srcs[0]]

if get_option('enable_python')                           
    soapy_pybind_sources += gen_srcs[2]
    soapy_pybind_names += 'source'
endif
