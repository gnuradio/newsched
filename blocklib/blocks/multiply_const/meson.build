blocks_multiply_const_files = files(['multiply_const_cpu.cc'])

if cuda_dep.found() and get_option('enable_cuda')
    blocks_multiply_const_files += files('multiply_const_cuda.cc')
    blocks_multiply_const_cu_files = files('multiply_const_cuda.cu')
endif

# # Copy the hpp file to the build directory
# gen_multiply_const_h = custom_target('gen_multiply_const_cpu_h',
#                         input : ['multiply_const.hh'],
#                         output : ['multiply_const.hh'],
#                         command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
#                             '--input_file', '@INPUT@', 
#                             '--output_file', '@OUTPUT@', 
#                             '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
#                         install : false,
#                         install_dir : 'include/gnuradio/blocks')

gen_multiply_const_h = custom_target('gen_multiply_const_h',
                        input : ['multiply_const.yml'],
                        output : ['multiply_const.hh'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_multiply_const_cc = custom_target('gen_multiply_const_cc',
                        input : ['multiply_const.yml'],
                        output : ['multiply_const.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')


blocks_deps += declare_dependency(sources : [gen_multiply_const_h] )  #, gen_multiply_const_cc])
blocks_sources += [blocks_multiply_const_files, gen_multiply_const_cc]
# blocks_multiply_const_deps = [gen_multiply_const_h, gen_multiply_const_cc]