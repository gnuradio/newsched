yml_file = 'copy.yml'

blocks_copy_files = files(['copy_cpu.cc'])


if cuda_dep.found() and get_option('enable_cuda')
    blocks_copy_files += files('copy_cuda.cc')
    blocks_copy_cu_files = files('copy_cuda.cu')
endif

# Copy the hpp file to the build directory
# gen_copy_h = custom_target('gen_copy_cpu_h',
#                         input : ['copy.hh'],
#                         output : ['copy.hh'],
#                         command : ['python3', join_paths(meson.source_root(),'utils/copyfile.py'),
#                             '--input_file', '@INPUT@', 
#                             '--output_file', '@OUTPUT@', 
#                             '--output_dir', join_paths(meson.build_root(), 'blocklib', 'blocks', 'include', 'gnuradio', 'blocks')],
#                         install : true,
#                         install_dir : 'include/gnuradio/blocks')

gen_copy_h = custom_target('gen_copy_cpu_h',
                        input : ['copy.yml'],
                        output : ['copy.hh'], #,'copy.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')

gen_copy_cc = custom_target('gen_copy_cpu_cc',
                        input : ['copy.yml'],
                        output : ['copy.cc'], #,'copy.cc'],
                        command : ['python3', join_paths(meson.source_root(),'utils/blockbuilder/scripts/process_folder.py'),
                            '--yaml_file', '@INPUT@', 
                            '--output_file', '@OUTPUT@', 
                            '--build_dir', join_paths(meson.build_root())],
                        install : true,
                        install_dir : 'include/gnuradio/blocks')                        

blocks_deps += declare_dependency(sources : [gen_copy_h])
blocks_sources += [blocks_copy_files, gen_copy_cc]