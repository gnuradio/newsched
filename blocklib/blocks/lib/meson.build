blocks_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep]

block_cpp_args = ['-DHAVE_CPU']
if cuda_dep.found() and get_option('enable_cuda')
    block_cpp_args += '-DHAVE_CUDA'

    newsched_blocklib_blocks_cu = library('newsched-blocklib-blocks-cu', 
        blocks_cu_sources, 
        include_directories : incdir, 
        install : true, 
        dependencies : [cuda_dep])

    newsched_blocklib_blocks_cu_dep = declare_dependency(include_directories : incdir,
                        link_with : newsched_blocklib_blocks_cu,
                        dependencies : cuda_dep)

    blocks_deps += [newsched_blocklib_blocks_cu_dep, cuda_dep]

endif

incdir = include_directories(['../include/gnuradio/blocks','../include'])
newsched_blocklib_blocks_lib = library('newsched-blocklib-blocks', 
    blocks_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : blocks_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_blocks_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_blocks_lib,
                       dependencies : blocks_deps)
