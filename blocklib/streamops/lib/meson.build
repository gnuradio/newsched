sources = [
]

streamops_sources += sources
streamops_deps += [newsched_gr_dep, volk_dep, fmt_dep, pmtf_dep]

block_cpp_args = ['-DHAVE_CPU']
link_args = []
if USE_CUDA
    block_cpp_args += '-DHAVE_CUDA'

    # newsched_blocklib_streamops_cu = library('newsched-blocklib-streamops-cu', 
    #     streamops_cu_sources, 
    #     include_directories : incdir, 
    #     install : true, 
    #     dependencies : [cuda_dep])

    # newsched_blocklib_streamops_cu_dep = declare_dependency(include_directories : incdir,
    #                     link_with : newsched_blocklib_streamops_cu,
    #                     dependencies : cuda_dep)

    streamops_deps += [cuda_dep, cusp_dep]
    link_args += ['-lcusp']

endif

incdir = include_directories(['../include/gnuradio/streamops','../include'])
newsched_blocklib_streamops_lib = library('newsched-blocklib-streamops', 
    streamops_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    link_args : link_args,
    dependencies : streamops_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_streamops_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_streamops_lib,
                       dependencies : streamops_deps)
