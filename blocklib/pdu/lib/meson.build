pdu_sources += []
pdu_deps += [gnuradio_gr_dep, volk_dep, fmt_dep, pmtf_dep, python3_embed_dep]

block_cpp_args = ['-DHAVE_CPU']

# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     gnuradio_blocklib_pdu_cu = library('gnuradio-blocklib-pdu-cu', 
#         pdu_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     gnuradio_blocklib_pdu_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : gnuradio_blocklib_pdu_cu,
#                         dependencies : cuda_dep)

#     pdu_deps += [gnuradio_blocklib_pdu_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/pdu','../include'])
gnuradio_blocklib_pdu_lib = library('gnuradio-blocklib-pdu', 
    pdu_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : pdu_deps,
    link_args : ['-lgnuradio-runtime'],
    cpp_args : block_cpp_args)

gnuradio_blocklib_pdu_dep = declare_dependency(include_directories : incdir,
					   link_with : gnuradio_blocklib_pdu_lib,
                       dependencies : pdu_deps)
