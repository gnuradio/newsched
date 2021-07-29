dtv_deps += [newsched_runtime_dep, newsched_blocklib_filter_dep, newsched_blocklib_math_dep, newsched_blocklib_fec_dep, volk_dep, fmt_dep, pmtf_dep]

sources = [
    'atsc_plinfo.cc',
    # 'atsc_fpll.cc',
    # 'atsc_sync.cc',
    # 'atsc_fs_checker.cc',
    # 'atsc_equalizer.cc',
    'atsc_single_viterbi.cc',
    # 'atsc_viterbi_decoder.cc',
    # 'atsc_deinterleaver.cc',
    # 'atsc_rs_decoder.cc',
    'atsc_randomize.cc',
    # 'atsc_derandomizer.cc'
]

dtv_sources += sources
block_cpp_args = ['-DHAVE_CPU']
# if cuda_dep.found() and get_option('enable_cuda')
#     block_cpp_args += '-DHAVE_CUDA'

#     newsched_blocklib_dtv_cu = library('newsched-blocklib-dtv-cu', 
#         dtv_cu_sources, 
#         include_directories : incdir, 
#         install : true, 
#         dependencies : [cuda_dep])

#     newsched_blocklib_dtv_cu_dep = declare_dependency(include_directories : incdir,
#                         link_with : newsched_blocklib_dtv_cu,
#                         dependencies : cuda_dep)

#     dtv_deps += [newsched_blocklib_dtv_cu_dep, cuda_dep]

# endif

incdir = include_directories(['../include/gnuradio/dtv','../include'])
newsched_blocklib_dtv_lib = library('newsched-blocklib-dtv', 
    dtv_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : dtv_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_dtv_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_dtv_lib,
                       dependencies : dtv_deps)
