############################################################################
#    CUDA Kernels to be compiled with NVCC
############################################################################
incdir = include_directories(['../../include/'])

srcs = [
    'add.cu',
    'filterN.cu',
    'adaptN.cu',
    'multiply.cu',
    'dot.cu',
    'atsc_sync.cu',
    'viterbi_decoder.cu',
    'atsc_fs_checker.cu'
]

newsched_blocklib_dtv_cu = library('newsched-blocklib-cuda-cu', 
    srcs, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, cuda_dep])

newsched_blocklib_dtv_cu_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_dtv_cu)

sources = [
    'atsc_sync_cuda.cpp',
    'atsc_fs_checker_cuda.cpp',
    'atsc_equalizer_cuda.cpp',
    'atsc_viterbi_decoder_cuda.cpp',
]

incdir = include_directories(['../../include/gnuradio/dtv','../../include'])
newsched_blocklib_dtv_cuda_lib = library('newsched-blocklib-dtv', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep, newsched_blocklib_filter_dep,
                    newsched_blocklib_dtv_cu_dep])

newsched_blocklib_dtv_cuda_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_dtv_cuda_lib,
                       dependencies : [newsched_blocklib_fec_dep])