incdir = include_directories(['../include/gnuradio/blocklib/cuda','../include'])

srcs = [
    'copy.cu',
    'fft.cu',
    'multiply_const.cu',
    'multiply_const_doublecopy.cu'
]

newsched_blocklib_cuda_cu = library('newsched-blocklib-cuda-cu', 
    srcs, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, cuda_dep])

newsched_blocklib_cuda_cu_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_cuda_cu)

srcs = [
    'copy.cpp',
    'fft.cpp',
    'multiply_const.cpp',
    'multiply_const_doublecopy.cpp'
]

incdir = include_directories(['../include/gnuradio/blocklib/cuda','../include'])
newsched_blocklib_cuda_lib = library('newsched-blocklib-cuda', 
    srcs, 
    include_directories : incdir,
    link_language : 'cpp',
    install : true, 
    dependencies : [newsched_runtime_dep, cuda_dep, newsched_blocklib_cuda_cu_dep])

newsched_blocklib_cuda_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_cuda_lib)
