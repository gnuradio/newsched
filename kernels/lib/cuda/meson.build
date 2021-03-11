incdir = include_directories([
    '../../include/gnuradio/kernels/common',
    '../../include/gnuradio/kernels/cuda', 
    '../../include'
])

srcs = [
    'copy.cu',
    'copy.cpp'
]

newsched_kernels_cuda = library('newsched-kernels-cuda', 
    srcs, 
    include_directories : incdir, 
    install : true, 
    dependencies : [cuda_dep]
)

newsched_kernels_cuda_dep = declare_dependency(
    include_directories : incdir, 
    dependencies : [cuda_dep],
    link_with : newsched_kernels_cuda
)
