incdir = include_directories([
    '../../include/gnuradio/kernels/common',
    '../../include/gnuradio/kernels/cuda', 
    '../../include'
])

srcs = []

newsched_kernels_cuda = library('newsched-kernels-cuda', 
    srcs, 
    include_directories : incdir, 
    install : true, 
    dependencies : []
)

newsched_kernels_cuda_dep = declare_dependency(
    include_directories : incdir, 
    link_with : newsched_kernels_cuda
)
