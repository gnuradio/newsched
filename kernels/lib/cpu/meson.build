incdir = include_directories(['../../include/gnuradio/kernels/cpu', '../../include'])

srcs = [
    'copy.cpp'
]

newsched_kernels_cpu = library('newsched-kernels-cpu', 
    srcs, 
    include_directories : incdir, 
    install : true, 
    dependencies : []
)

newsched_kernels_cpu_dep = declare_dependency(
    include_directories : incdir, 
    link_with : newsched_kernels_cpu
)
