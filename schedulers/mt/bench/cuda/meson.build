incdir = include_directories('../../include')

srcs = ['bm_copy.cpp']
executable('bm_mt_cuda_copy', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   newsched_runtime_cu_dep,
                   cuda_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_blocklib_cuda_dep,
                   newsched_blocklib_cuda_cu_dep,
                   newsched_scheduler_mt_dep,
                   boost_dep], 
    install : true)

srcs = ['bm_cufft.cpp']
executable('bm_mt_cuda_fft', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   newsched_runtime_cu_dep,
                   cuda_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_blocklib_cuda_dep,
                   newsched_blocklib_cuda_cu_dep,
                   newsched_scheduler_mt_dep,
                   boost_dep], 
    install : true)