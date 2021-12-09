incdir = include_directories('../../include')

srcs = ['bm_copy.cc']
executable('bm_nbt_cuda_copy', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   cuda_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_nbt_dep,
                   boost_po_dep], 
    install : true)
