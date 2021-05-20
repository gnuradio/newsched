incdir = include_directories('../include')

srcs = ['bm_copy.cc']
executable('bm_mt_copy', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_mt_dep,
                   boost_dep], 
    install : true)

srcs = ['bm_fanout.cc']
executable('bm_mt_fanout', 
    srcs, 
    include_directories : incdir,
    link_language : 'cpp', 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_mt_dep,
                   boost_dep], 
    install : true)

srcs = ['bm_nop.cc']
executable('bm_mt_nop', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_mt_dep,
                   boost_dep], 
    install : true)

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif