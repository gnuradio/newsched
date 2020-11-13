incdir = include_directories('../include')

srcs = ['bm_copy.cpp']
executable('bm_copy', 
    srcs, 
    include_directories : incdir, 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_st_dep,
                   boost_dep], 
    install : true)

srcs = ['bm_fanout.cpp']
executable('bm_fanout', 
    srcs, 
    include_directories : incdir, 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_st_dep,
                   boost_dep], 
    install : true)