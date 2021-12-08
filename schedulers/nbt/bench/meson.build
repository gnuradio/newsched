incdir = include_directories('../include')

if (boost_po_dep.found())
srcs = ['bm_copy.cc']
executable('bm_nbt_copy', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_nbt_dep,
                   boost_po_dep], 
    install : true)

srcs = ['bm_fanout.cc']
executable('bm_nbt_fanout', 
    srcs, 
    include_directories : incdir,
    link_language : 'cpp', 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_nbt_dep,
                   boost_po_dep], 
    install : true)

srcs = ['bm_nop.cc']
executable('bm_nbt_nop', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_nbt_dep,
                   boost_po_dep], 
    install : true)

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
endif