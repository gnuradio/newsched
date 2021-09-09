incdir = include_directories('../../include')

###################################################
#    QA
###################################################

if get_option('enable_testing')
    srcs = ['qa_scheduler_nbt_cuda_copy.cc']
    e = executable('qa_scheduler_nbt_cuda_copy', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_runtime_cu_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)
    test('NBT Cuda Copy Tests', e)

endif

