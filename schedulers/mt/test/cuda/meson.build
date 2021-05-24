incdir = include_directories('../../include')

###################################################
#    QA
###################################################

if get_option('enable_testing')
    srcs = ['qa_scheduler_mt_cuda_copy.cc']
    e = executable('qa_scheduler_mt_cuda_copy', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_runtime_cu_dep,
                    newsched_blocklib_blocks_dep,
                    cuda_dep, # don't know why this has to go here
                    newsched_blocklib_cuda_dep,
                    newsched_blocklib_cuda_cu_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT Cuda Copy Tests', e)

endif

