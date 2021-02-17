incdir = include_directories('../include')

###################################################
#    QA
###################################################

if get_option('enable_testing')
    srcs = ['qa_scheduler_mt.cpp']
    e = executable('qa_scheduler_mt', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('Multi Threaded Scheduler Tests', e)

    srcs = ['qa_block_grouping.cpp']
    e = executable('qa_block_grouping', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT Block Grouping Tests', e)

    srcs = ['qa_message_ports.cpp']
    e = executable('qa_message_ports', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    # test('MT Message Port Tests', e)

endif

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
