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

    srcs = ['qa_tags.cpp']
    e = executable('qa_tags', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT Tags Tests', e)

    srcs = ['qa_fileio.cpp']
    e = executable('qa_fileio', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_fileio_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT FileIO Tests', e)

    srcs = ['test_atsc.cpp']
    e = executable('test_atsc', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    newsched_blocklib_analog_dep,
                    newsched_blocklib_filter_dep,
                    newsched_blocklib_dtv_dep,
                    newsched_blocklib_filter_dep,
                    newsched_blocklib_fileio_dep,
                    newsched_blocklib_streamops_dep,
                    newsched_blocklib_math_dep,
                    volk_dep,
                    gtest_dep], 
        install : true)
    

endif

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
