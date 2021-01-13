incdir = include_directories('../include')

###################################################
#    QA
###################################################

if get_option('enable_testing')
    srcs = ['qa_scheduler_st.cpp']
    e = executable('qa_scheduler_st', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_st_dep,
                    gtest_dep], 
        install : true)
    test('Single Threaded Scheduler Tests', e)
endif