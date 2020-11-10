incdir = include_directories('../include')

srcs = ['test_scheduler_st.cpp']
executable('test_scheduler_st', 
    srcs, 
    include_directories : incdir, 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_st_dep], 
    install : true)


srcs = ['test_buffers.cpp']
executable('test_buffers', 
    srcs, 
    include_directories : incdir, 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_st_dep], 
    install : true)

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
                    catch_dep], 
        install : false)
    test('Single Threaded Scheduler Tests', e)
endif