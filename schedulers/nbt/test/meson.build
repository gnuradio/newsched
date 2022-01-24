incdir = include_directories('../include')

###################################################
#    QA
###################################################

if get_option('enable_testing')
    srcs = ['qa_scheduler_nbt.cc']
    e = executable('qa_scheduler_nbt', 
        srcs, 
        include_directories : incdir,
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_math_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)
    test('Multi Threaded Scheduler Tests', e, env: TEST_ENV)

    srcs = ['qa_block_grouping.cc']
    e = executable('qa_block_grouping', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_math_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)
    test('NBT Block Grouping Tests', e, env: TEST_ENV)

    srcs = ['qa_single_mapped_buffers.cc']
    e = executable('qa_single_mapped_buffers', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_math_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)

    test('NBT Single Mapped Buffers', e, env: TEST_ENV)

    srcs = ['qa_message_ports.cc']
    e = executable('qa_message_ports', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_math_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)
    # test('NBT Message Port Tests', e, env: TEST_ENV)

    srcs = ['qa_tags.cc']
    e = executable('qa_tags', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_math_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)
    test('NBT Tags Tests', e, env: TEST_ENV)

    srcs = ['qa_zmq_buffers.cc']
    e = executable('qa_zmq_buffers', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_blocklib_math_dep,
                    newsched_scheduler_nbt_dep,
                    gtest_dep], 
        install : true)
    test('NBT ZMQ Custom Buffers', e, env: TEST_ENV)

    test('Basic Python', py3, args : files('qa_basic.py'), env: TEST_ENV)
    test('Block Parameters', py3, args : files('qa_parameters.py'), env: TEST_ENV)
    test('Python Blocks', py3, args : files('qa_python_block.py'), env: TEST_ENV)

endif

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
