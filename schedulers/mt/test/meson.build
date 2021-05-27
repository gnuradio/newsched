incdir = include_directories('../include')

###################################################
#    QA
###################################################

if get_option('enable_testing')
    srcs = ['qa_scheduler_mt.cc']
    e = executable('qa_scheduler_mt', 
        srcs, 
        include_directories : incdir,
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('Multi Threaded Scheduler Tests', e)

    srcs = ['qa_block_grouping.cc']
    e = executable('qa_block_grouping', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT Block Grouping Tests', e)

    srcs = ['qa_single_mapped_buffers.cc']
    e = executable('qa_single_mapped_buffers', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT Single Mapped Buffers', e)

    srcs = ['qa_message_ports.cc']
    e = executable('qa_message_ports', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    # test('MT Message Port Tests', e)

    srcs = ['qa_tags.cc']
    e = executable('qa_tags', 
        srcs, 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [newsched_runtime_dep,
                    newsched_blocklib_blocks_dep,
                    newsched_scheduler_mt_dep,
                    gtest_dep], 
        install : true)
    test('MT Tags Tests', e)

    # env = environment()
    # env.prepend('PYTHONPATH', join_paths('PYTHONPATH', meson.build_root(),'python'))
    # test('Basic Python', find_program('qa_basic.py'), env: env)

endif

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
