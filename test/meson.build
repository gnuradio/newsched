
# GR namespace tests
qa_srcs = ['qa_default_runtime',
           'qa_scheduler_nbt',
           'qa_block_grouping',
           'qa_single_mapped_buffers',
           'qa_message_ports',
           'qa_tags',
           'qa_zmq_buffers',
           'qa_fast_atan2f',
           'qa_fxpt_nco',
           'qa_fxpt_vco',
           'qa_fxpt',
           'qa_math'
          ]
deps = [newsched_gr_dep,
                newsched_blocklib_blocks_dep,
                newsched_blocklib_math_dep,
                newsched_scheduler_nbt_dep,
                gtest_dep,
                gr_kernel_lib_dep]

foreach qa : qa_srcs
    e = executable(qa, 
        qa + '.cc', 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: deps, 
        install : false)
    test(qa, e, env: TEST_ENV)
endforeach

test('Basic Python', py3, args : files('qa_basic.py'), env: TEST_ENV)
test('Block Parameters', py3, args : files('qa_parameters.py'), env: TEST_ENV)
test('Python Blocks', py3, args : files('qa_python_block.py'), env: TEST_ENV)

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
