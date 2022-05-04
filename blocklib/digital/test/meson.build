###################################################
#    QA
###################################################

if get_option('enable_testing')
    test('qa_chunks_to_symbols', find_program('qa_chunks_to_symbols.py'), env: TEST_ENV)
    # test('qa_crc', find_program('qa_crc.py'), env: TEST_ENV)
    # if (cuda_available and get_option('enable_cuda'))
    # test('qa_cufft', find_program('qa_cufft.py'), env: TEST_ENV)
    # endif

endif
