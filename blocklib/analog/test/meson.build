###################################################
#    QA
###################################################

if get_option('enable_testing')
    test('qa_sig_source', find_program('qa_sig_source.py'), env: TEST_ENV)
    test('qa_noise', find_program('qa_noise.py'), env: TEST_ENV)
    # test('qa_agc', find_program('qa_agc.py'), env: TEST_ENV)
    # if (cuda_available and get_option('enable_cuda'))
    # test('qa_cufft', find_program('qa_cufft.py'), env: TEST_ENV)
    # endif

endif
