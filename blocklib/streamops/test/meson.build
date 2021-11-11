###################################################
#    QA
###################################################

if get_option('enable_testing')
    test('qa_type_conversions', py3, args : files('qa_type_conversions.py'), env: TEST_ENV)
    # if (cuda_available and get_option('enable_cuda'))
    # test('qa_cufft', find_program('qa_cufft.py'), env: TEST_ENV)
    # endif

endif
