###################################################
#    QA
###################################################

test('qa_delay', find_program('qa_delay.py'), env: TEST_ENV)

if (cuda_available and get_option('enable_cuda'))
subdir('cuda')
endif
