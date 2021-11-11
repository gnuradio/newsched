###################################################
#    QA
###################################################

test('qa_delay', py3, args : files('qa_delay.py'), env: TEST_ENV)

if (cuda_available and get_option('enable_cuda'))
subdir('cuda')
endif
