test('qa_conjugate', find_program('qa_conjugate.py'), env: TEST_ENV)
test('qa_type_conversions(math)', find_program('qa_type_conversions.py'), env: TEST_ENV)
test('qa_add_mult_div_sub', find_program('qa_add_mult_div_sub.py'), env: TEST_ENV)
if (cuda_available and get_option('enable_cuda'))
subdir('cuda')
endif