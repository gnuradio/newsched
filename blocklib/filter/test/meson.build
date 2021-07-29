###################################################
#    QA
###################################################

if get_option('enable_testing')
    env = environment()
    env.prepend('LD_LIBRARY_PATH', join_paths( meson.build_root(),'schedulers','mt','lib'))
    env.prepend('PYTHONPATH', join_paths(meson.build_root(),'python'))

    # test('qa_fir_filter', find_program('qa_fir_filter.py'), env: env)
    test('qa_moving_average', find_program('qa_moving_average.py'), env: env)
    # if (cuda_available and get_option('enable_cuda'))
    # test('qa_cufft', find_program('qa_cufft.py'), env: env)
    # endif

endif
