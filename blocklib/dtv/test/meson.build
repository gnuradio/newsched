###################################################
#    QA
###################################################

if get_option('enable_testing')
    env = environment()
    env.prepend('LD_LIBRARY_PATH', join_paths( meson.build_root(),'schedulers','mt','lib'))
    env.prepend('PYTHONPATH', join_paths(meson.build_root(),'python'))

    # test('qa_agc', find_program('qa_agc.py'), env: env)
    # if (cuda_available and get_option('enable_cuda'))
    # test('qa_cufft', find_program('qa_cufft.py'), env: env)
    # endif

endif
