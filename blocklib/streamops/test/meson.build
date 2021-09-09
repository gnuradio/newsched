###################################################
#    QA
###################################################

if get_option('enable_testing')
    env = environment()
    env.prepend('LD_LIBRARY_PATH', join_paths( meson.build_root(),'schedulers','nbt','lib'))
    env.prepend('PYTHONPATH', join_paths(meson.build_root(),'python'))

    test('qa_type_conversions', find_program('qa_type_conversions.py'), env: env)
    # if (cuda_available and get_option('enable_cuda'))
    # test('qa_cufft', find_program('qa_cufft.py'), env: env)
    # endif

endif
