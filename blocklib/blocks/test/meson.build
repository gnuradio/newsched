###################################################
#    QA
###################################################

env = environment()
env.prepend('LD_LIBRARY_PATH', join_paths( meson.build_root(),'schedulers','nbt','lib'))
env.prepend('PYTHONPATH', join_paths(meson.build_root(),'python'))

test('qa_delay', find_program('qa_delay.py'), env: env)

if (cuda_available and get_option('enable_cuda'))
subdir('cuda')
endif
