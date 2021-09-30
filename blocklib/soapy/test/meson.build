###################################################
#    QA
###################################################

if get_option('enable_testing')
    env = environment()
    env.prepend('LD_LIBRARY_PATH', join_paths( meson.build_root(),'schedulers','nbt','lib'))
    env.prepend('PYTHONPATH', join_paths(meson.build_root(),'python'))
endif
