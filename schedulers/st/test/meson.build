test_st_sources = ['test_scheduler_st.cpp']

incdir = include_directories('../include')
executable('test_scheduler_st', 
    test_st_sources, 
    include_directories : incdir, 
    dependencies: [newsched_runtime_dep,
                   newsched_blocklib_blocks_dep,
                   newsched_scheduler_st_dep], 
    install : true)
# executable('test_pmt', test_pmt_sources, include_directories : incdir, install : true)


