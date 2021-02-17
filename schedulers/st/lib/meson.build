scheduler_st_sources = [
    'buffer_management.cpp',
    'graph_executor.cpp',
    'scheduler_st.cpp',
    'thread_wrapper.cpp'
]
scheduler_st_deps = [newsched_runtime_dep, threads_dep, fmt_dep, pmtf_dep]

incdir = include_directories('../include', '../include/gnuradio/schedulers/st')
newsched_scheduler_st_lib = library('newsched-scheduler-st', 
    scheduler_st_sources, include_directories : incdir, 
    install : true,
    dependencies : scheduler_st_deps)

newsched_scheduler_st_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_scheduler_st_lib,
                       dependencies : scheduler_st_deps )
