scheduler_mt_sources = [
    'graph_executor.cc',
    'thread_wrapper.cc',
    'scheduler_mt.cc',
]
scheduler_mt_deps = [newsched_runtime_dep, threads_dep, fmt_dep, pmtf_dep]

incdir = include_directories('../include', '../include/gnuradio/schedulers/mt')
newsched_scheduler_mt_lib = library('newsched-scheduler-mt', 
    scheduler_mt_sources, include_directories : incdir, 
    install : true,
    link_language : 'cpp',
    dependencies : scheduler_mt_deps)

newsched_scheduler_mt_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_scheduler_mt_lib,
                       dependencies : scheduler_mt_deps )
