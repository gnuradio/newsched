scheduler_mt_sources = [
    'scheduler_mt.cpp',
]
scheduler_mt_deps = [newsched_runtime_dep, newsched_scheduler_st_dep, threads_dep, fmt_dep, pmtf_dep]

incdir = include_directories('../include', '../include/gnuradio/schedulers/')
newsched_scheduler_mt_lib = library('newsched-scheduler-mt', 
    scheduler_mt_sources, include_directories : incdir, 
    install : true,
    dependencies : scheduler_mt_deps)

newsched_scheduler_mt_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_scheduler_mt_lib,
                       dependencies : scheduler_mt_deps )
