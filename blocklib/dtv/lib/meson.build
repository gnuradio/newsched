sources = [
    'atsc_plinfo.cpp',
    'atsc_fpll.cpp',
    'atsc_sync.cpp',
    'atsc_fs_checker.cpp',
    'atsc_equalizer.cpp',
    'atsc_single_viterbi.cpp',
    'atsc_viterbi_decoder.cpp',
    'atsc_deinterleaver.cpp',
    'atsc_rs_decoder.cpp',
    'atsc_randomize.cpp',
    'atsc_derandomizer.cpp'
]

incdir = include_directories(['../include/gnuradio/dtv','../include'])
newsched_blocklib_dtv_lib = library('newsched-blocklib-dtv', 
    sources, 
    include_directories : incdir, 
    install : true, 
    dependencies : [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep,
                    newsched_blocklib_math_dep, newsched_blocklib_filter_dep, newsched_blocklib_fec_dep])

newsched_blocklib_dtv_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_dtv_lib)
