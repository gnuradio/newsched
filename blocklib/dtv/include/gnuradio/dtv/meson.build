headers = [
    'atsc_fpll.hpp',
    'atsc_equalizer.hpp',
    'atsc_fs_checker.hpp',
    'atsc_plinfo.hpp',
    'atsc_sync.hpp',
    'atsc_viterbi_decoder.hpp',
    'atsc_deinterleaver.hpp'
]

install_headers(headers, subdir : 'gnuradio/dtv')

if cuda_dep.found() and get_option('enable_cuda')
    subdir('cuda')
endif
