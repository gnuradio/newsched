files = [
    'pmtf.hpp',
    'pmtf_scalar.hpp',
    'pmtf_vector.hpp',
    'pmtf_string.hpp',
    'pmtf_map.hpp',
    'pmtf_map2.hpp',
    'pmtf_wrap.hpp'
]
install_headers(files, subdir : 'pmt')

pmt_gen_h = custom_target('flatc-output',
                        input : ['pmt.fbs'],
                        output : ['pmt_generated.h'],
                        command : [flatc, '-c', '--reflect-types','--gen-mutable','--reflect-names','--cpp-std','c++17', '-o', '@OUTDIR@', '@INPUT@'])

pmt_gen_h_dep = declare_dependency(sources : [pmt_gen_h])
