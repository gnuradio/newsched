incdir = include_directories('../include')

# srcs = ['bm_pmt_serialize_uvec.cpp']
# executable('bm_pmt_serialize_uvec', 
#     srcs, 
#     include_directories : incdir, 
#     dependencies: [newsched_pmtf_dep,
#                    boost_dep], 
#     install : true)

srcs = ['bm_pmt_dict_ref.cpp']
executable('bm_pmt_dict_ref', 
    srcs, 
    include_directories : incdir, 
    dependencies: [pmtf_dep,
                   boost_dep], 
    install : true)

srcs = ['bm_pmt_dict_pack_unpack.cpp']
executable('bm_pmt_dict_pack_unpack', 
    srcs, 
    include_directories : incdir, 
    dependencies: [pmtf_dep,
                   boost_dep], 
    install : true)
