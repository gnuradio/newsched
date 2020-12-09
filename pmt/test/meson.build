

test_pmt_sources = ['test_pmt.cpp']

incdir = include_directories('../include')
executable('test_pmt', test_pmt_sources, include_directories : incdir, dependencies: [newsched_pmt_dep], install : false)

# test_pmtf_sources = ['test_pmtf.cpp']
# executable('test_pmtf', test_pmtf_sources, include_directories : incdir, dependencies: [newsched_pmtf_dep], install : false)

test_pmtf_sources = ['test_pmtf.cpp']
executable('test_pmtf', test_pmtf_sources, include_directories : incdir, dependencies: [newsched_pmtf_dep], install : false)

# executable('test_pmt', test_pmt_sources, include_directories : incdir, install : true)



if get_option('enable_testing')
    srcs = ['qa_pmt.cpp']
    e = executable('qa_pmt', 
        srcs, 
        include_directories : incdir, 
        dependencies: [newsched_pmtf_dep,
                    gtest_dep], 
        install : true)
    test('PMT Basic Tests', e)

endif