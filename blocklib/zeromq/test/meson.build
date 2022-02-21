###################################################
#    QA
###################################################

if get_option('enable_testing')
    test('qa_zeromq_pushpull', py3, args : files('qa_zeromq_pushpull.py'), env: TEST_ENV)
    test('qa_zeromq_pubsub', py3, args : files('qa_zeromq_pubsub.py'), env: TEST_ENV)
endif
