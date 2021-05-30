fft_deps += [newsched_runtime_dep, volk_dep, fmt_dep, pmtf_dep, fftw_dep]

fft_sources += 'fftw_fft.cc'

if cuda_dep.found() and get_option('enable_cuda')
fft_sources += 'cufft.cc'
fft_deps += cuda_dep
endif

block_cpp_args = ['-DHAVE_CPU']
if cuda_dep.found() and get_option('enable_cuda') and fft_cu_sources != []
    block_cpp_args += '-DHAVE_CUDA'

    newsched_blocklib_fft_cu = library('newsched-blocklib-fft-cu', 
        fft_cu_sources, 
        include_directories : incdir, 
        install : true, 
        dependencies : [cuda_dep])

    newsched_blocklib_fft_cu_dep = declare_dependency(include_directories : incdir,
                        link_with : newsched_blocklib_fft_cu,
                        dependencies : cuda_dep)

    fft_deps += [newsched_blocklib_fft_cu_dep, cuda_dep]

endif

incdir = include_directories(['../include/gnuradio/fft','../include'])
newsched_blocklib_fft_lib = library('newsched-blocklib-fft', 
    fft_sources, 
    include_directories : incdir, 
    install : true,
    link_language: 'cpp',
    dependencies : fft_deps,
    cpp_args : block_cpp_args)

newsched_blocklib_fft_dep = declare_dependency(include_directories : incdir,
					   link_with : newsched_blocklib_fft_lib,
                       dependencies : fft_deps)
