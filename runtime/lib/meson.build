incdir = [
    '../include',
]


runtime_deps = [yaml_dep, spdlog_dep, threads_dep, fmt_dep, rt_dep, pmtf_dep]

if cuda_dep.found() and get_option('enable_cuda')
  cu_sources = [
    'cudabuffer.cu',
    'cudabuffer_pinned.cu'
  ]

  newsched_runtime_cu_lib = library('newsched-runtime-cu',
      cu_sources, 
      cuda_args: ['-DTAG_IS_SHARED=1', '-DTAG_IS_BUILDING=1'],
      install : true, 
      dependencies : [cuda_dep, pmtf_dep],
      include_directories : incdir)

  newsched_runtime_cu_dep = declare_dependency(include_directories : incdir,
              link_with : newsched_runtime_cu_lib)

  runtime_deps += [newsched_runtime_cu_dep]
endif

runtime_sources = [
  'buffer.cpp',
  'buffer_management.cpp',
  'simplebuffer.cpp',
  'realtime.cpp',
  'thread.cpp',
  'parameter_types.cpp',
  'edge.cpp',
  'graph.cpp',
  'graph_utils.cpp',
  'flat_graph.cpp',
  'flowgraph_monitor.cpp',
  'flowgraph.cpp',
  'logging.cpp',
  'pagesize.cpp',
  'sys_paths.cpp',
  'vmcircbuf.cpp',
  'vmcircbuf_sysv_shm.cpp',
  # mmap requires librt - FIXME - handle this a conditional dependency
  'vmcircbuf_mmap_shm_open.cpp'
]

cpp_args = []
compiler = meson.get_compiler('cpp')
if compiler.has_header('sys/time.h')
  cpp_args += '-DHAVE_SYS_TIME_H'
endif
if compiler.has_header('sys/types.h')
  cpp_args += '-DHAVE_SYS_TYPES_H'
endif
if compiler.has_header('sys/socket.h')
  cpp_args += '-DHAVE_SYS_SOCKET_H'
endif
if compiler.has_header('io.h')
  cpp_args += '-DHAVE_IO_H'
endif
if compiler.has_header('sys/mman.h')
  cpp_args += '-DHAVE_SYS_MMAN_H'
endif
if compiler.has_header('sys/ipc.h')
  cpp_args += '-DHAVE_SYS_IPC_H'
endif
if compiler.has_header('sys/shm.h')
  cpp_args += '-DHAVE_SYS_SHM_H'
endif
if compiler.has_header('signal.h')
  cpp_args += '-DHAVE_SIGNAL_H'
endif
if compiler.has_header('netinet/in.h')
  cpp_args += '-DHAVE_NETINET_IN_H'
endif
if compiler.has_header('arpa/inet.h')
  cpp_args += '-DHAVE_ARPA_INET_H'
endif
if compiler.has_header('unistd.h')
  cpp_args += '-DHAVE_UNISTD_H'
endif
if compiler.has_header('malloc.h')
  cpp_args += '-DHAVE_MALLOC_H'
endif


code = '''#include <signal.h>
    int main(){sigaction(0, 0, 0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SIGACTION')
  cpp_args += '-DHAVE_SIGACTION'
endif

code = '''#include <unistd.h>
    int main(){sysconf(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SYSCONF')
  cpp_args += '-DHAVE_SYSCONF'
endif

code = '''#include <unistd.h>
     int main(){getpagesize(); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_GETPAGESIZE')
  cpp_args += '-DHAVE_GETPAGESIZE'
endif

code = '''#include <Winbase.h>
     int main(){Sleep(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SSLEEP')
  cpp_args += '-DHAVE_SSLEEP'
endif

code = '''#include <stdlib.h>
     int main(){posix_memalign(0, 0, 0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_POSIX_MEMALIGN')
  cpp_args += '-DHAVE_POSIX_MEMALIGN'
endif

code = '''#include <malloc.h>
     int main(){valloc(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_VALLOC')
  cpp_args += '-DHAVE_VALLOC'
endif


# if(APPLE)
#     # Apple's malloc has been 16 byte aligned since at least 10.4
#     ADD_DEFINITIONS(-DALIGNED_MALLOC=16)
# else()
#     ADD_DEFINITIONS(-DALIGNED_MALLOC=0)
# endif()

code = '''#include <signal.h>
     int main(){pthread_sigmask(0, 0, 0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_PTHREAD_SIGMASK', dependencies : [threads_dep])
  cpp_args += '-DHAVE_PTHREAD_SIGMASK'
endif

code = '''#include <windows.h>
    int main(){
        HANDLE handle;
        int size;
        LPCTSTR lpName;
        handle = CreateFileMapping(
                         INVALID_HANDLE_VALUE,    // use paging file
                         NULL,                    // default security
                         PAGE_READWRITE,          // read/write access
                         0,                       // max. object size
                         size,                    // buffer size
                         lpName);                 // name of mapping object
        return 0;
'''
if compiler.compiles(code, name : 'HAVE_CREATEFILEMAPPING', dependencies : [threads_dep])
  cpp_args += '-DHAVE_CREATEFILEMAPPING'
endif

code = '''#include <sys/types.h>
     #include <sys/mman.h>
     int main(){shm_open(0, 0, 0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SHM_OPEN', dependencies : [rt_dep])
  cpp_args += '-DHAVE_SHM_OPEN'
endif

code = '''#define _GNU_SOURCE
     #include <math.h>
     int main(){double x, sin, cos; sincos(x, &sin, &cos); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SINCOS')
  cpp_args += '-DHAVE_SINCOS'
endif

code = '''#define _GNU_SOURCE
     #include <math.h>
     int main(){float x, sin, cos; sincosf(x, &sin, &cos); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SINCOSF')
  cpp_args += '-DHAVE_SINCOSF'
endif

code = '''#include <math.h>
     int main(){sinf(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_SINF')
  cpp_args += '-DHAVE_SINF'
endif

code = '''#include <math.h>
     int main(){cosf(0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_COSF')
  cpp_args += '-DHAVE_COSF'
endif

code = '''#include <sys/mman.h>
    int main(){mmap(0, 0, 0, 0, 0, 0); return 0;}
'''
if compiler.compiles(code, name : 'HAVE_MMAP')
  cpp_args += '-DHAVE_MMAP'
endif

code = '''#include <pthread.h>
          int main(){
            pthread_t pthread;
            pthread_setschedparam(pthread,  0, 0);
            return 0;
        }
'''
if compiler.compiles(code, name : 'HAVE_PTHREAD_SETSCHEDPARAM', dependencies : [threads_dep])
  cpp_args += '-DHAVE_PTHREAD_SETSCHEDPARAM'
endif

code = '''#include <sched.h>
            int main(){
            pid_t pid;
            sched_setscheduler(pid, 0, 0);
            return 0;
        }
'''
if compiler.compiles(code, name : 'HAVE_SCHED_SETSCHEDULER', dependencies : [threads_dep])
  cpp_args += '-DHAVE_SCHED_SETSCHEDULER'
endif

newsched_runtime_lib = library('newsched-runtime', 
    runtime_sources, 
    install : true, 
    dependencies : runtime_deps,
    link_language : 'cpp',
    include_directories : incdir,
    cpp_args : cpp_args)

newsched_runtime_dep = declare_dependency(include_directories : incdir,
					  #  dependencies: runtime_deps, // have to disable this for blocklib_blocks_cuda_cu to link
					   link_with : newsched_runtime_lib)
