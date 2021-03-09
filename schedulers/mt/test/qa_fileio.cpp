#include <gtest/gtest.h>

#include <iostream>
#include <thread>

#include <gnuradio/flowgraph.hpp>
#include <gnuradio/fileio/file_source.hpp>
#include <gnuradio/fileio/file_sink.hpp>
#include <gnuradio/schedulers/mt/scheduler_mt.hpp>

#include <cstdio>

using namespace gr;

#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>

bool compareFiles(const char * p1, const char * p2) {
  std::ifstream f1(p1, std::ifstream::binary|std::ifstream::ate);
  std::ifstream f2(p2, std::ifstream::binary|std::ifstream::ate);

  if (f1.fail() || f2.fail()) {
    return false; //file problem
  }

  if (f1.tellg() != f2.tellg()) {
    return false; //size mismatch
  }

  //seek back to beginning and use std::equal to compare contents
  f1.seekg(0, std::ifstream::beg);
  f2.seekg(0, std::ifstream::beg);
  return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                    std::istreambuf_iterator<char>(),
                    std::istreambuf_iterator<char>(f2.rdbuf()));
}

TEST(SchedulerMTTest, FileTest)
{
    // Create a tmpfile at tmpnam
    char filename_in[1024];
    auto fn1 = tmpnam(filename_in);
    std::cout << filename_in << std::endl;

    char filename_out[1024];
    auto fn2 = tmpnam(filename_out);
    std::cout << filename_out << std::endl;
    
    auto tmpf = fopen(filename_in, "wb");
    for (int i = 0; i<10; i++)
    {
        // fwrite(&i, 1, 1, tmpf);
        fprintf(tmpf, "%d,", i);
    }

    // std::fputs("Hello, world", tmpf);
    fclose(tmpf);

    auto src = fileio::file_source::make(sizeof(uint8_t), filename_in);
    auto snk = fileio::file_sink::make(sizeof(uint8_t), filename_out);

    flowgraph_sptr fg(new flowgraph());
    fg->connect(src, 0, snk, 0);

    auto sched = schedulers::scheduler_mt::make();
    fg->set_scheduler(sched);
    fg->validate();

    fg->start();
    fg->wait();

    EXPECT_TRUE(compareFiles(filename_in, filename_out));

}