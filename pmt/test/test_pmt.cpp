#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <pmt/pmt.hpp>

using namespace pmt;
using namespace std;

int main(int argc, char* argv[])
{
    auto int_pmt = pmt_scalar<int>(17);
    std::cout << (int_pmt == 17) << std::endl;
    std::cout << (int_pmt != 17) << std::endl;
    auto int_vec_pmt = pmt_vector<int>({ 4, 5, 6 });
    auto int_vec_pmt2 = pmt_vector<int>({ 7, 8, 9 });
    auto vec = int_vec_pmt2.value();

    for (auto d : vec) {
        std::cout << d << ",";
    }
    std::cout << std::endl;
}
