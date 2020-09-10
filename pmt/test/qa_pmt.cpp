// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include <pmt/pmt.hpp>
using namespace pmt;


TEST_CASE("Basic PMT properties")
{
    pmt_scalar<int32_t> x(47);

    REQUIRE(x.value()==47);
    REQUIRE((x==47));
    REQUIRE(x.data_type() == pmt_data_type_t::INT32);
    REQUIRE(x.container_type() == pmt_container_type_t::NONE);


    std::vector<float> input_vec{13.,18.,24.};
    pmt_vector<float> pvf(input_vec);
    REQUIRE((pvf.value() == input_vec));
    REQUIRE((pvf == input_vec));
    REQUIRE(pvf.data_type() == pmt_data_type_t::FLOAT);
    REQUIRE(pvf.container_type() == pmt_container_type_t::VECTOR);
}

