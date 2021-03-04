#include <gtest/gtest.h>
#include <complex>

#include <pmt/pmtf.hpp>
#include <pmt/pmtf_map.hpp>
#include <pmt/pmtf_scalar.hpp>
#include <pmt/pmtf_string.hpp>
#include <pmt/pmtf_vector.hpp>

using namespace pmtf;

TEST(Pmt, BasicPmtTests)
{
    std::complex<float> cplx_val = std::complex<float>(1.2, -3.4);
    auto x = pmt_scalar<std::complex<float>>::make(cplx_val);

    EXPECT_EQ(x->value(), cplx_val);
    EXPECT_EQ(x->data_type(), Data::ScalarComplex64);

    std::vector<int32_t> int_vec_val{ 5, 9, 23445, 63, -25 };
    auto int_pmt_vec = pmt_vector<int32_t>::make(int_vec_val);
    EXPECT_EQ(int_pmt_vec->value(), int_vec_val);
    EXPECT_EQ(int_pmt_vec->data_type(), Data::VectorInt32);

    std::vector<std::complex<float>> cf_vec_val{ {0,1},{2,3},{4,5} };
    auto cf_pmt_vec = pmt_vector<std::complex<float>>::make(cf_vec_val);
    EXPECT_EQ(cf_pmt_vec->value(), cf_vec_val);
    EXPECT_EQ(cf_pmt_vec->data_type(), Data::VectorComplex64);
}

TEST(Pmt, PmtStringTests)
{
    auto str_pmt = pmt_string::make("hello");

    EXPECT_EQ(str_pmt->value(), "hello");
}


TEST(Pmt, PmtMapTests)
{
    std::complex<float> val1(1.2, -3.4);
    std::vector<int32_t> val2{ 44, 34563, -255729, 4402 };

    // Create the PMT map
    std::map<std::string, pmt_sptr> input_map({
        { "key1", pmt_scalar<std::complex<float>>::make(val1) },
        { "key2", pmt_vector<int32_t>::make(val2) },
    });
    auto map_pmt = pmt_map<std::string>::make(input_map);

    // Lookup values in the PMT map and compare with what was put in there
    auto vv1 = std::static_pointer_cast<pmt_scalar<std::complex<float>>>(map_pmt->ref("key1"));
    EXPECT_EQ(vv1->value(), val1);

    auto vv2 = std::static_pointer_cast<pmt_vector<int32_t>>(map_pmt->ref("key2"));
    EXPECT_EQ(vv2->value(), val2);
   
    // Pull the map back out of the PMT
    auto newmap = map_pmt->value();
    EXPECT_EQ(std::static_pointer_cast<pmt_scalar<std::complex<float>>>(newmap["key1"])->value(), val1);
    EXPECT_EQ(std::static_pointer_cast<pmt_vector<int32_t>>(newmap["key2"])->value(), val2);

}
