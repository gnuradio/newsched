#include <gtest/gtest.h>
#include <complex>

#include <pmt/pmtf.hpp>
#include <pmt/pmtf_map.hpp>
#include <pmt/pmtf_scalar.hpp>
#include <pmt/pmtf_string.hpp>
#include <pmt/pmtf_vector.hpp>
#include <pmt/pmtf_wrap.hpp>

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

    std::vector<std::complex<float>> cf_vec_val{ { 0, 1 }, { 2, 3 }, { 4, 5 } };
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
    auto vv1 =
        std::static_pointer_cast<pmt_scalar<std::complex<float>>>(map_pmt->ref("key1"));
    EXPECT_EQ(vv1->value(), val1);

    auto vv2 = std::static_pointer_cast<pmt_vector<int32_t>>(map_pmt->ref("key2"));
    EXPECT_EQ(vv2->value(), val2);

    // Pull the map back out of the PMT
    auto newmap = map_pmt->value();
    EXPECT_EQ(std::static_pointer_cast<pmt_scalar<std::complex<float>>>(newmap["key1"])
                  ->value(),
              val1);
    EXPECT_EQ(std::static_pointer_cast<pmt_vector<int32_t>>(newmap["key2"])->value(),
              val2);
}

TEST(Pmt, VectorWrites)
{
    {
        std::vector<std::complex<float>> cf_vec_val{ { 0, 1 }, { 2, 3 }, { 4, 5 } };
        std::vector<std::complex<float>> cf_vec_val_modified{ { 4, 5 },
                                                              { 6, 7 },
                                                              { 8, 9 } };
        auto cf_pmt_vec = pmt_vector<std::complex<float>>::make(cf_vec_val);
        EXPECT_EQ(cf_pmt_vec->value(), cf_vec_val);
        EXPECT_EQ(cf_pmt_vec->data_type(), Data::VectorComplex64);

        auto writable_vec = cf_pmt_vec->writable_elements();
        writable_vec[0] = { 4, 5 };
        writable_vec[1] = { 6, 7 };
        writable_vec[2] = { 8, 9 };

        EXPECT_EQ(cf_pmt_vec->value(), cf_vec_val_modified);
    }
    {
        std::vector<uint32_t> int_vec_val{ 1, 2, 3, 4, 5 };
        std::vector<uint32_t> int_vec_val_modified{ 6, 7, 8, 9, 10 };
        auto int_pmt_vec = pmt_vector<uint32_t>::make(int_vec_val);
        EXPECT_EQ(int_pmt_vec->value(), int_vec_val);
        EXPECT_EQ(int_pmt_vec->data_type(), Data::VectorUInt32);

        auto writable_vec = int_pmt_vec->writable_elements();
        writable_vec[0] = 6;
        writable_vec[1] = 7;
        writable_vec[2] = 8;
        writable_vec[3] = 9;
        writable_vec[4] = 10;

        EXPECT_EQ(int_pmt_vec->value(), int_vec_val_modified);
    }
}

TEST(Pmt, VectorWrapper) {
    pmt_vector_wrapper<uint32_t> x(10);
    pmt_vector_wrapper<uint32_t> y{1,2,3,4,6,7};
    std::vector<uint32_t> data{1,2,3,4,6,7};
    for (size_t i = 0; i < y.size(); i++) {
        EXPECT_EQ(y[i], data[i]);
    }
    // Make sure that range based for loop works.
    size_t i = 0;
    for (auto& e : y) {
        EXPECT_EQ(e, data[i++]);
    }

    // Make sure I can mutate the data
    for (auto& e: y) {
        e += 2;
    }
    i = 0;
    for (auto& e: y) {
        EXPECT_EQ(e, data[i++]+2);
    }

    // Create from an std::vector
    pmt_vector_wrapper<uint32_t> x_vec(std::vector<uint32_t>{1,2,3,4,6,7});

    // Check the other constructors
    pmt_vector_wrapper<uint32_t> vec1(4);
    EXPECT_EQ(vec1.size(), 4);
    for (auto& e: vec1)
        EXPECT_EQ(e, 0);

    pmt_vector_wrapper<uint32_t> vec2(4, 2);
    for (auto& e: vec2)
        EXPECT_EQ(e, 2);

    pmt_vector_wrapper<uint32_t> vec3(data.begin(), data.end());
    EXPECT_EQ(vec3.size(), data.size());
    i = 0;
    for (auto& e: vec3)
        EXPECT_EQ(e, data[i++]);

    pmt_vector_wrapper<uint32_t> vec4(vec3);
    EXPECT_EQ(vec3.ptr(), vec4.ptr());
}

TEST(Pmt, MapWrapper) {
    pmt_map_wrapper<std::string> x();

}

TEST(Pmt, PmtWrap) {
    pmt_wrap x(4);
    pmt_wrap y(std::vector({1,2,3,4}));
}
