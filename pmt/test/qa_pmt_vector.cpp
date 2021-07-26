#include <gtest/gtest.h>
#include <complex>

#include <pmt/pmtf.hpp>
#include <pmt/pmtf_vector.hpp>
#include <pmt/pmtf_wrap.hpp>

using namespace pmtf;

using testing_types = ::testing::Types<uint8_t,
                                       int8_t,
                                       uint16_t,
                                       int16_t,
                                       uint32_t,
                                       int32_t,
                                       uint64_t,
                                       int64_t,
                                       float,
                                       double,
                                       std::complex<float>>; //,
                                                             // std::complex<double>>;

// using testing_types = ::testing::Types<float>; //,
//                                                //    std::complex<double>>;


template <typename T>
class PmtVectorFixture : public ::testing::Test
{
public:
    T get_value(int i) { return (T)i; }
    T zero_value() { return (T)0; }
    T nonzero_value() { return (T)17; }
    static const int num_values_ = 100;
};


template <>
std::complex<float> PmtVectorFixture<std::complex<float>>::get_value(int i)
{
    return std::complex<float>(i, -i);
}

template <>
std::complex<double> PmtVectorFixture<std::complex<double>>::get_value(int i)
{
    return std::complex<double>(i, -i);
}

template <>
std::complex<float> PmtVectorFixture<std::complex<float>>::zero_value()
{
    return std::complex<float>(0, 0);
}
template <>
std::complex<double> PmtVectorFixture<std::complex<double>>::zero_value()
{
    return std::complex<double>(0, 0);
}

template <>
std::complex<float> PmtVectorFixture<std::complex<float>>::nonzero_value()
{
    return std::complex<float>(17, -19);
}
template <>
std::complex<double> PmtVectorFixture<std::complex<double>>::nonzero_value()
{
    return std::complex<double>(17, -19);
}

TYPED_TEST_SUITE(PmtVectorFixture, testing_types);

TYPED_TEST(PmtVectorFixture, PmtVectorBasic)
{

    std::vector<TypeParam> vec(this->num_values_);
    for (auto i = 0; i < this->num_values_; i++) {
        vec[i] = this->get_value(i);
    }
    // Init from std::vector
    auto pmt_vec = pmt_vector<TypeParam>(vec);
    EXPECT_EQ(pmt_vec == vec, true);

    // Copy Constructor
    auto a = pmt_vector<TypeParam>(pmt_vec);
    EXPECT_EQ(a == vec, true);
    EXPECT_EQ(a == pmt_vec, true);

    // Assignment operator from std::vector
    a = vec;
    EXPECT_EQ(a == vec, true);
    EXPECT_EQ(a == pmt_vec, true);

    a = pmt_vec;
    EXPECT_EQ(a == vec, true);
    EXPECT_EQ(a == pmt_vec, true);

    // TODO: Add in Move contstructor
}

TYPED_TEST(PmtVectorFixture, RangeBasedLoop)
{

    std::vector<TypeParam> vec(this->num_values_);
    std::vector<TypeParam> vec_doubled(this->num_values_);
    std::vector<TypeParam> vec_squared(this->num_values_);
    for (auto i = 0; i < this->num_values_; i++) {
        vec[i] = this->get_value(i);
        vec_doubled[i] = vec[i] + vec[i];
        vec_squared[i] = vec[i] * vec[i];
    }
    // Init from std::vector
    auto pmt_vec = pmt_vector<TypeParam>(vec);

    for (auto& xx : pmt_vec) {
        xx *= xx;
    }
    EXPECT_EQ(pmt_vec == vec_squared, true);

    pmt_vec = pmt_vector<TypeParam>(vec);
    for (auto& xx : pmt_vec) {
        xx += xx;
    }
    EXPECT_EQ(pmt_vec == vec_doubled, true);
}

TYPED_TEST(PmtVectorFixture, PmtVectorWrap)
{
    // Initialize a PMT Wrap from a std::vector object
    std::vector<TypeParam> vec(this->num_values_);
    for (auto i = 0; i < this->num_values_; i++) {
        vec[i] = this->get_value(i);
    }

    pmt_wrap generic_pmt_obj = std::vector<TypeParam>(vec);
    auto y = get_pmt_vector<TypeParam>(generic_pmt_obj); 
    EXPECT_EQ(y == vec, true);

    // Try to cast as a scalar type
    EXPECT_THROW(get_pmt_scalar<int8_t>(generic_pmt_obj), std::runtime_error);
}


TYPED_TEST(PmtVectorFixture, VectorWrites)
{
    // Initialize a PMT Wrap from a std::vector object
    std::vector<TypeParam> vec(this->num_values_);
    std::vector<TypeParam> vec_modified(this->num_values_);
    for (auto i = 0; i < this->num_values_; i++) {
        vec[i] = this->get_value(i);
        vec_modified[i] = vec[i];
        if (i%7 == 2) {
            vec_modified[i] = vec[i] + this->get_value(i);
        }
    }

    auto pmt_vec = pmt_vector(vec);
    for (auto i = 0; i < this->num_values_; i++) {
        if (i%7 == 2) {
            pmt_vec[i] = pmt_vec[i] + this->get_value(i);
        }
    }
    EXPECT_EQ(pmt_vec, vec_modified);

}

TYPED_TEST(PmtVectorFixture, OtherConstructors) {

    // Check the other constructors
    pmt_vector<TypeParam> vec1(4);
    EXPECT_EQ(vec1.size(), 4);
    for (auto& e: vec1)
        EXPECT_EQ(e, this->zero_value());

    pmt_vector<TypeParam> vec2(4, this->nonzero_value());
    for (auto& e: vec2)
        EXPECT_EQ(e, this->nonzero_value());


    std::vector<TypeParam> data(this->num_values_);
    for (auto i = 0; i < this->num_values_; i++) {
        data[i] = this->get_value(i);
    }

    pmt_vector<TypeParam> vec3(data.begin(), data.end());
    EXPECT_EQ(vec3.size(), data.size());
    size_t i = 0;
    for (auto& e: vec3)
        EXPECT_EQ(e, data[i++]);

    pmt_vector<TypeParam> vec4(vec3);
    EXPECT_EQ(vec3.ptr(), vec4.ptr());
}

