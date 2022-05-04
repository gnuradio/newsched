#include <gtest/gtest.h>

#include <iostream>

#include <gnuradio/pdu.h>
#include <gnuradio/runtime.h>

using namespace gr;

TEST(PDU, Basics)
{
    gr::pdu<float> pdu1(10);
    pdu1[1] = 7;
    EXPECT_EQ(pdu1[1], 7);
    gr::pdu<float> pdu2({1,2,3,4,5,6,7,8,9,10});

    EXPECT_EQ(pdu2[2], 3);


}

TEST(PDU, PassedAsPmt)
{
    std::vector<float> x{1,2,3,4,5,6,7,8,9,10};
    gr::pdu<float> pdu2(x.data(), x.size());

    pmtf::pmt p = pdu2;

    gr::pdu<float> pdu3 = p;
    EXPECT_EQ(pdu3[2], 3);
}