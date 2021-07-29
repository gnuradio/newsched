#pragma once

#include <gnuradio/dtv/atsc_deinterleaver.hh>

#include "interleaver_fifo.hh"

namespace gr {
namespace dtv {

class atsc_deinterleaver_cpu : public atsc_deinterleaver
{
public:
    atsc_deinterleaver_cpu(const block_args& args);
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

    static constexpr int s_interleavers = 52;

    //! transform a single symbol
    unsigned char transform(unsigned char input)
    {
        unsigned char retval = m_fifo[m_commutator].stuff(input);
        m_commutator++;
        if (m_commutator >= s_interleavers)
            m_commutator = 0;
        return retval;
    }

    /*!
     * Note: The use of the alignment_fifo keeps the encoder and decoder
     * aligned if both are synced to a field boundary.  There may be other
     * ways to implement this function.  This is a best guess as to how
     * this should behave, as we have no test vectors for either the
     * interleaver or deinterleaver.
     */
    interleaver_fifo<unsigned char> alignment_fifo;

    int m_commutator;
    std::vector<interleaver_fifo<unsigned char>> m_fifo;

    //! reset interleaver (flushes contents and resets commutator)
    void reset();

    //! sync interleaver (resets commutator, but doesn't flush fifos)
    void sync() { m_commutator = 0; }
};

} // namespace dtv
} // namespace gr