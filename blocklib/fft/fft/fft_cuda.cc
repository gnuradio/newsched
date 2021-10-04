#include "fft_cuda.hh"


extern void exec_fft_shift(const cuFloatComplex* in,
                    cuFloatComplex* out,
                    int n,
                    int grid_size,
                    int block_size,
                    cudaStream_t stream);

namespace gr {
namespace fft {

template <class T, bool forward>
typename fft<T, forward>::sptr fft<T, forward>::make_cuda(const block_args& args)
{
    return std::make_shared<fft_cuda<T, forward>>(args);
}

template <class T, bool forward>
fft_cuda<T, forward>::fft_cuda(const typename fft<T, forward>::block_args& args) 
    : sync_block("fft_cuda"), fft<T, forward>(args),
      d_fft_size(args.fft_size),
      d_shift(args.shift),
      d_fft(args.fft_size, 64)
{
    if (args.window.empty() || args.window.size() == d_fft_size) {
        d_window = args.window;
    } else {
        throw std::runtime_error("fft: window not the same length as fft_size");
    }

    
    cudaStreamCreate(&d_stream);
    // this->set_output_multiple(64);

}


template <>
void fft_cuda<gr_complex, true>::fft_and_shift(const gr_complex* in, gr_complex* out, int batch)
{
    int blockSize = 1024;
    int gridSize = (batch*d_fft_size + blockSize - 1) / blockSize;
    if (d_shift)
        exec_fft_shift((cuFloatComplex *)in, (cuFloatComplex *)in, batch*d_fft_size, gridSize, blockSize, d_stream);
    d_fft.execute(in, out);
}

template <>
void fft_cuda<gr_complex, false>::fft_and_shift(const gr_complex* in, gr_complex* out, int batch)
{
    int blockSize = 1024;
    int gridSize = (batch*d_fft_size + blockSize - 1) / blockSize;
    if (d_shift)
        exec_fft_shift((cuFloatComplex *)in, (cuFloatComplex *)in, batch*d_fft_size, gridSize, blockSize, d_stream);
    d_fft.execute(in, out);
}

template <>
void fft_cuda<float, true>::fft_and_shift(const float* in, gr_complex* out, int batch)
{
    
}

template <>
void fft_cuda<float, false>::fft_and_shift(const float* in, gr_complex* out, int batch)
{

}

template <class T, bool forward>
work_return_code_t fft_cuda<T, forward>::work(std::vector<block_work_input>& work_input,
                                             std::vector<block_work_output>& work_output)
{
    // auto* iptr = (uint8_t*)work_input[0].items();
    // int size = work_output[0].n_items * d_itemsize;
    // auto* optr = (uint8_t*)work_output[0].items();
    // // std::fft(iptr, iptr + size, optr);
    // memcpy(optr, iptr, size);

    auto in = static_cast<const T*>(work_input[0].items());
    auto out = static_cast<gr_complex*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;

    int count = 0;

    while (count < noutput_items) {

        fft_and_shift(in, out, 64);
        // cudaDeviceSynchronize();

        // T host_in[d_fft_size];
        // T host_out[d_fft_size];

        // cudaMemcpy(host_in, in, d_fft_size*sizeof(T), cudaMemcpyDeviceToHost);
        // cudaMemcpy(host_out, out, d_fft_size*sizeof(T), cudaMemcpyDeviceToHost);


        in += d_fft_size * 64;
        out += d_fft_size * 64;
        count += 64;
    }

    cudaStreamSynchronize(d_stream);
    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

template class fft<gr_complex, true>;
template class fft<gr_complex, false>;
template class fft<float, true>;
template class fft<float, false>;


} // namespace fft
} // namespace gr