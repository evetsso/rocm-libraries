#include "../../shared/arithmetic.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_complex.h"
#include "../../shared/rocfft_params.h"
#include <fstream>
#include <string>

typedef rocfft_complex<float> Tdata;

std::string read_file(const char* filename)
{
    std::string   src;
    std::ifstream infile(filename);

    std::string line;
    while(std::getline(infile, line))
    {
        if(line.starts_with("#include"))
            continue;
        src += line;
        src += "\n";
    }
    return src;
}

std::string fwd_load_callback_src(size_t length, size_t lengthPadded)
{
    std::string lengthStr       = std::to_string(length);
    std::string lengthPaddedStr = std::to_string(lengthPadded);
    std::string src             = read_file("../shared/rocfft_complex.h");
    src += "typedef rocfft_complex<float> Tdata;\n";
    src += "extern \"C\"\n";
    src += "__device__ Tdata fwd_load_callback(Tdata* input, size_t offset, void* cbdata, void* "
           "sharedMem)\n";
    src += "{\n";
    src += "  auto batch = offset / " + lengthPaddedStr + ";\n";
    src += "  auto idx = offset % " + lengthPaddedStr + ";\n";
    src += "  if(idx < " + lengthStr + ")\n";
    src += "    return input[batch * " + lengthStr + " + idx];\n";
    src += "  else\n";
    src += "    return Tdata{0.0,0.0};\n";
    src += "}\n";
    return src;
}

std::string back_load_callback_src(size_t length, size_t lengthPadded)
{
    std::string lengthStr       = std::to_string(length);
    std::string lengthPaddedStr = std::to_string(lengthPadded);
    std::string src             = read_file("../shared/rocfft_complex.h");
    src += "typedef rocfft_complex<float> Tdata;\n";
    src += "extern \"C\"\n";
    src += "__device__ Tdata back_load_callback(Tdata* input, size_t offset, void* cbdata, void* "
           "sharedMem)\n";
    src += "{\n";
    src += "  auto idx = offset % " + lengthPaddedStr + ";\n";
    src += "  auto G = static_cast<Tdata*>(cbdata);\n";
    src += "  return input[offset] * G[idx];\n";
    src += "}\n";
    return src;
}

std::string back_store_callback_src(size_t length, size_t lengthPadded)
{
    std::string lengthStr       = std::to_string(length);
    std::string lengthPaddedStr = std::to_string(lengthPadded);
    std::string src             = read_file("../shared/rocfft_complex.h");
    src += "typedef rocfft_complex<float> Tdata;\n";
    src += "extern \"C\"\n";
    src += "__device__ void back_store_callback(Tdata* output, size_t offset, Tdata elem, void* "
           "cbdata, void* "
           "sharedMem)\n";
    src += "{\n";
    src += "  auto batch = offset / " + lengthPaddedStr + ";\n";
    src += "  auto idx = offset % " + lengthPaddedStr + ";\n";
    src += "  if(idx < " + lengthStr + ")\n";
    src += "    output[batch * " + lengthStr + " + idx] = elem;\n";
    src += "}\n";
    return src;
}

__global__ void pad_kernel(Tdata* __restrict__ F,
                           Tdata* __restrict__ padded_F,
                           size_t length,
                           size_t lengthPadded)
{
    auto offset_padded_F = blockIdx.x * 32 + threadIdx.x;
    auto batch           = offset_padded_F / lengthPadded;
    auto F_idx           = offset_padded_F % lengthPadded;

    if(F_idx < length)
        padded_F[offset_padded_F] = F[batch * length + F_idx];
    else
        padded_F[offset_padded_F] = Tdata{0.0, 0.0};
}

__global__ void
    hadamard_product_kernel(Tdata* __restrict__ F, Tdata* __restrict__ G, size_t lengthPadded)
{
    auto offset_F = blockIdx.x * 32 + threadIdx.x;
    auto offset_G = offset_F % lengthPadded;
    F[offset_F]   = F[offset_F] * G[offset_G];
}

std::vector<char> compile_jit_callback(const std::string& src)
{
    struct RaiiState
    {
        hiprtcProgram prog = nullptr;
        ~RaiiState()
        {
            if(prog)
            {
                hiprtcDestroyProgram(&prog);
            }
        }
    };
    RaiiState state;

    auto err
        = hiprtcCreateProgram(&state.prog, src.c_str(), "rocfft_callback.hip", 0, nullptr, nullptr);
    if(err != HIPRTC_SUCCESS)
    {
        throw hiprtc_runtime_error{"unable to create program", err};
    }

    std::vector<const char*> options;
#ifdef __HIP_PLATFORM_AMD__
    options.push_back("-O3");
    options.push_back("--offload-arch=amdgcnspirv");
#else
#ifdef HIPFFT_CUDA_INCLUDE
    options.push_back("-I" HIPFFT_CUDA_INCLUDE);
#endif
    options.push_back("-dlto");
    options.push_back("--relocatable-device-code=true");
#endif

    err = hiprtcCompileProgram(state.prog, options.size(), options.data());
    if(err != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(state.prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(state.prog, log.data()) == HIPRTC_SUCCESS)
                throw hiprtc_runtime_error{std::string(log.begin(), log.end()), err};
        }
        throw hiprtc_runtime_error{"compile failed without log", err};
    }

    size_t            codeSize;
    std::vector<char> code;
#ifdef __HIP_PLATFORM_AMD__
    err = hiprtcGetBitcodeSize(state.prog, &codeSize);
    if(err != HIPRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode size", err};

    code.resize(codeSize);
    err = hiprtcGetBitcode(state.prog, code.data());
    if(err != HIPRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode", err};
#else
    auto nverr = nvrtcGetLTOIRSize(state.prog, &codeSize);
    if(nverr != NVRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode size", nvrtcResultTohiprtcResult(nverr)};

    code.resize(codeSize);
    nverr = nvrtcGetLTOIR(state.prog, code.data());
    if(nverr != NVRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode", nvrtcResultTohiprtcResult(nverr)};
#endif
    return code;
}

struct convolution
{
    convolution(gpubuf_t<Tdata>& F, gpubuf_t<Tdata>& G, size_t batch)
        : F(F)
        , G(G)
        , length(F.size() / sizeof(Tdata) / batch)
        , lengthPadded(1 << CeilPo2(length + length - 1))
        , batch(batch)
    {
        // For the sake of this example, ensure that a single batch of
        // F and G are the same size
        if(F.size() / batch != G.size())
            throw std::runtime_error("F size doesn't match G size");

        params_fwd_F.length        = {lengthPadded};
        params_fwd_F.nbatch        = batch;
        params_fwd_G.length        = {lengthPadded};
        params_fwd_G.nbatch        = 1;
        params_back.transform_type = fft_transform_type_complex_inverse;
        params_back.length         = {lengthPadded};
        params_back.nbatch         = batch;

        params_back.scale_factor = 1.0 / lengthPadded;

        params_fwd_F.validate();
        params_fwd_G.validate();
        params_back.validate();
        if(!params_fwd_F.valid() || !params_fwd_G.valid() || !params_back.valid())
            throw std::runtime_error("invalid params");

        if(padded_F.alloc(lengthPadded * sizeof(Tdata) * batch) != hipSuccess
           || padded_G.alloc(lengthPadded * sizeof(Tdata)) != hipSuccess)
            throw std::runtime_error("failed to alloc padded");
    }

    virtual void execute() = 0;

    gpubuf_t<Tdata>& F;
    gpubuf_t<Tdata>& G;

    gpubuf_t<Tdata> padded_F;
    gpubuf_t<Tdata> padded_G;

    rocfft_params params_fwd_F;
    rocfft_params params_fwd_G;
    rocfft_params params_back;

    size_t length;
    size_t lengthPadded;
    size_t batch;
};

struct convolution_kernel : public convolution
{
    convolution_kernel(gpubuf_t<Tdata>& F, gpubuf_t<Tdata>& G, size_t batch)
        : convolution(F, G, batch)
    {
        params_fwd_F.setup_structs();
        params_fwd_G.setup_structs();
        params_back.setup_structs();
    }

    void execute() override
    {
        // pad F and G

        // {
        //     dim3 blockDim{32, 1, 1};
        //     dim3 gridDim{static_cast<unsigned int>(lengthPadded * batch / 32), 1, 1};
        //     pad_kernel<<<gridDim, blockDim>>>(F.data(), padded_F.data(), length, lengthPadded);
        // }
        // {
        //     dim3 blockDim{32, 1, 1};
        //     dim3 gridDim{static_cast<unsigned int>(lengthPadded / 32), 1, 1};
        //     pad_kernel<<<gridDim, blockDim>>>(G.data(), padded_G.data(), length, lengthPadded);
        // }
        if(hipMemset(padded_F.data(), 0, padded_F.size()) != hipSuccess
           || hipMemset(padded_G.data(), 0, padded_G.size()) != hipSuccess)
            throw std::runtime_error("failed to memset");

        if(hipMemcpy(padded_F.data(), F.data(), F.size(), hipMemcpyDeviceToDevice) != hipSuccess
           || hipMemcpy(padded_G.data(), G.data(), G.size(), hipMemcpyDeviceToDevice) != hipSuccess)
            throw std::runtime_error("failed to memcpy");

        // padded forward transforms
        std::vector<void*> ptrs(1);
        ptrs[0] = padded_F.data();
        params_fwd_F.execute(ptrs.data(), ptrs.data());
        ptrs[0] = padded_G.data();
        params_fwd_G.execute(ptrs.data(), ptrs.data());

        // hadamard product
        {
            dim3 blockDim{32, 1, 1};
            dim3 gridDim{static_cast<unsigned int>(lengthPadded * batch / 32), 1, 1};
            hadamard_product_kernel<<<gridDim, blockDim>>>(
                padded_F.data(), padded_G.data(), lengthPadded);
        }

        // inverse transform
        ptrs[0] = padded_F.data();
        params_back.execute(ptrs.data(), ptrs.data());
    }
};

struct convolution_jit : public convolution
{
    convolution_jit(gpubuf_t<Tdata>& F, gpubuf_t<Tdata>& G, size_t batch)
        : convolution(F, G, batch)
    {
        auto fwd_load_callback = compile_jit_callback(fwd_load_callback_src(length, lengthPadded));
        auto back_load_callback
            = compile_jit_callback(back_load_callback_src(length, lengthPadded));
        auto back_store_callback
            = compile_jit_callback(back_store_callback_src(length, lengthPadded));

        // add load callback to forward plans to remove the need for
        // padding the input, though the transforms now need to be
        // out-of-place to materialize the padded results in memory
        params_fwd_F.load_jit_cb_state         = std::make_shared<fft_params::jit_cb_state_t>();
        params_fwd_F.load_jit_cb_state->symbol = "fwd_load_callback";
        params_fwd_F.load_jit_cb_state->func   = fwd_load_callback;
        params_fwd_F.placement                 = fft_placement_notinplace;
        params_fwd_F.run_callbacks             = fft_callback_type_jit;
        params_fwd_F.setup_structs();
        params_fwd_G.load_jit_cb_state         = std::make_shared<fft_params::jit_cb_state_t>();
        params_fwd_G.load_jit_cb_state->symbol = "fwd_load_callback";
        params_fwd_G.load_jit_cb_state->func   = fwd_load_callback;
        params_fwd_G.placement                 = fft_placement_notinplace;
        params_fwd_G.run_callbacks             = fft_callback_type_jit;
        params_fwd_G.setup_structs();

        // add load callback to backward plan to compute hadamard
        // product without needing to materialize it in memory
        params_back.load_jit_cb_state         = std::make_shared<fft_params::jit_cb_state_t>();
        params_back.load_jit_cb_state->symbol = "back_load_callback";
        params_back.load_jit_cb_state->func   = back_load_callback;
        // pass G pointer to the callback so it can do the computation
        params_back.load_jit_cb_state->data.resize(1);
        params_back.load_jit_cb_state->data[0]
            = gpubuf::make_nonowned(padded_G.data(), padded_G.size());
        // store callback allows us to only bother writing 'length'
        // elements since the padded results beyond that are garbage anyway
        params_back.store_jit_cb_state         = std::make_shared<fft_params::jit_cb_state_t>();
        params_back.store_jit_cb_state->symbol = "back_store_callback";
        params_back.store_jit_cb_state->func   = back_store_callback;
        params_back.run_callbacks              = fft_callback_type_jit;

        params_back.setup_structs();
    }
    void execute() override
    {
        std::vector<void*> ptrs_in(1);
        std::vector<void*> ptrs_out(1);
        ptrs_in[0]  = F.data();
        ptrs_out[0] = padded_F.data();
        params_fwd_F.execute(ptrs_in.data(), ptrs_out.data());
        ptrs_in[0]  = G.data();
        ptrs_out[0] = padded_G.data();
        params_fwd_G.execute(ptrs_in.data(), ptrs_out.data());

        // inverse transform
        std::vector<void*> ptrs(1);
        ptrs[0] = padded_F.data();
        params_back.execute(ptrs.data(), ptrs.data());
    }
};

void run_trial(convolution_jit&    conv_jit,
               std::vector<float>& samples_jit,

               convolution_kernel& conv_kernel,
               std::vector<float>& samples_kernel,
               hipEvent_wrapper_t& start,
               hipEvent_wrapper_t& stop,
               bool                run_jit)
{
    auto& conv
        = run_jit ? static_cast<convolution&>(conv_jit) : static_cast<convolution&>(conv_kernel);
    auto& samples = run_jit ? samples_jit : samples_kernel;

    if(hipEventRecord(start) != hipSuccess)
        throw std::runtime_error("failed to record start");

    conv.execute();

    if(hipEventRecord(stop) != hipSuccess)
        throw std::runtime_error("failed to record start");

    if(hipEventSynchronize(stop) != hipSuccess)
        throw std::runtime_error("failed to sync");
    float elapsed = 0.0;
    if(hipEventElapsedTime(&elapsed, start, stop) != hipSuccess)
        throw std::runtime_error("hipEventElapsedTime failed");

    samples.push_back(elapsed);
}

void run_testcase(size_t length, size_t batch)
{
    gpubuf_t<Tdata> F, G;
    if(F.alloc(length * sizeof(Tdata) * batch) != hipSuccess
       || G.alloc(length * sizeof(Tdata)) != hipSuccess)
        throw std::runtime_error("alloc failed");

    rocfft_params params_F, params_G;
    params_F.length = {length};
    params_F.nbatch = batch;
    params_G.length = {length};
    params_G.nbatch = 1;
    params_F.validate();
    params_G.validate();
    if(!params_F.valid() || !params_G.valid())
        throw std::runtime_error("invalid params");

    std::vector<gpubuf> input_F(1);
    input_F.front() = gpubuf::make_nonowned(F.data(), F.size());
    std::vector<gpubuf> input_G(1);
    input_G.front() = gpubuf::make_nonowned(G.data(), G.size());

    params_F.compute_input(input_F);
    params_G.compute_input(input_G);

    convolution_jit    conv_jit{F, G, batch};
    convolution_kernel conv_kernel{F, G, batch};

    std::vector<float> samples_jit;
    std::vector<float> samples_kernel;

    hipEvent_wrapper_t start;
    hipEvent_wrapper_t stop;
    start.alloc();
    stop.alloc();

    const size_t       NTRIALS = 10;
    std::random_device randdev;
    while(samples_kernel.size() < NTRIALS || samples_jit.size() < NTRIALS)
    {
        bool run_jit = randdev() % 2;

        run_trial(conv_jit, samples_jit, conv_kernel, samples_kernel, start, stop, run_jit);
    }

    // compute medians
    auto get_median = [](const std::vector<float>& samples) {
        if(samples.size() % 2)
        {
            return static_cast<double>(samples[samples.size() / 2]);
        }
        else
        {
            return (static_cast<double>(samples[samples.size() / 2])
                    + static_cast<double>(samples[samples.size() / 2 + 1]))
                   / 2;
        }
    };

    printf("length %zu batch %zu\n ", length, batch);

    auto median_kernel = get_median(samples_kernel);
    auto median_jit    = get_median(samples_jit);
    printf("  median kernel: %f (%zu samples)\n  median jit: %f (%zu samples)\n  speedup: %.2f\n",
           median_kernel,
           samples_kernel.size(),
           median_jit,
           samples_jit.size(),
           median_kernel / median_jit);
}

int main()
{
    rocfft_params params;
    params.setup();

    run_testcase(32768, 100);

    params.cleanup();
}
