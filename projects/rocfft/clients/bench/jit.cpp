#include "../../shared/arithmetic.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"

const char* callback_src{
    R"_CALLBACK_SRC_(
extern "C"
__device__ float2 load_callback(float2* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto elem   = input[offset];
    elem.x *= 2;
    elem.y *= 2;
    return elem;
}

extern "C"
__device__ void store_callback(float2* output, size_t offset, float2 elem, void* cbdata, void* sharedMem)
{
    elem.x /= 2;
    elem.y /= 2;
    output[offset] = elem;
}
)_CALLBACK_SRC_"};

__device__ size_t compute_offset(size_t dim, const size_t* lengths)
{
    size_t       offset = 0;
    size_t       stride = 1;
    unsigned int idx    = blockIdx.x * 32 + threadIdx.x;
    for(size_t i = 0; i < dim; ++i)
    {
        offset += idx % lengths[i] * stride;
        idx = idx / lengths[i];
        stride *= lengths[i];
    }
    offset += blockIdx.z * stride;
    return offset;
}

__global__ void
    load_callback_kernel(float2* __restrict__ input, size_t dim, const size_t* __restrict__ lengths)
{
    auto offset = compute_offset(dim, lengths);
    auto elem   = input[offset];
    elem.x *= 2;
    elem.y *= 2;
    input[offset] = elem;
}

__global__ void
    store_callback_kernel(float2* output, size_t dim, const size_t* __restrict__ lengths)
{
    auto offset = compute_offset(dim, lengths);
    auto elem   = output[offset];
    elem.x /= 2;
    elem.y /= 2;
    output[offset] = elem;
}

template <typename Tkernel>
void apply_callback(Tkernel                 kernel,
                    float2*                 ptr,
                    const gpubuf_t<size_t>& lengths_device,
                    rocfft_params&          params)
{
    if(params.length.front() % 32)
        throw std::runtime_error("X dim needs to be divisible by 32");

    dim3 gridDim{
        static_cast<unsigned int>(params.length.front() / 32
                                  * product(params.length.begin() + 1, params.length.end())),
        1U,
        static_cast<unsigned int>(params.nbatch)};
    dim3 blockDim{32U, 1U, 1U};

    kernel<<<gridDim, blockDim>>>(ptr, params.length.size(), lengths_device.data());
}

void run_trial(rocfft_params&          params_kernel,
               std::vector<float>&     samples_kernel,
               rocfft_params&          params_jit,
               std::vector<float>&     samples_jit,
               hipEvent_wrapper_t&     start,
               hipEvent_wrapper_t&     stop,
               const gpubuf&           data_orig,
               const gpubuf_t<size_t>& lengths_device,
               bool                    run_jit)
{
    auto& params  = run_jit ? params_jit : params_kernel;
    auto& samples = run_jit ? samples_jit : samples_kernel;

    gpubuf data;
    if(data.alloc(data_orig.size()) != hipSuccess)
        throw std::runtime_error("failed to alloc per-trial data");
    if(hipMemcpy(data.data(), data_orig.data(), data_orig.size(), hipMemcpyDeviceToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to memcpy input");

    std::vector<void*> ptrs(1);
    ptrs[0] = data.data();

    if(hipEventRecord(start) != hipSuccess)
        throw std::runtime_error("failed to record start");

    if(!run_jit)
    {
        // launch kernel to apply load callback
        apply_callback(
            load_callback_kernel, static_cast<float2*>(data.data()), lengths_device, params);
    }

    if(params.execute(ptrs.data(), ptrs.data()) != fft_status_success)
        throw std::runtime_error("execute failed");

    if(!run_jit)
    {
        // launch kernel to apply store callback
        apply_callback(
            store_callback_kernel, static_cast<float2*>(data.data()), lengths_device, params);
    }

    if(hipEventRecord(stop) != hipSuccess)
        throw std::runtime_error("failed to record start");

    if(hipEventSynchronize(stop) != hipSuccess)
        throw std::runtime_error("failed to sync");
    float elapsed = 0.0;
    if(hipEventElapsedTime(&elapsed, start, stop) != hipSuccess)
        throw std::runtime_error("hipEventElapsedTime failed");

    samples.push_back(elapsed);
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

void jitify(rocfft_params& params_jit)
{
    auto callback_bitcode                 = compile_jit_callback(callback_src);
    params_jit.load_jit_cb_state          = std::make_shared<fft_params::jit_cb_state_t>();
    params_jit.load_jit_cb_state->symbol  = "load_callback";
    params_jit.load_jit_cb_state->func    = callback_bitcode;
    params_jit.store_jit_cb_state         = std::make_shared<fft_params::jit_cb_state_t>();
    params_jit.store_jit_cb_state->symbol = "store_callback";
    params_jit.store_jit_cb_state->func   = callback_bitcode;
}

void run_testcase(const std::vector<size_t>& length, size_t batch)
{
    rocfft_params params_jit;
    params_jit.length = length;
    params_jit.nbatch = batch;
    jitify(params_jit);
    params_jit.validate();

    if(!params_jit.valid())
        throw std::runtime_error("invalid params");
    params_jit.create_plan();

    rocfft_params params_kernel;
    params_kernel.length = length;
    params_kernel.nbatch = batch;

    params_kernel.validate();

    if(!params_kernel.valid())
        throw std::runtime_error("invalid params");
    params_kernel.create_plan();

    std::vector<float> samples_jit;
    std::vector<float> samples_kernel;

    std::vector<gpubuf> data(1);
    gpubuf_t<size_t>    lengths_device;
    if(lengths_device.alloc(sizeof(size_t) * (length.size() + 1)) != hipSuccess)
        throw std::runtime_error("failed to alloc lengths");
    if(hipMemcpy(lengths_device.data(),
                 length.data(),
                 sizeof(size_t) * length.size(),
                 hipMemcpyHostToDevice)
           != hipSuccess
       || hipMemcpy(
              lengths_device.data() + length.size(), &batch, sizeof(size_t), hipMemcpyHostToDevice)
              != hipSuccess)
        throw std::runtime_error("failed to memcpy lengths");

    if(data[0].alloc(params_jit.isize.front() * sizeof(float) * 2) != hipSuccess)
        throw std::runtime_error("failed to alloc");

    params_jit.compute_input(data);

    hipEvent_wrapper_t start;
    hipEvent_wrapper_t stop;
    start.alloc();
    stop.alloc();

    const size_t       NTRIALS = 10;
    std::random_device randdev;
    while(samples_kernel.size() < NTRIALS || samples_jit.size() < NTRIALS)
    {
        bool run_jit = randdev() % 2;

        run_trial(params_kernel,
                  samples_kernel,
                  params_jit,
                  samples_jit,
                  start,
                  stop,
                  data.front(),
                  lengths_device,
                  run_jit);
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
    printf("median kernel: %f (%zu samples)\nmedian jit: %f (%zu samples)\n",
           get_median(samples_kernel),
           samples_kernel.size(),
           get_median(samples_jit),
           samples_jit.size());
}

int main()
{
    rocfft_params params;
    params.setup();

    run_testcase({256, 256, 256}, 10);

    params.cleanup();
}
