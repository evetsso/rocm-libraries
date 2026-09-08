#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"

const char* callback_src{
    R"_CALLBACK_SRC_(
extern "C"
__device__ float2 load_callback(float2* input, size_t offset, void* cbdata, void* sharedMem)
{
  return input[offset];
}

extern "C"
__device__ void store_callback(float2* output, size_t offset, float2 element, void* cbdata, void* sharedMem)
{
  output[offset] = element;
}
)_CALLBACK_SRC_"};

__global__ void load_callback_kernel() {}

__global__ void store_callback_kernel() {}

void run_trial(rocfft_params&      params_kernel,
               std::vector<float>& samples_kernel,
               rocfft_params&      params_jit,
               std::vector<float>& samples_jit,
               hipEvent_wrapper_t& start,
               hipEvent_wrapper_t& stop,
               gpubuf&             data_orig,
               bool                run_jit)
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
    }

    params.execute(ptrs.data(), ptrs.data());

    if(!run_jit)
    {
        // launch kernel to apply store callback
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

void run_testcase(const std::vector<size_t>& length, size_t batch)
{
    rocfft_params params_jit;
    params_jit.length = length;
    params_jit.nbatch = batch;

    params_jit.validate();

    if(!params_jit.valid())
        throw std::runtime_error("invalid params");

    rocfft_params params_kernel;
    params_kernel.length = length;
    params_kernel.nbatch = batch;

    params_kernel.validate();

    if(!params_kernel.valid())
        throw std::runtime_error("invalid params");

    std::vector<float> samples_jit;
    std::vector<float> samples_kernel;

    std::vector<gpubuf> data(1);

    if(data[0].alloc(params_jit.isize.front() * sizeof(float) * 2) != hipSuccess)
        throw std::runtime_error("failed to alloc");

    params_jit.compute_input(data);

    hipEvent_wrapper_t start;
    hipEvent_wrapper_t stop;
    start.alloc();
    stop.alloc();

    const size_t       NTRIALS = 20;
    std::random_device randdev;
    while(samples_kernel.size() < NTRIALS && samples_jit.size() < NTRIALS)
    {
        bool run_jit = randdev() % 2;

        run_trial(params_kernel,
                  samples_kernel,
                  params_jit,
                  samples_jit,
                  start,
                  stop,
                  data.front(),
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
    printf(
        "median kernel: %f\nmedian jit: %f\n", get_median(samples_kernel), get_median(samples_jit));
}

int main()
{
    rocfft_params params;
    params.setup();

    run_testcase({256, 256, 256}, 10);

    params.cleanup();
}
