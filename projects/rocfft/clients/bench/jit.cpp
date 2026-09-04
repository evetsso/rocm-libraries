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

float run_trial(rocfft_params&      params,
                hipEvent_wrapper_t& start,
                hipEvent_wrapper_t& stop,
                std::vector<void*>& ptrs)
{
    if(hipEventRecord(start) != hipSuccess)
        throw std::runtime_error("failed to record start");
    params.execute(ptrs.data(), ptrs.data());
    if(hipEventRecord(stop) != hipSuccess)
        throw std::runtime_error("failed to record start");

    if(hipEventSynchronize(stop) != hipSuccess)
        throw std::runtime_error("failed to sync");
    float elapsed = 0.0;
    if(hipEventElapsedTime(&elapsed, start, stop) != hipSuccess)
        throw std::runtime_error("hipEventElapsedTime failed");

    printf("%.5f\n", static_cast<double>(elapsed));
    return elapsed;
}

void run_testcase(const std::vector<size_t>& length, size_t batch)
{
    rocfft_params params;
    params.length = length;
    params.nbatch = batch;

    params.validate();

    if(!params.valid())
        throw std::runtime_error("invalid params");

    std::vector<gpubuf> data(1);
    std::vector<void*>  ptrs(1);

    if(data[0].alloc(params.isize.front() * sizeof(float) * 2) != hipSuccess)
        throw std::runtime_error("failed to alloc");
    ptrs[0] = data[0].data();

    params.compute_input(data);

    hipEvent_wrapper_t start;
    hipEvent_wrapper_t stop;
    start.alloc();
    stop.alloc();

    run_trial(params, start, stop, ptrs);
}

int main()
{
    rocfft_params params;
    params.setup();

    run_testcase({256, 256, 256}, 10);

    params.cleanup();
}
