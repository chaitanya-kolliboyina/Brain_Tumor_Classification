#include <torch/extension.h> #include <hip/hip_runtime.h>

global void normalize_kernel(float* input, float* output, int C, int H, int W, float* means, float* stds){
    int h = blockIdx.y * blockDim.y + threadIdx.y; 
    int w = blockIdx.x * blockDim.x + threadIdx.x; 
    if (h < H && w < W) {
        for (int c = 0; c < C; ++c) { 
            int idx = c * H * W + h * W + w;
            output[idx] = (input[idx] - means[c]) / stds[c]; 
        } 
    }
}

void normalize_hip(torch::Tensor input, torch::Tensor output, torch::Tensor means, torch::Tensor stds) {
    int C = input.size(0);
    int H = input.size(1); 
    int W = input.size(2); 
    dim3 block(16, 16); 
    dim3 grid((W + 15) / 16, (H + 15) / 16); 
    hipLaunchKernelGGL(normalize_kernel, grid, block, 0, 0, input.data_ptr(), output.data_ptr(), C, H, W, means.data_ptr(), stds.data_ptr());
    hipDeviceSynchronize(); // Ensure completion
    }