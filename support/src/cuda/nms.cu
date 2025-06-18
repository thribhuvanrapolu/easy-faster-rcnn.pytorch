// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const* const a, float const* const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float* dev_boxes, unsigned long long* dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 5];
    if (threadIdx.x < col_size) {
        for (int i = 0; i < 5; i++) {
            block_boxes[threadIdx.x * 5 + i] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + i];
        }
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float* cur_box = dev_boxes + cur_box_idx * 5;
        unsigned long long t = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0;

        for (int i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }

        int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
    AT_ASSERTM(boxes.is_cuda(), "boxes must be a CUDA tensor");
    auto scores = boxes.select(1, 4);
    auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
    auto boxes_sorted = boxes.index_select(0, order_t);

    int boxes_num = boxes.size(0);
    int col_blocks = (boxes_num + threadsPerBlock - 1) / threadsPerBlock;

    auto options = at::device(boxes.device()).dtype(at::kLong);
    at::Tensor mask = at::zeros({boxes_num * col_blocks}, options);

    float* boxes_dev = boxes_sorted.data_ptr<float>();
    unsigned long long* mask_dev;
    cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long));

    dim3 blocks(col_blocks, col_blocks);
    dim3 threads(threadsPerBlock);

    nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev);

    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    cudaMemcpy(mask_host.data(), mask_dev,
               sizeof(unsigned long long) * boxes_num * col_blocks,
               cudaMemcpyDeviceToHost);

    std::vector<unsigned long long> remv(col_blocks);
    memset(remv.data(), 0, sizeof(unsigned long long) * col_blocks);

    at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
    int64_t* keep_out = keep.data_ptr<int64_t>();

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_out[num_to_keep++] = i;
            unsigned long long* p = &mask_host[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
        }
    }

    cudaFree(mask_dev);

    return order_t.index({keep.narrow(0, 0, num_to_keep).to(order_t.device(), at::kLong)});
}
