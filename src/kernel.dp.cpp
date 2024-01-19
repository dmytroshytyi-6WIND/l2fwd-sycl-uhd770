/* Copyright (c) 2023-2024, Dmytro Shytyi dmytro@shytyi.net */

/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include "cuda_related.h"
#include <dpct/dpct.hpp>
#include <rte_ether.h>
#include <rte_gpudev.h>
#include <sycl/sycl.hpp>

__dpct_inline__ unsigned long long __globaltimer() {
  unsigned long long globaltimer;
  return globaltimer;
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Regular CUDA kernel -w 2
/////////////////////////////////////////////////////////////////////////////////////////
void kernel_mac_update(struct rte_gpu_comm_list *comm_list, uint64_t wtime_n,
                       const sycl::nd_item<3> &item_ct1) {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  uint16_t temp;
  unsigned long long pkt_start;

  if (idx < comm_list->num_pkts && comm_list->pkt_list[idx].addr != 0) {
    if (wtime_n)
      pkt_start = __globaltimer();

    struct rte_ether_hdr *eth =
        (struct rte_ether_hdr *)(((uint8_t *)(comm_list->pkt_list[idx].addr)));
    uint16_t *src_addr = (uint16_t *)(&eth->src_addr);
    uint16_t *dst_addr = (uint16_t *)(&eth->dst_addr);

#ifdef DEBUG_PRINT
    /* Code to verify source and dest of ethernet addresses */
    uint8_t *src = (uint8_t *)(&eth->src_addr);
    uint8_t *dst = (uint8_t *)(&eth->dst_addr);
    printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: "
           "%02x:%02x:%02x:%02x:%02x:%02x\n",
           src[0], src[1], src[2], src[3], src[4], src[5], dst[0], dst[1],
           dst[2], dst[3], dst[4], dst[5]);
#endif

    /* MAC update */
    temp = dst_addr[0];
    dst_addr[0] = src_addr[0];
    src_addr[0] = temp;
    temp = dst_addr[1];
    dst_addr[1] = src_addr[1];
    src_addr[1] = temp;
    temp = dst_addr[2];
    dst_addr[2] = src_addr[2];
    src_addr[2] = temp;

    if (wtime_n)
      while ((__globaltimer() - pkt_start) < wtime_n)
        ;
#ifdef DEBUG_PRINT
    /* Code to verify source and dest of ethernet addresses */
    printf("After Swap, Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
           ((uint8_t *)(src_addr))[0], ((uint8_t *)(src_addr))[1],
           ((uint8_t *)(src_addr))[2], ((uint8_t *)(src_addr))[3],
           ((uint8_t *)(src_addr))[4], ((uint8_t *)(src_addr))[5],
           ((uint8_t *)(dst_addr))[0], ((uint8_t *)(dst_addr))[1],
           ((uint8_t *)(dst_addr))[2], ((uint8_t *)(dst_addr))[3],
           ((uint8_t *)(dst_addr))[4], ((uint8_t *)(dst_addr))[5]);
#endif
  }

  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

  item_ct1.barrier();

  if (idx == 0) {
    RTE_GPU_VOLATILE(*(comm_list->status_d)) = RTE_GPU_COMM_LIST_DONE;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
  }

  item_ct1.barrier();
}
void workload_launch_gpu_processing(struct rte_gpu_comm_list *comm_list,
                                    uint64_t wtime_n, int cuda_blocks,
                                    int cuda_threads, dpct::queue_ptr stream) {
  assert(cuda_blocks == 1);
  assert(cuda_threads > 0);

  if (comm_list) {
    auto platforms = sycl::platform::get_platforms();
    /*
    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R)
    FPGA Emulation Device 1.2 [2023.16.7.0.21_160000] [opencl:cpu:1] Intel(R)
    OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
    [opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) UHD Graphics 770 3.0
    [23.22.026516] [opencl:cpu:3] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM)
    i9-13900K 3.0 [2023.16.7.0.21_160000] [opencl:acc:4] Intel(R) FPGA Emulation
    Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2
    [2023.16.7.0.21_160000] [ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero,
    Intel(R) UHD Graphics 770 1.3 [1.3.26516]
    */
    sycl::queue c_stream_tmp{platforms[2].get_devices()[0]};

    sycl::buffer<rte_gpu_comm_list_status, 1> comm_list_status_buffer(
        comm_list->status_h, sycl::range<1>(sizeof(rte_gpu_comm_list_status)),
        {sycl::property::buffer::use_host_ptr()});

    c_stream_tmp.submit([&](sycl::handler &h) {
      sycl::stream out(1024, 256, h);

      auto comm_list_status_write_accessor =
          comm_list_status_buffer
              .template get_access<sycl::access::mode::write>(h);
      // auto comm_pkt_accessor = comm_pkt_buffer.template
      // get_access<sycl::access::mode::read>(h);

      h.parallel_for<class FillMacawefawfeawfaregae>(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1) * sycl::range<3>(1, 1, 512),
                            sycl::range<3>(1, 1, 512)),
          [=](sycl::nd_item<3> item_ct1) {
            uint16_t temp;
            int idx =
                item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                item_ct1.get_local_id(2);

            if (idx == 511) {

              int status = RTE_GPU_COMM_LIST_DONE;
              rte_gpu_comm_list_status *external_status =
                  comm_list_status_write_accessor.get_pointer();
              std::memcpy(external_status, &status,
                          (sizeof(rte_gpu_comm_list_status)));
            }
          });
    });
  }
}
void workload_launch_gpu_processing2(struct rte_gpu_comm_list *comm_list,
                                     uint64_t wtime_n, int cuda_blocks,
                                     int cuda_threads, dpct::queue_ptr stream) {
  assert(cuda_blocks == 1);
  assert(cuda_threads > 0);

  if (comm_list) {

    auto platforms = sycl::platform::get_platforms();
    /*
    [opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R)
    FPGA Emulation Device 1.2 [2023.16.7.0.21_160000] [opencl:cpu:1] Intel(R)
    OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
    [opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) UHD Graphics 770 3.0
    [23.22.026516] [opencl:cpu:3] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM)
    i9-13900K 3.0 [2023.16.7.0.21_160000] [opencl:acc:4] Intel(R) FPGA Emulation
    Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2
    [2023.16.7.0.21_160000] [ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero,
    Intel(R) UHD Graphics 770 1.3 [1.3.26516]
    */
    sycl::queue c_stream_tmp{platforms[2].get_devices()[0]};

    sycl::buffer<rte_gpu_comm_list> comm_list_buffer(
        comm_list, (sizeof(rte_gpu_comm_list) * MAX_BURSTS_X_QUEUE));
    sycl::buffer<rte_gpu_comm_pkt> comm_pkt_buffer(
        comm_list->pkt_list, (sizeof(rte_gpu_comm_pkt) * MAX_BURSTS_X_QUEUE));
    rte_gpu_comm_list_status *arr_s =
        sycl::malloc_shared<rte_gpu_comm_list_status>(
            (sizeof(rte_gpu_comm_list_status)), c_stream_tmp);

    c_stream_tmp
        .submit([&](sycl::handler &h) {
          sycl::stream out(1024, 256, h);

          auto comm_list_accessor =
              comm_list_buffer
                  .template get_access<sycl::access::mode::read_write>(h);
          auto comm_pkt_accessor =
              comm_pkt_buffer
                  .template get_access<sycl::access::mode::read_write>(h);

          h.parallel_for<class FillMacawefawfeawfaregae2>(
              sycl::nd_range<3>(sycl::range<3>(1, 1, 1) *
                                    sycl::range<3>(1, 1, 16),
                                sycl::range<3>(1, 1, 16)),
              [=](sycl::nd_item<3> item_ct1) {
                uint16_t temp;

                int idx =
                    item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

                struct rte_gpu_comm_list *comm_list_internal =
                    comm_list_accessor.get_pointer();
                struct rte_gpu_comm_pkt *comm_pkt_internal =
                    comm_pkt_accessor.get_pointer();

                out << "1" << sycl::endl;
                if (idx < comm_list_internal->num_pkts &&
                    comm_pkt_internal[idx].addr != 0) {
                  out << "2, idx: " << idx << sycl::endl;
                  struct rte_ether_hdr *eth = (struct rte_ether_hdr *)((
                      (uint8_t *)(comm_pkt_internal[idx].addr)));
                  uint16_t *src_addr = (uint16_t *)(&eth->src_addr);
                  uint16_t *dst_addr = (uint16_t *)(&eth->dst_addr);

                  /* MAC update */
                  temp = dst_addr[0];
                  dst_addr[0] = src_addr[0];
                  src_addr[0] = temp;
                  temp = dst_addr[1];
                  dst_addr[1] = src_addr[1];
                  src_addr[1] = temp;
                  temp = dst_addr[2];
                  dst_addr[2] = src_addr[2];
                  src_addr[2] = temp;
                }

                if (idx == 0) {
                  RTE_GPU_VOLATILE(*(comm_list_internal->status_d)) =
                      RTE_GPU_COMM_LIST_DONE;
                }
                if (idx == 15) {
                  RTE_GPU_VOLATILE(comm_list_internal->status_h[0]) =
                      RTE_GPU_COMM_LIST_DONE;
                  std::memcpy(arr_s, comm_list_internal->status_h,
                              (sizeof(rte_gpu_comm_list_status)));
                }

                out << "STATUS(GPU2): "
                    << static_cast<int>(comm_list_internal->status_h[0])
                    << sycl::endl;
              });
        })
        .wait();

  std:
    memcpy(comm_list->status_d, arr_s, (sizeof(rte_gpu_comm_list_status)));
    std::cout << "STATUS(HOST3): " << static_cast<int>(comm_list->status_d[0])
              << std::endl;
    sycl::free(arr_s, c_stream_tmp);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Persistent CUDA kernel -w 3
/////////////////////////////////////////////////////////////////////////////////////////
void kernel_persistent_mac_update(struct rte_gpu_comm_list *comm_list,
                                  uint64_t wtime_n,
                                  const sycl::nd_item<3> &item_ct1,
                                  uint32_t *wait_status_shared) {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int item_index = 0;
  unsigned long long pkt_start;
  struct rte_ether_hdr *eth;
  uint16_t *src_addr, *dst_addr, temp;
  uint32_t wait_status;

  item_ct1.barrier();

  while (1) {
    if (idx == 0) {
      while (1) {
        wait_status = RTE_GPU_VOLATILE(comm_list[item_index].status_d[0]);
        if (wait_status != RTE_GPU_COMM_LIST_FREE) {
          wait_status_shared[0] = wait_status;
          sycl::atomic_fence(sycl::memory_order::acq_rel,
                             sycl::memory_scope::work_group);
          break;
        }
      }
    }

    item_ct1.barrier();

    if (wait_status_shared[0] != RTE_GPU_COMM_LIST_READY)
      break;

    if (idx < comm_list[item_index].num_pkts &&
        comm_list->pkt_list[idx].addr != 0) {
      if (wtime_n)
        pkt_start = __globaltimer();

      eth = (struct rte_ether_hdr *)((
          (uint8_t *)(comm_list[item_index].pkt_list[idx].addr)));
      src_addr = (uint16_t *)(&eth->src_addr);
      dst_addr = (uint16_t *)(&eth->dst_addr);

#ifdef DEBUG_PRINT
      /* Code to verify source and dest of ethernet addresses */
      uint8_t *src = (uint8_t *)(&eth->src_addr);
      uint8_t *dst = (uint8_t *)(&eth->dst_addr);
      printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: "
             "%02x:%02x:%02x:%02x:%02x:%02x\n",
             src[0], src[1], src[2], src[3], src[4], src[5], dst[0], dst[1],
             dst[2], dst[3], dst[4], dst[5]);
#endif
      temp = dst_addr[0];
      dst_addr[0] = src_addr[0];
      src_addr[0] = temp;
      temp = dst_addr[1];
      dst_addr[1] = src_addr[1];
      src_addr[1] = temp;
      temp = dst_addr[2];
      dst_addr[2] = src_addr[2];
      src_addr[2] = temp;

#ifdef DEBUG_PRINT
      /* Code to verify source and dest of ethernet addresses */
      printf("After Swap, Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
             ((uint8_t *)(src_addr))[0], ((uint8_t *)(src_addr))[1],
             ((uint8_t *)(src_addr))[2], ((uint8_t *)(src_addr))[3],
             ((uint8_t *)(src_addr))[4], ((uint8_t *)(src_addr))[5],
             ((uint8_t *)(dst_addr))[0], ((uint8_t *)(dst_addr))[1],
             ((uint8_t *)(dst_addr))[2], ((uint8_t *)(dst_addr))[3],
             ((uint8_t *)(dst_addr))[4], ((uint8_t *)(dst_addr))[5]);
#endif
      if (wtime_n)
        while ((__globaltimer() - pkt_start) < wtime_n)
          ;
    }

    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

    item_ct1.barrier();

    if (idx == 0) {
      RTE_GPU_VOLATILE(comm_list[item_index].status_d[0]) =
          RTE_GPU_COMM_LIST_DONE;

      sycl::atomic_fence(sycl::memory_order::acq_rel,
                         sycl::memory_scope::system);
    }

    item_index = (item_index + 1) % MAX_BURSTS_X_QUEUE;
  }
}

void workload_launch_persistent_gpu_processing(
    struct rte_gpu_comm_list *comm_list, uint64_t wtime_n, int cuda_blocks,
    int cuda_threads, dpct::queue_ptr stream) {
  assert(cuda_blocks == 1);
  assert(cuda_threads > 0);
  if (comm_list == NULL)
    return;

  "cudaGetErrorString is not supported" /*cudaGetErrorString(CUDA_CHECK(cudaGetLastError()))*/
      ;

  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> wait_status_shared_acc_ct1(
        sycl::range<1>(1), cgh);

    cgh.parallel_for<class kernel_2>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, cuda_blocks) *
                              sycl::range<3>(1, 1, cuda_threads),
                          sycl::range<3>(1, 1, cuda_threads)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_persistent_mac_update(
              comm_list, wtime_n, item_ct1,
              wait_status_shared_acc_ct1.get_pointer());
        });
  });

  "cudaGetErrorString is not supported" /*cudaGetErrorString(CUDA_CHECK(cudaGetLastError()))*/
      ;
}

/////////////////////////////////////////////////////////////////////////////////////////
//// CUDA GRAPHS kernel -w 4
/////////////////////////////////////////////////////////////////////////////////////////
void kernel_graphs_mac_update(struct rte_gpu_comm_list *comm_item_list,
                              uint64_t wtime_n,
                              const sycl::nd_item<3> &item_ct1,
                              uint32_t *wait_status_shared) {
  int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  uint16_t temp;
  unsigned long long pkt_start;
  uint32_t wait_status;

  if (idx == 0) {
    while (1) {
      wait_status = RTE_GPU_VOLATILE(comm_item_list->status_d[0]);
      if (wait_status != RTE_GPU_COMM_LIST_FREE) {
        wait_status_shared[0] = wait_status;

        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::work_group);
        break;
      }
    }
  }

  item_ct1.barrier();

  if (wait_status_shared[0] != RTE_GPU_COMM_LIST_READY)
    return;

  if (idx < comm_item_list->num_pkts &&
      comm_item_list->pkt_list[idx].addr != 0) {
    if (wtime_n)
      pkt_start = __globaltimer();

    struct rte_ether_hdr *eth = (struct rte_ether_hdr *)((
        (uint8_t *)(comm_item_list->pkt_list[idx].addr)));
    uint16_t *src_addr = (uint16_t *)(&eth->src_addr);
    uint16_t *dst_addr = (uint16_t *)(&eth->dst_addr);

#ifdef DEBUG_PRINT
    /* Code to verify source and dest of ethernet addresses */
    uint8_t *src = (uint8_t *)(&eth->src_addr);
    uint8_t *dst = (uint8_t *)(&eth->dst_addr);
    printf("GRAPHS before Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: "
           "%02x:%02x:%02x:%02x:%02x:%02x\n",
           src[0], src[1], src[2], src[3], src[4], src[5], dst[0], dst[1],
           dst[2], dst[3], dst[4], dst[5]);
#endif

    temp = dst_addr[0];
    dst_addr[0] = src_addr[0];
    src_addr[0] = temp;
    temp = dst_addr[1];
    dst_addr[1] = src_addr[1];
    src_addr[1] = temp;
    temp = dst_addr[2];
    dst_addr[2] = src_addr[2];
    src_addr[2] = temp;

    if (wtime_n)
      while ((__globaltimer() - pkt_start) < wtime_n)
        ;

#ifdef DEBUG_PRINT
    /* Code to verify source and dest of ethernet addresses */
    printf("GRAPHS after Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
           ((uint8_t *)(src_addr))[0], ((uint8_t *)(src_addr))[1],
           ((uint8_t *)(src_addr))[2], ((uint8_t *)(src_addr))[3],
           ((uint8_t *)(src_addr))[4], ((uint8_t *)(src_addr))[5],
           ((uint8_t *)(dst_addr))[0], ((uint8_t *)(dst_addr))[1],
           ((uint8_t *)(dst_addr))[2], ((uint8_t *)(dst_addr))[3],
           ((uint8_t *)(dst_addr))[4], ((uint8_t *)(dst_addr))[5]);
#endif
  }

  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
  item_ct1.barrier();

  if (idx == 0) {
    RTE_GPU_VOLATILE(*(comm_item_list->status_d)) = RTE_GPU_COMM_LIST_DONE;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
  }
  item_ct1.barrier();
}

void workload_launch_gpu_graph_processing(struct rte_gpu_comm_list *bitem,
                                          uint64_t wtime_n, int cuda_blocks,
                                          int cuda_threads,
                                          dpct::queue_ptr stream) {
  assert(cuda_blocks == 1);
  assert(cuda_threads > 0);

  "cudaGetErrorString is not supported" /*cudaGetErrorString(CUDA_CHECK(cudaGetLastError()))*/
      ;
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> wait_status_shared_acc_ct1(
        sycl::range<1>(1), cgh);

    cgh.parallel_for<class kernel_3>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, cuda_blocks) *
                              sycl::range<3>(1, 1, cuda_threads),
                          sycl::range<3>(1, 1, cuda_threads)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_graphs_mac_update(bitem, wtime_n, item_ct1,
                                   wait_status_shared_acc_ct1.get_pointer());
        });
  });

  "cudaGetErrorString is not supported" /*cudaGetErrorString(CUDA_CHECK(cudaGetLastError()))*/
      ;
}
