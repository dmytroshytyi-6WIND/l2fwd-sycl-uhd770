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

#include "pipeline.h"
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

Pipeline::Pipeline(int _index, int _workload_type, int _pkt_time_ns,
                   int _rx_queue, int _tx_queue, int _gpu_id) try
    : index(_index), workload_type(_workload_type), pkt_time_ns(_pkt_time_ns),
      rx_queue(_rx_queue), tx_queue(_tx_queue), gpu_id(_gpu_id) {
  uint32_t idx_l;
  int ret;
  int dev_id = 0;

  void *temp_comm_list = rte_zmalloc(
      NULL, sizeof(struct rte_gpu_comm_list) * MAX_BURSTS_X_QUEUE, 0);
  if (temp_comm_list == nullptr) {
    rte_errno = ENOMEM;
    return;
  }
  comm_list = static_cast<struct rte_gpu_comm_list *>(temp_comm_list);

  comm_list[0].status_d = (enum rte_gpu_comm_list_status *)rte_zmalloc(
      NULL, sizeof(enum rte_gpu_comm_list_status) * MAX_BURSTS_X_QUEUE, 0);

  if (comm_list[0].status_h == NULL) {
    /*
     * If CPU mapping is not supported by driver
     * use regular CPU registered memory.
     */
    comm_list[0].status_h = (enum rte_gpu_comm_list_status *)rte_zmalloc(
        NULL, sizeof(enum rte_gpu_comm_list_status) * MAX_BURSTS_X_QUEUE, 0);

    comm_list[0].status_d = comm_list[0].status_h;
  }

  for (idx_l = 0; idx_l < MAX_BURSTS_X_QUEUE; idx_l++) {
    comm_list[idx_l].pkt_list = (struct rte_gpu_comm_pkt *)rte_zmalloc(
        NULL, sizeof(struct rte_gpu_comm_pkt) * RTE_GPU_COMM_LIST_PKTS_MAX, 0);
    if (comm_list[idx_l].pkt_list == NULL) {
      rte_errno = ENOMEM;
      return;
    }

    if (ret < 0) {
      rte_errno = ENOMEM;
      return;
    }

    comm_list[idx_l].num_pkts = 0;
    comm_list[idx_l].dev_id = dev_id;

    comm_list[idx_l].mbufs = (struct rte_mbuf **)rte_zmalloc(
        NULL, sizeof(struct rte_mbuf *) * RTE_GPU_COMM_LIST_PKTS_MAX, 0);
    if (comm_list[idx_l].mbufs == NULL) {
      rte_errno = ENOMEM;
      return;
    }

    if (idx_l > 0) {
      comm_list[idx_l].status_h = &(comm_list[0].status_h[idx_l]);
      comm_list[idx_l].status_d = &(comm_list[0].status_d[idx_l]);

      ret = rte_gpu_comm_set_status(&comm_list[idx_l], RTE_GPU_COMM_LIST_FREE);
      if (ret < 0) {
        rte_errno = ENOMEM;
        return;
      }
    }
  }

  start_rx_measure = false;
  start_tx_measure = false;
  rx_pkts = 0;
  tx_pkts = 0;
  rx_bytes = 0;
  tx_bytes = 0;
  start_rx_core = Time::zeroNs();
  stop_rx_core = Time::zeroNs();
  start_tx_core = Time::zeroNs();
  stop_tx_core = Time::zeroNs();
  pipeline_force_quit = 0;

  dpct::queue_ptr c_stream;
  if (workload_type >= GPU_WORKLOAD)
    "cudaGetErrorString is not supported" /*cudaGetErrorString(CUDA_CHECK(cudaStreamCreateWithFlags(&(c_stream),
                                             cudaStreamNonBlocking)))*/
        ;

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //// PERSISTENT WORKLOAD (PK or GRAPHS)
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (workload_type >= GPU_PK_WORKLOAD) {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// ONE CUDA KERNEL PER PIPELINE
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (workload_type == GPU_PK_WORKLOAD)
      workload_launch_persistent_gpu_processing(
          comm_list, pkt_time_ns, PK_CUDA_BLOCKS, PK_CUDA_THREADS, c_stream);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// CUDA GRAPHS
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (workload_type == GPU_GRAPHS_WORKLOAD) {
      for (int index_g = 0; index_g < N_GRAPHS; index_g++) {
        for (int index_b = index_g * GRAPH_BURST;
             index_b < ((index_g + 1) * GRAPH_BURST); index_b++)
          workload_launch_gpu_graph_processing(&(comm_list[index_b]),
                                               pkt_time_ns, MAC_CUDA_BLOCKS,
                                               MAC_THREADS_BLOCK, c_stream);

        cudaGraphInstantiate(&winstance[index_g], wgraph[index_g], NULL, NULL,
                             0);
      }
    }
  }
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

Pipeline::~Pipeline() {

  rte_gpu_comm_destroy_list(comm_list, MAX_BURSTS_X_QUEUE);

  if (workload_type == GPU_GRAPHS_WORKLOAD) {
    for (int index_g = 0; index_g < N_GRAPHS; index_g++) {
      cudaGraphDestroy(wgraph[index_g]);
    }
  }
}

void Pipeline::terminateWorkload() try {

  if (workload_type == GPU_PK_WORKLOAD ||
      workload_type == GPU_GRAPHS_WORKLOAD) {
    printf("Terminating pending CUDA kernels...\n");
    for (int index_item = 0; index_item < MAX_BURSTS_X_QUEUE; index_item++) {
      if (rte_gpu_comm_set_status(&comm_list[index_item],
                                  RTE_GPU_COMM_LIST_ERROR)) {
        fprintf(stderr, "Can't set status RTE_GPU_COMM_LIST_ERROR on item %d\n",
                index_item);
      }
    }
    "cudaGetErrorString is not supported" /*cudaGetErrorString(CUDA_CHECK(cudaStreamSynchronize(c_stream)))*/
        ;
  }
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
