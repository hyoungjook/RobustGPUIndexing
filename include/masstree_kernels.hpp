/*
 *   Copyright 2022 The Regents of the University of California, Davis
 *   Copyright 2025 Hyoungjoo Kim, Carnegie Mellon University
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
#pragma once

#define _CG_ABI_EXPERIMENTAL
#include <cooperative_groups.h>
#include <macros.hpp>
#include <simple_dummy_reclaim.hpp>
#include <simple_debra_reclaim.hpp>

namespace GpuMasstree {
namespace kernels {
namespace cg = cooperative_groups;

template <typename masstree>
__global__ void initialize_kernel(masstree tree) {
  using allocator_type = typename masstree::device_allocator_context_type;
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<masstree::cg_tile_size>(block);
  allocator_type allocator{tree.allocator_, tile};
  tree.allocate_root_node(tile, allocator);
}

template <bool do_reclaim, typename device_func, typename masstree>
__global__ void batch_kernel(masstree tree,
                             const device_func func,
                             uint32_t num_requests) {
  using allocator_type = typename masstree::device_allocator_context_type;
  using reclaimer_type = typename masstree::device_reclaimer_context_type;
  __shared__ cg::block_tile_memory<reclaimer_type::block_size_> block_tile_shmem;
  auto block = cg::this_thread_block(block_tile_shmem);
  auto tile = cg::tiled_partition<masstree::cg_tile_size>(block);
  allocator_type allocator{tree.allocator_, tile};
  auto block_wide_tile = cg::tiled_partition<reclaimer_type::block_size_>(block);
  extern __shared__ uint32_t reclaimer_shmem_buffer[];
  reclaimer_type reclaimer{tree.reclaimer_,
                           (reclaimer_type::required_shmem_size() > 0) ? &reclaimer_shmem_buffer[0] : nullptr,
                           gridDim.x,
                           block_wide_tile};
  uint32_t block_size = blockDim.x;
  uint32_t num_request_blocks = (num_requests + block_size - 1) / block_size;
  uint32_t num_worker_blocks = gridDim.x;
  for (uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
       thread_id < (num_request_blocks * block_size);
       thread_id += (num_worker_blocks * block_size)) {
    bool task_exists = (thread_id < num_requests);
    typename device_func::dev_regs regs;
    if (task_exists) { regs = func.load(thread_id, tile); }
    if constexpr (do_reclaim) { reclaimer.begin_critical_section(block_wide_tile, allocator); }
    auto work_queue = tile.ballot(task_exists);
    while (work_queue) {
      int cur_rank = __ffs(work_queue) - 1;
      func.exec(tree, regs, tile, allocator, reclaimer, cur_rank);
      if (tile.thread_rank() == cur_rank) { task_exists = false; }
      work_queue = tile.ballot(task_exists);
    }
    if constexpr (do_reclaim) { reclaimer.end_critical_section(block_wide_tile); }
    if (thread_id < num_requests) { func.store(regs, thread_id); }
  }
  if constexpr (do_reclaim) { reclaimer.drain_all(block_wide_tile, tile, allocator); }
}

template <bool do_reclaim, typename device_func0, typename device_func1, typename masstree>
__global__ void batch_concurrent_two_funcs_kernel(masstree tree,
                                                  const device_func0 func0,
                                                  uint32_t num_requests0,
                                                  const device_func1 func1,
                                                  uint32_t num_requests1) {
  using allocator_type = typename masstree::device_allocator_context_type;
  using reclaimer_type = typename masstree::device_reclaimer_context_type;
  __shared__ cg::block_tile_memory<reclaimer_type::block_size_> block_tile_shmem;
  auto block = cg::this_thread_block(block_tile_shmem);
  auto tile = cg::tiled_partition<masstree::cg_tile_size>(block);
  allocator_type allocator{tree.allocator_, tile};
  auto block_wide_tile = cg::tiled_partition<reclaimer_type::block_size_>(block);
  extern __shared__ uint32_t reclaimer_shmem_buffer[];
  reclaimer_type reclaimer{tree.reclaimer_,
                           (reclaimer_type::required_shmem_size() > 0) ? &reclaimer_shmem_buffer[0] : nullptr,
                           gridDim.x,
                           block_wide_tile};
  // even'th tile -> do func0, odd'th tile -> do func1
  // assume num_requests0 ~= num_requests1
  uint32_t block_size = blockDim.x;
  uint32_t num_request_tiles = (max(num_requests0, num_requests1) + masstree::cg_tile_size - 1) / masstree::cg_tile_size * 2;
  uint32_t num_request_blocks = (num_request_tiles * masstree::cg_tile_size + block_size - 1) / block_size;
  uint32_t num_worker_blocks = gridDim.x;
  for (uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
       thread_id < (num_request_blocks * block_size);
       thread_id += (num_worker_blocks * block_size)) {
    uint32_t tile_id = thread_id / masstree::cg_tile_size;
    uint32_t request_id = (tile_id % 2);
    uint32_t thread_id_within_request = (tile_id / 2) * masstree::cg_tile_size + tile.thread_rank();
    if (request_id == 0) {
      bool task_exists = (thread_id_within_request < num_requests0);
      typename device_func0::dev_regs regs;
      if (task_exists) { regs = func0.load(thread_id_within_request, tile); }
      if constexpr (do_reclaim) { reclaimer.begin_critical_section(block_wide_tile, allocator); }
      auto work_queue = tile.ballot(task_exists);
      while (work_queue) {
        int cur_rank = __ffs(work_queue) - 1;
        func0.exec(tree, regs, tile, allocator, reclaimer, cur_rank);
        if (tile.thread_rank() == cur_rank) { task_exists = false; }
        work_queue = tile.ballot(task_exists);
      }
      if constexpr (do_reclaim) { reclaimer.end_critical_section(block_wide_tile); }
      if (thread_id_within_request < num_requests0) { func0.store(regs, thread_id_within_request); }
    }
    else { // request_id == 1
      bool task_exists = (thread_id_within_request < num_requests1);
      typename device_func1::dev_regs regs;
      if (task_exists) { regs = func1.load(thread_id_within_request, tile); }
      if constexpr (do_reclaim) { reclaimer.begin_critical_section(block_wide_tile, allocator); }
      auto work_queue = tile.ballot(task_exists);
      while (work_queue) {
        int cur_rank = __ffs(work_queue) - 1;
        func1.exec(tree, regs, tile, allocator, reclaimer, cur_rank);
        if (tile.thread_rank() == cur_rank) { task_exists = false; }
        work_queue = tile.ballot(task_exists);
      }
      if constexpr (do_reclaim) { reclaimer.end_critical_section(block_wide_tile); }
      if (thread_id_within_request < num_requests1) { func1.store(regs, thread_id_within_request); }
    }
  }
  if constexpr (do_reclaim) { reclaimer.drain_all(block_wide_tile, tile, allocator); }
}

template <typename key_slice_type, typename size_type, typename value_type>
struct insert_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  const value_type* d_values;
  bool update_if_exists;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .key = &d_keys[max_key_length * thread_id],
      .key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length,
      .value = d_values[thread_id]
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    tree.cooperative_insert(cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, update_if_exists);
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <typename key_slice_type, typename size_type, typename value_type>
struct find_device_func {
  static constexpr bool reclaim_required = false;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  bool concurrent;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .key = &d_keys[max_key_length * thread_id],
      .key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tree.cooperative_find(cur_key, cur_key_length, tile, allocator, concurrent);
    if (tile.thread_rank() == cur_rank) {
      regs.value = cur_value;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    d_values[thread_id] = regs.value;
  }
};

template <bool do_merge, bool do_remove_empty_root, typename key_slice_type, typename size_type, typename value_type>
struct erase_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  bool concurrent;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .key = &d_keys[max_key_length * thread_id],
      .key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length,
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    tree.cooperative_erase<do_merge, do_remove_empty_root>(cur_key, cur_key_length, tile, allocator, reclaimer, concurrent);
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool use_upper_key, typename key_slice_type, typename size_type, typename value_type>
struct range_device_func {
  static constexpr bool reclaim_required = false;
  // kernel args
  const key_slice_type* d_lower_keys;
  const size_type* d_lower_key_lengths;
  size_type max_key_length;
  size_type max_count_per_query;
  const key_slice_type* d_upper_keys;
  const size_type* d_upper_key_lengths;
  size_type* d_counts;
  value_type* d_values;
  key_slice_type* d_out_keys;
  size_type* d_out_key_lengths;
  bool concurrent;
  // device-side registers
  struct dev_regs {
    const key_slice_type* lower_key;
    size_type lower_key_length;
    const key_slice_type* upper_key;
    size_type upper_key_length;
    size_type count;
    value_type* value;
    key_slice_type* out_key;
    size_type* out_key_length;
  };
  // device-side functions
  template <typename tile_type>
  DEVICE_QUALIFIER dev_regs load(uint32_t thread_id, tile_type& tile) const {
    return dev_regs{
      .lower_key = &d_lower_keys[max_key_length * thread_id],
      .lower_key_length = d_lower_key_lengths ? d_lower_key_lengths[thread_id] : max_key_length,
      .upper_key = d_upper_keys ? &d_upper_keys[max_key_length * thread_id] : nullptr,
      .upper_key_length = d_upper_key_lengths ? d_upper_key_lengths[thread_id] : max_key_length,
      .value = d_values ? &d_values[max_count_per_query * thread_id] : nullptr,
      .out_key = d_out_keys ? &d_out_keys[max_count_per_query * max_key_length * thread_id] : nullptr,
      .out_key_length = d_out_key_lengths ? &d_out_key_lengths[max_count_per_query * thread_id] : nullptr
    };
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_lower_key = tile.shfl(regs.lower_key, cur_rank);
    auto cur_lower_key_length = tile.shfl(regs.lower_key_length, cur_rank);
    auto cur_upper_key = tile.shfl(regs.upper_key, cur_rank);
    auto cur_upper_key_length = tile.shfl(regs.upper_key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    auto cur_out_key = tile.shfl(regs.out_key, cur_rank);
    auto cur_out_key_length = tile.shfl(regs.out_key_length, cur_rank);
    auto cur_count = tree.cooperative_range<use_upper_key>(
      cur_lower_key, cur_lower_key_length, tile, allocator, cur_upper_key, cur_upper_key_length,
      max_count_per_query, cur_value, cur_out_key, cur_out_key_length, max_key_length, concurrent);
    if (tile.thread_rank() == cur_rank) {
      regs.count = cur_count;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    if (d_counts) { d_counts[thread_id] = regs.count; }
  }
};

} // namespace kernels
} // namespace GpuMasstree
