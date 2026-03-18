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

namespace cg = cooperative_groups;
namespace kernels {

enum request_type: uint8_t {
  request_type_insert = 0,
  request_type_erase = 1,
  request_type_find = 2
};

static constexpr auto target_blocks_per_sm = 8;

template <bool do_reclaim, typename device_func, typename index_type>
__launch_bounds__(index_type::host_reclaimer_type::block_size_, target_blocks_per_sm)
__global__ void batch_kernel(index_type index,
                             const device_func func,
                             uint32_t num_requests) {
  using allocator_type = typename index_type::device_allocator_context_type;
  using reclaimer_type = typename index_type::device_reclaimer_context_type;
  __shared__ cg::block_tile_memory<reclaimer_type::block_size_> block_tile_shmem;
  auto block = cg::this_thread_block(block_tile_shmem);
  auto tile = cg::tiled_partition<index_type::cg_tile_size>(block);
  allocator_type allocator{index.allocator_, tile};
  auto block_wide_tile = cg::tiled_partition<reclaimer_type::block_size_>(block);
  extern __shared__ uint32_t reclaimer_shmem_buffer[];
  reclaimer_type reclaimer{index.reclaimer_,
                           (do_reclaim && reclaimer_type::required_shmem_size() > 0) ? &reclaimer_shmem_buffer[0] : nullptr,
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
    func.load(regs, index, tile, allocator, thread_id, task_exists);
    if constexpr (do_reclaim) { reclaimer.begin_critical_section(block_wide_tile, allocator); }
    auto work_queue = tile.ballot(task_exists);
    while (work_queue) {
      int cur_rank = __ffs(work_queue) - 1;
      func.exec(index, regs, tile, allocator, reclaimer, cur_rank);
      if (tile.thread_rank() == cur_rank) { task_exists = false; }
      work_queue = tile.ballot(task_exists);
    }
    if constexpr (do_reclaim) { reclaimer.end_critical_section(block_wide_tile); }
    if (thread_id < num_requests) { func.store(regs, thread_id); }
  }
  if constexpr (do_reclaim) { reclaimer.drain_all(block_wide_tile, tile, allocator); }
}

template <typename index_type, typename device_func>
void launch_batch_kernel(index_type& index, const device_func& func, uint32_t num_requests, cudaStream_t stream) {
  static constexpr bool do_reclaim = device_func::reclaim_required;
  int block_size = index_type::host_reclaimer_type::block_size_;
  std::size_t shmem_size = do_reclaim ? sizeof(uint32_t) * index_type::device_reclaimer_context_type::required_shmem_size() : 0;
  int num_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm,
    kernels::batch_kernel<do_reclaim, device_func, index_type>,
    block_size,
    shmem_size);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  uint32_t num_blocks = num_blocks_per_sm * device_prop.multiProcessorCount;

  kernels::batch_kernel<do_reclaim><<<num_blocks, block_size, shmem_size, stream>>>(
      index, func, num_requests);
}

namespace GpuMasstree {

template <typename masstree, typename size_type>
__global__ void initialize_kernel(masstree tree, size_type* d_root_index) {
  using allocator_type = typename masstree::device_allocator_context_type;
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<masstree::cg_tile_size>(block);
  allocator_type allocator{tree.allocator_, tile};
  tree.allocate_root_node(d_root_index, tile, allocator);
}

template <bool enable_suffix, bool reuse_root, typename key_slice_type, typename size_type, typename value_type>
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
    uint64_t root_lane_elem;
  };
  // device-side functions
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, masstree& tree, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
      regs.value = d_values[thread_id];
    }
    if constexpr (reuse_root) {
      regs.root_lane_elem = tree.template cooperative_fetch_root<true>(tile, allocator);
    }
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    if constexpr (reuse_root) {
      tree.template cooperative_insert_from_root<enable_suffix>(regs.root_lane_elem, cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, update_if_exists);
    }
    else {
      tree.template cooperative_insert<enable_suffix>(cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, update_if_exists);
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool concurrent, bool reuse_root, typename key_slice_type, typename size_type, typename value_type>
struct find_device_func {
  static constexpr bool reclaim_required = false;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
    uint64_t root_lane_elem;
  };
  // device-side functions
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, masstree& tree, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
    }
    if constexpr (reuse_root) {
      regs.root_lane_elem = tree.template cooperative_fetch_root<true>(tile, allocator);
    }
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = reuse_root ?
      tree.template cooperative_find_from_root<concurrent>(regs.root_lane_elem, cur_key ,cur_key_length, tile, allocator) :
      tree.template cooperative_find<concurrent>(cur_key, cur_key_length, tile, allocator);
    if (tile.thread_rank() == cur_rank) {
      regs.value = cur_value;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    d_values[thread_id] = regs.value;
  }
};

template <bool concurrent, bool do_merge, bool do_remove_empty_root, bool reuse_root, typename key_slice_type, typename size_type, typename value_type>
struct erase_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    uint64_t root_lane_elem;
  };
  // device-side functions
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, masstree& tree, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
    }
    if constexpr (reuse_root) {
      regs.root_lane_elem = tree.template cooperative_fetch_root<true>(tile, allocator);
    }
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    if constexpr (reuse_root) {
      tree.template cooperative_erase_from_root<concurrent, do_merge, do_remove_empty_root>(regs.root_lane_elem, cur_key, cur_key_length, tile, allocator, reclaimer);
    }
    else {
      tree.template cooperative_erase<concurrent, do_merge, do_remove_empty_root>(cur_key, cur_key_length, tile, allocator, reclaimer);
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool use_upper_key, bool concurrent, bool reuse_root, typename key_slice_type, typename size_type, typename value_type>
struct scan_device_func {
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
    uint64_t root_lane_elem;
  };
  // device-side functions
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, masstree& tree, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.lower_key = &d_lower_keys[max_key_length * thread_id];
      regs.lower_key_length = d_lower_key_lengths ? d_lower_key_lengths[thread_id] : max_key_length;
      regs.upper_key = d_upper_keys ? &d_upper_keys[max_key_length * thread_id] : nullptr;
      regs.upper_key_length = d_upper_key_lengths ? d_upper_key_lengths[thread_id] : max_key_length;
      regs.value = d_values ? &d_values[max_count_per_query * thread_id] : nullptr;
      regs.out_key = d_out_keys ? &d_out_keys[max_count_per_query * max_key_length * thread_id] : nullptr;
      regs.out_key_length = d_out_key_lengths ? &d_out_key_lengths[max_count_per_query * thread_id] : nullptr;
    }
    if constexpr (reuse_root) {
      regs.root_lane_elem = tree.template cooperative_fetch_root<true>(tile, allocator);
    }
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
    auto cur_count = reuse_root ?
      tree.template cooperative_scan_from_root<use_upper_key, concurrent>(
        regs.root_lane_elem,
        cur_lower_key, cur_lower_key_length, tile, allocator, cur_upper_key, cur_upper_key_length,
        max_count_per_query, cur_value, cur_out_key, cur_out_key_length, max_key_length) :
      tree.template cooperative_scan<use_upper_key, concurrent>(
        cur_lower_key, cur_lower_key_length, tile, allocator, cur_upper_key, cur_upper_key_length,
        max_count_per_query, cur_value, cur_out_key, cur_out_key_length, max_key_length);
    if (tile.thread_rank() == cur_rank) {
      regs.count = cur_count;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    if (d_counts) { d_counts[thread_id] = regs.count; }
  }
};

template <bool enable_suffix,
          bool erase_do_merge,
          bool erase_do_remove_empty_root,
          bool reuse_root,
          typename key_slice_type,
          typename size_type,
          typename value_type>
struct mixed_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const request_type* d_types;
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  bool* d_results;
  bool insert_update_if_exists;
  // device-side registers
  struct dev_regs {
    request_type type;
    const key_slice_type* key;
    size_type key_length;
    value_type value;
    bool result;
    uint64_t root_lane_elem;
  };
  // device-side functions
  template <typename masstree, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, masstree& tree, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.type = d_types[thread_id];
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
      regs.value = d_values[thread_id];
    }
    if constexpr (reuse_root) {
      regs.root_lane_elem = tree.template cooperative_fetch_root<true>(tile, allocator);
    }
  }
  template <typename masstree, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(masstree& tree, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_type = tile.shfl(regs.type, cur_rank);
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    if (cur_type == request_type_insert) {
      auto cur_value = tile.shfl(regs.value, cur_rank);
      auto cur_result = reuse_root ?
        tree.template cooperative_insert_from_root<enable_suffix>(regs.root_lane_elem, cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, insert_update_if_exists) :  
        tree.template cooperative_insert<enable_suffix>(cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, insert_update_if_exists);
      if (tile.thread_rank() == cur_rank) { regs.result = cur_result; }
    }
    else if (cur_type == request_type_find) {
      auto cur_value = reuse_root ?
        tree.template cooperative_find_from_root<true>(regs.root_lane_elem, cur_key, cur_key_length, tile, allocator) :
        tree.template cooperative_find<true>(cur_key, cur_key_length, tile, allocator);
      if (tile.thread_rank() == cur_rank) { regs.value = cur_value; }
    }
    else if (cur_type == request_type_erase) {
      auto cur_result = reuse_root ?
        tree.template cooperative_erase_from_root<true, erase_do_merge, erase_do_remove_empty_root>(regs.root_lane_elem, cur_key, cur_key_length, tile, allocator, reclaimer) :
        tree.template cooperative_erase<true, erase_do_merge, erase_do_remove_empty_root>(cur_key, cur_key_length, tile, allocator, reclaimer);
      if (tile.thread_rank() == cur_rank) { regs.result = cur_result; }
    }
    else {  // request_type_successor
      assert(false); // TODO
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    if (regs.type <= request_type_erase) {  // insert of erase
      if (d_results) { d_results[thread_id] = regs.result; }
    }
    else {  // find or successor
      d_values[thread_id] = regs.value;
    }
  }
};

template <typename masstree, typename func>
__global__ void traverse_tree_nodes_kernel(masstree tree, func task) {
  // called with single warp; not parallelized for debug purpose
  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
  assert(blockDim.x == masstree::cg_tile_size && blockDim.y == 1 && blockDim.z == 1);
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<masstree::cg_tile_size>(block);
  task.init(tile);
  tree.cooperative_traverse_tree_nodes(task, tile);
  task.fini(tile);
}

} // namespace GpuMasstree

namespace GpuHashtable {

template <typename hashtable>
__global__ void initialize_kernel(hashtable table) {
  using allocator_type = typename hashtable::device_allocator_context_type;
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<hashtable::cg_tile_size>(block);
  auto bucket_index = blockIdx.x;
  allocator_type allocator{table.allocator_, tile};
  table.initialize_bucket(bucket_index, tile, allocator);
}

template <bool use_hash_tag, typename key_slice_type, typename size_type, typename value_type>
struct insert_device_func {
  static constexpr bool reclaim_required = false;
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
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
      regs.value = d_values[thread_id];
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    table.template cooperative_insert<use_hash_tag>(cur_key, cur_key_length, cur_value, tile, allocator, update_if_exists);
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool concurrent, bool use_hash_tag, typename key_slice_type, typename size_type, typename value_type>
struct find_device_func {
  static constexpr bool reclaim_required = false;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = table.template cooperative_find<concurrent, use_hash_tag>(cur_key, cur_key_length, tile, allocator);
    if (tile.thread_rank() == cur_rank) {
      regs.value = cur_value;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    d_values[thread_id] = regs.value;
  }
};

template <bool use_hash_tag, bool do_merge, typename key_slice_type, typename size_type, typename value_type>
struct erase_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    table.template cooperative_erase<use_hash_tag, do_merge>(cur_key, cur_key_length, tile, allocator, reclaimer);
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool use_hash_tag,
          bool erase_do_merge,
          typename key_slice_type,
          typename size_type,
          typename value_type>
struct mixed_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const request_type* d_types;
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  bool* d_results;
  bool insert_update_if_exists;
  // device-side registers
  struct dev_regs {
    request_type type;
    const key_slice_type* key;
    size_type key_length;
    value_type value;
    bool result;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.type = d_types[thread_id];
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
      regs.value = d_values[thread_id];
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_type = tile.shfl(regs.type, cur_rank);
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    if (cur_type == request_type_insert) {
      auto cur_value = tile.shfl(regs.value, cur_rank);
      auto cur_result = table.template cooperative_insert<use_hash_tag>(cur_key, cur_key_length, cur_value, tile, allocator, insert_update_if_exists);
      if (tile.thread_rank() == cur_rank) { regs.result = cur_result; }
    }
    else if (cur_type == request_type_find) {
      auto cur_value = table.template cooperative_find<true, use_hash_tag>(cur_key, cur_key_length, tile, allocator);
      if (tile.thread_rank() == cur_rank) { regs.value = cur_value; }
    }
    else if (cur_type == request_type_erase) {
      auto cur_result = table.template cooperative_erase<use_hash_tag, erase_do_merge>(cur_key, cur_key_length, tile, allocator, reclaimer);
      if (tile.thread_rank() == cur_rank) { regs.result = cur_result; }
    }
    else {  // request_type_successor
      assert(false); // TODO
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    if (regs.type <= request_type_erase) {  // insert of erase
      if (d_results) { d_results[thread_id] = regs.result; }
    }
    else {  // find or successor
      d_values[thread_id] = regs.value;
    }
  }
};

template <typename hashtable, typename func>
__global__ void traverse_nodes_kernel(hashtable table, func task) {
  // called with single warp; not parallelized for debug purpose
  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
  assert(blockDim.x == 32 && blockDim.y == 1 && blockDim.z == 1);
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<hashtable::cg_tile_size>(block);
  task.init(tile);
  table.cooperative_traverse_nodes(task, tile);
  task.fini(tile);
}

} // namespace GpuHashtable

namespace GpuLinearHashtable {

template <typename linearhashtable>
__global__ void initialize_kernel(linearhashtable table) {
  using allocator_type = typename linearhashtable::device_allocator_context_type;
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<linearhashtable::cg_tile_size>(block);
  auto bucket_index = blockIdx.x;
  allocator_type allocator{table.allocator_, tile};
  table.initialize_bucket(bucket_index, tile, allocator);
}

template <bool use_hash_tag, bool tag_use_same_hash, bool reuse_dirsize, typename key_slice_type, typename size_type, typename value_type>
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
    size_type directory_size;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
      regs.value = d_values[thread_id];
    }
    if constexpr (reuse_dirsize) {
      regs.directory_size = table.template cooperative_fetch_dirsize<true>();
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = tile.shfl(regs.value, cur_rank);
    if constexpr (reuse_dirsize) {
      table.template cooperative_insert_from_dirsize<use_hash_tag, tag_use_same_hash>(regs.directory_size, cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, update_if_exists);
    }
    else {
      table.template cooperative_insert<use_hash_tag, tag_use_same_hash>(cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, update_if_exists);
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool concurrent, bool use_hash_tag, bool tag_use_same_hash, bool reuse_dirsize, typename key_slice_type, typename size_type, typename value_type>
struct find_device_func {
  static constexpr bool reclaim_required = false;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    value_type value;
    size_type directory_size;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
    }
    if constexpr (reuse_dirsize) {
      regs.directory_size = table.template cooperative_fetch_dirsize<concurrent>();
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    auto cur_value = reuse_dirsize ?
      table.template cooperative_find_from_dirsize<concurrent, use_hash_tag, tag_use_same_hash>(regs.directory_size, cur_key, cur_key_length, tile, allocator) :
      table.template cooperative_find<concurrent, use_hash_tag, tag_use_same_hash>(cur_key, cur_key_length, tile, allocator);
    if (tile.thread_rank() == cur_rank) {
      regs.value = cur_value;
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    d_values[thread_id] = regs.value;
  }
};

template <bool use_hash_tag, bool tag_use_same_hash, bool do_merge_chains, bool do_merge_buckets, bool reuse_dirsize, typename key_slice_type, typename size_type, typename value_type>
struct erase_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  // device-side registers
  struct dev_regs {
    const key_slice_type* key;
    size_type key_length;
    size_type directory_size;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
    }
    if constexpr (reuse_dirsize) {
      regs.directory_size = table.template cooperative_fetch_dirsize<true>();
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    if constexpr (reuse_dirsize) {
      table.template cooperative_erase_from_dirsize<use_hash_tag, tag_use_same_hash, do_merge_chains, do_merge_buckets>(regs.directory_size, cur_key, cur_key_length, tile, allocator, reclaimer);
    }
    else {
      table.template cooperative_erase<use_hash_tag, tag_use_same_hash, do_merge_chains, do_merge_buckets>(cur_key, cur_key_length, tile, allocator, reclaimer);
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const noexcept {}
};

template <bool use_hash_tag,
          bool tag_use_same_hash,
          bool erase_do_merge_chains,
          bool erase_do_merge_buckets,
          bool reuse_dirsize,
          typename key_slice_type,
          typename size_type,
          typename value_type>
struct mixed_device_func {
  static constexpr bool reclaim_required = true;
  // kernel args
  const request_type* d_types;
  const key_slice_type* d_keys;
  size_type max_key_length;
  const size_type* d_key_lengths;
  value_type* d_values;
  bool* d_results;
  bool insert_update_if_exists;
  // device-side registers
  struct dev_regs {
    request_type type;
    const key_slice_type* key;
    size_type key_length;
    value_type value;
    bool result;
    size_type directory_size;
  };
  // device-side functions
  template <typename hashtable, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void load(dev_regs& regs, hashtable& table, const tile_type& tile, allocator_type& allocator, uint32_t thread_id, bool task_exists) const {
    if (task_exists) {
      regs.type = d_types[thread_id];
      regs.key = &d_keys[max_key_length * thread_id];
      regs.key_length = d_key_lengths ? d_key_lengths[thread_id] : max_key_length;
      regs.value = d_values[thread_id];
    }
    if constexpr (reuse_dirsize) {
      regs.directory_size = table.template cooperative_fetch_dirsize<true>();
    }
  }
  template <typename hashtable, typename tile_type, typename allocator_type, typename reclaimer_type>
  DEVICE_QUALIFIER void exec(hashtable& table, dev_regs& regs, tile_type& tile, allocator_type& allocator, reclaimer_type& reclaimer, int cur_rank) const {
    auto cur_type = tile.shfl(regs.type, cur_rank);
    auto cur_key = tile.shfl(regs.key, cur_rank);
    auto cur_key_length = tile.shfl(regs.key_length, cur_rank);
    if (cur_type == request_type_insert) {
      auto cur_value = tile.shfl(regs.value, cur_rank);
      auto cur_result = reuse_dirsize ?
        table.template cooperative_insert_from_dirsize<use_hash_tag, tag_use_same_hash>(regs.directory_size, cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, insert_update_if_exists) :
        table.template cooperative_insert<use_hash_tag, tag_use_same_hash>(cur_key, cur_key_length, cur_value, tile, allocator, reclaimer, insert_update_if_exists);
      if (tile.thread_rank() == cur_rank) { regs.result = cur_result; }
    }
    else if (cur_type == request_type_find) {
      auto cur_value = reuse_dirsize ?
        table.template cooperative_find_from_dirsize<true, use_hash_tag, tag_use_same_hash>(regs.directory_size, cur_key, cur_key_length, tile, allocator) :
        table.template cooperative_find<true, use_hash_tag, tag_use_same_hash>(cur_key, cur_key_length, tile, allocator);
      if (tile.thread_rank() == cur_rank) { regs.value = cur_value; }
    }
    else if (cur_type == request_type_erase) {
      auto cur_result = reuse_dirsize ?
        table.template cooperative_erase_from_dirsize<use_hash_tag, tag_use_same_hash, erase_do_merge_chains, erase_do_merge_buckets>(regs.directory_size, cur_key, cur_key_length, tile, allocator, reclaimer) :
        table.template cooperative_erase<use_hash_tag, tag_use_same_hash, erase_do_merge_chains, erase_do_merge_buckets>(cur_key, cur_key_length, tile, allocator, reclaimer);
      if (tile.thread_rank() == cur_rank) { regs.result = cur_result; }
    }
    else {  // request_type_successor
      assert(false); // TODO
    }
  }
  DEVICE_QUALIFIER void store(dev_regs& regs, uint32_t thread_id) const {
    if (regs.type <= request_type_erase) {  // insert of erase
      if (d_results) { d_results[thread_id] = regs.result; }
    }
    else {  // find or successor
      d_values[thread_id] = regs.value;
    }
  }
};

template <typename hashtable, typename func>
__global__ void traverse_nodes_kernel(hashtable table, func task) {
  // called with single warp; not parallelized for debug purpose
  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
  assert(blockDim.x == 32 && blockDim.y == 1 && blockDim.z == 1);
  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<hashtable::cg_tile_size>(block);
  task.init(tile);
  table.cooperative_traverse_nodes(task, tile);
  task.fini(tile);
}

} // namespace GpuLinearHashtable

} // namespace kernels
