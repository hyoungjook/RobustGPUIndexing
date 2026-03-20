/*
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
#define _CG_ABI_EXPERIMENTAL  // enable experimental CGs API

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <kernels.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <nodes.hpp>
#include <compute_hash.hpp>
#include <queue>
#include <sstream>
#include <type_traits>

#include <dynamic_stack.hpp>
#include <simple_bump_alloc.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_dummy_reclaim.hpp>
#include <simple_debra_reclaim.hpp>

namespace GpuHashtable {

template <typename Allocator,
          typename Reclaimer,
          bool use_subwarp = true>
struct gpu_chainhashtable {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using table_ptr_type = uint64_t;
  static constexpr bool use_subwarp_ = use_subwarp;
  static auto constexpr bucket_size = 32;
  static std::size_t constexpr bucket_bytes = sizeof(elem_type) * bucket_size;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  using host_allocator_type = Allocator;
  using device_allocator_instance_type = typename host_allocator_type::device_instance_type;
  using device_allocator_context_type = device_allocator_context<host_allocator_type>;

  using host_reclaimer_type = Reclaimer;
  using device_reclaimer_instance_type = typename host_reclaimer_type::device_instance_type;
  using device_reclaimer_context_type = device_reclaimer_context<host_reclaimer_type>;

  gpu_chainhashtable() = delete;
  gpu_chainhashtable(const host_allocator_type& host_allocator,
                     const host_reclaimer_type& host_reclaimer,
                     size_type num_buckets)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance())
      , num_buckets_(num_buckets) {
    allocate();
  }
  gpu_chainhashtable(const host_allocator_type& host_allocator,
                     const host_reclaimer_type& host_reclaimer,
                     std::size_t num_elements,
                     float fill_factor)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance()) {
    num_buckets_ = std::max(static_cast<std::size_t>(static_cast<double>(num_elements) / fill_factor / 15), 1UL);
    allocate();
  }

  gpu_chainhashtable& operator=(const gpu_chainhashtable& other) = delete;
  gpu_chainhashtable(const gpu_chainhashtable& other)
      : d_table_(other.d_table_)
      , is_owner_(false)
      , num_buckets_(other.num_buckets_)
      , allocator_(other.allocator_)
      , reclaimer_(other.reclaimer_) {}

  ~gpu_chainhashtable() {
    deallocate();
  }

  // host-side APIs
  // if key_lengths == NULL, we use max_key_length as a fixed length
  template <bool concurrent = false,
            bool use_hash_tag = true>
  void find(const key_slice_type* keys,
            const size_type max_key_length,
            const size_type* key_lengths,
            value_type* values,
            const size_type num_keys,
            cudaStream_t stream = 0) {
    kernels::GpuHashtable::find_device_func<gpu_chainhashtable, concurrent, use_hash_tag>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true>
  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0,
              bool update_if_exists = false) {
    kernels::GpuHashtable::insert_device_func<gpu_chainhashtable, use_hash_tag>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true,
            bool do_merge = true>
  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0) {
    kernels::GpuHashtable::erase_device_func<gpu_chainhashtable, use_hash_tag, do_merge>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true,
            bool erase_do_merge = true>
  void mixed_batch(const kernels::request_type* request_types,
                   const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   value_type* values,
                   bool* results,
                   const size_type num_requests,
                   cudaStream_t stream = 0,
                   bool insert_update_if_exists = false) {
    kernels::GpuHashtable::mixed_device_func<gpu_chainhashtable, use_hash_tag, erase_do_merge>
      func{.d_types = request_types, .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .d_results = results, .insert_update_if_exists = insert_update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_requests, stream);
  }

  // device-side APIs
  template <bool concurrent, bool use_hash_tag, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index;
    const bool more_key = (key_length > 1);
    if (use_hash_tag && more_key) {
      auto hash = utils::compute_hashx2<utils::PRIME0, utils::PRIME1>(key, key_length, tile);
      bucket_index = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index = utils::compute_hash<utils::PRIME0>(key, key_length, tile);
      first_slice = key[0];
    }
    bucket_index %= num_buckets_;
    suffix_type suffix_if_found(tile, allocator);
    auto node = node_type(bucket_index, tile, allocator);
    node.template load_from_array<concurrent>(d_table_);
    int location_if_found = coop_traverse_until_found<concurrent, use_hash_tag>(
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
    if (location_if_found >= 0) { // found
      if (more_key) {
        return suffix_if_found.get_value();
      }
      else {
        return node.get_value_from_location(location_if_found);
      }
    }
    // not found
    return invalid_value;
  }

  template <bool use_hash_tag, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           bool update_if_exists = false) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index;
    const bool more_key = (key_length > 1);
    if (use_hash_tag && more_key) {
      auto hash = utils::compute_hashx2<utils::PRIME0, utils::PRIME1>(key, key_length, tile);
      bucket_index = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index = utils::compute_hash<utils::PRIME0>(key, key_length, tile);
      first_slice = key[0];
    }
    bucket_index %= num_buckets_;
    node_type::lock(d_table_, bucket_index, tile);
    suffix_type suffix_if_found(tile, allocator);
    auto node = node_type(bucket_index, tile, allocator);
    node.template load_from_array<true>(d_table_);
    int location_if_found = coop_traverse_until_found<false, use_hash_tag>( // use weak load here b/c the first load did memory_order_acquire
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
    if (location_if_found >= 0) { // already exists
      if (update_if_exists) {
        if (more_key) {
          suffix_if_found.update_value(value);
          suffix_if_found.store_head();
        }
        else {
          node.update(location_if_found, value);
          node.template store_head_to_array_aux_to_allocator<false>(d_table_);
        }
      }
      node_type::unlock(d_table_, bucket_index, tile);
      return update_if_exists;
    }
    // not exists
    value_type to_insert = value;
    if (more_key) {
      to_insert = allocator.allocate(tile);
      auto suffix = suffix_type(to_insert, tile, allocator);
      static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1;
      suffix.create_from(key + suffix_offset, key_length - suffix_offset, value);
      suffix.store_head();
    }
    if (node.is_full()) {
      auto next_index = allocator.allocate(tile);
      auto new_node = node_type(next_index, tile, allocator);
      new_node.initialize_empty(false);
      new_node.insert(first_slice, to_insert, more_key);
      // write order: new_node -> node
      new_node.template store_to_allocator<false>();
      node.set_next_index(next_index);
      node.set_has_next();
    }
    else { // !node.is_full()
      node.insert(first_slice, to_insert, more_key);
    }
    node.template store_head_to_array_aux_to_allocator<true>(d_table_);
    node_type::unlock(d_table_, bucket_index, tile);
    return true;
  }

  template <bool use_hash_tag, bool do_merge, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index;
    const bool more_key = (key_length > 1);
    if (use_hash_tag && more_key) {
      auto hash = utils::compute_hashx2<utils::PRIME0, utils::PRIME1>(key, key_length, tile);
      bucket_index = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index = utils::compute_hash<utils::PRIME0>(key, key_length, tile);
      first_slice = key[0];
    }
    bucket_index %= num_buckets_;
    node_type::lock(d_table_, bucket_index, tile);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    auto node = node_type(bucket_index, tile, allocator);
    node.template load_from_array<true>(d_table_);
    if constexpr (do_merge) {
      location_if_found = coop_traverse_until_found_merge<use_hash_tag>(
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator, reclaimer);
    }
    else {
      location_if_found = coop_traverse_until_found<false, use_hash_tag>( // use weak load here b/c the first load did memory_order_acquire
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
    }
    if (location_if_found >= 0) { // exists
      node.erase(location_if_found);
      node.template store_head_to_array_aux_to_allocator<false>(d_table_);
      if (more_key) {
        suffix_if_found.retire(reclaimer);
      }
      node_type::unlock(d_table_, bucket_index, tile);
      return true;
    }
    // not exists
    node_type::unlock(d_table_, bucket_index, tile);
    return false;
  }

 private:
  // device-side helper functions
  template <bool concurrent, bool use_hash_tag, typename tile_type>
  DEVICE_QUALIFIER int coop_traverse_until_found(hashtable_node<tile_type, device_allocator_context_type>& node,
                                                 const key_slice_type& first_slice,
                                                 bool more_key,
                                                 const key_slice_type* key,
                                                 const size_type& key_length,
                                                 suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                 const tile_type& tile,
                                                 device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    while (true) {
      uint32_t to_check = node.match_key_in_node(first_slice, more_key);
      if (more_key) {
        // if length > 1, compare suffixes
        while (to_check != 0) {
          auto cur_location = __ffs(to_check) - 1;
          auto suffix_index = node.get_value_from_location(cur_location);
          auto suffix = suffix_type(suffix_index, tile, allocator);
          suffix.load_head();
          static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1;
          if (suffix.streq(key + suffix_offset, key_length - suffix_offset)) {
            // found
            suffix_if_found = suffix;
            return cur_location;
          }
          to_check &= ~(1u << cur_location);
        }
      }
      else {
        // if length == 1, match means match
        if (to_check != 0) {
          // found
          return __ffs(to_check) - 1;
        }
      }
      // done searching this node, move on to next
      if (!node.has_next()) { break; }
      auto next_index = node.get_next_index();
      node = node_type(next_index, tile, allocator);
      node.template load_from_allocator<concurrent>();
    }
    // not found until the end
    return -1;
  }

  template <bool use_hash_tag, typename tile_type>
  DEVICE_QUALIFIER int coop_traverse_until_found_merge(hashtable_node<tile_type, device_allocator_context_type>& node,
                                                       const key_slice_type& first_slice,
                                                       bool more_key,
                                                       const key_slice_type* key,
                                                       const size_type& key_length,
                                                       suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator,
                                                       device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    bool current_node_store_deferred = false;
    while (true) {
      uint32_t to_check = node.match_key_in_node(first_slice, more_key);
      if (more_key) {
        // if length > 1, compare suffixes
        while (to_check != 0) {
          auto cur_location = __ffs(to_check) - 1;
          auto suffix_index = node.get_value_from_location(cur_location);
          auto suffix = suffix_type(suffix_index, tile, allocator);
          suffix.load_head();
          static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1;
          if (suffix.streq(key + suffix_offset, key_length - suffix_offset)) {
            // found
            suffix_if_found = suffix;
            // current_node_store_deferred: USER SHOULD STORE the node returned
            return cur_location;
          }
          to_check &= ~(1u << cur_location);
        }
      }
      else {
        // if length == 1, match means match
        if (to_check != 0) {
          // found
          return __ffs(to_check) - 1;
          // current_node_store_deferred: USER SHOULD STORE the node returned
        }
      }
      if (current_node_store_deferred) {
        node.template store_head_to_array_aux_to_allocator<false>(d_table_);  // future unlock will do memory_order_release
        current_node_store_deferred = false;
      }
      // done searching this node, move on to next
      if (!node.has_next()) { break; }
      auto next_index = node.get_next_index();
      auto next_node = node_type(next_index, tile, allocator);
      next_node.template load_from_allocator<false>();  // first load after lock did memory_order_acquire
      if (node.is_mergeable(next_node)) {
        node.merge(next_node);
        current_node_store_deferred = true;
        reclaimer.retire(next_index, tile);
      }
      else {
        node = next_node;
      }
    }
    // not found until the end
    return -1;
  }

 public:
  // device-side debug functions
  template <typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_nodes(Func& task, const tile_type& tile) {
    // debug-purpose, so inefficient implementation
    // called with single warp
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    device_allocator_context_type allocator{allocator_, tile};
    for (size_type bucket_index = 0; bucket_index < num_buckets_; bucket_index++) {
      auto node = node_type(bucket_index, tile, allocator);
      node.template load_from_array<false>(d_table_);
      task.exec(node, bucket_index, tile, allocator);
      while (node.has_next()) {
        auto next_index = node.get_next_index();
        node = node_type(next_index, tile, allocator);
        node.template load_from_allocator<false>();
        task.exec(node, -1, tile, allocator);
      }
    }
  }

  template <typename func>
  void traverse_nodes(func task) {
    static constexpr auto block_size = use_subwarp ? 16 : 32;
    kernels::GpuHashtable::traverse_nodes_kernel<block_size><<<1, block_size>>>(*this, task);
    cudaDeviceSynchronize();
  }

  struct print_nodes_task {
    template <typename tile_type>
    DEVICE_QUALIFIER void init(const tile_type& tile) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int head_index, const tile_type& tile, device_allocator_context_type& allocator) {
      if (head_index >= 0 && tile.thread_rank() == 0) printf("HEAD[%d] ", head_index);
      node.print(allocator);
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void fini(const tile_type& tile) {}
  };
  void print() {
    print_nodes_task task;
    traverse_nodes(task);
  }

  struct validate_nodes_task {
    template <typename tile_type>
    DEVICE_QUALIFIER void init(const tile_type& tile) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int head_index, const tile_type& tile, device_allocator_context_type& allocator) {
      if (head_index >= 0) {
        // wrap up previous num_entry count and update stats
        max_entries_per_bucket_ = max(max_entries_per_bucket_, this_bucket_num_entries_);
        max_nodes_per_bucket_ = max(max_nodes_per_bucket_, this_bucket_num_nodes_);
        this_bucket_num_entries_ = 0;
        this_bucket_num_nodes_ = 0;
      }
      uint16_t num_keys = node.num_keys();
      for (uint16_t i = 0; i < num_keys; i++) {
        bool suffix_bit = node.get_suffix_of_location(i);
        if (suffix_bit) {
          auto suffix_index = node.get_value_from_location(i);
          auto suffix = suffix_node<tile_type, device_allocator_context_type>(suffix_index, tile, allocator);
          suffix.load_head();
          num_suffix_nodes_ += suffix.get_num_nodes();
        }
      }
      this_bucket_num_entries_ += num_keys;
      num_entries_ += num_keys;
      this_bucket_num_nodes_++;
      if (head_index >= 0) { num_head_nodes_++; }
      else { num_aux_nodes_++; }
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void fini(const tile_type& tile) {
      max_entries_per_bucket_ = max(max_entries_per_bucket_, this_bucket_num_entries_);
      max_nodes_per_bucket_ = max(max_nodes_per_bucket_, this_bucket_num_nodes_);
      float avg_entries_per_bucket = float(num_entries_) / num_head_nodes_;
      float avg_nodes_per_bucket = float(num_head_nodes_ + num_aux_nodes_) / num_head_nodes_;
      float fill_factor = float(num_entries_) / (float(num_head_nodes_ + num_aux_nodes_) * 15.0f);
      uint64_t total_bytes_used = (num_head_nodes_ + num_aux_nodes_ + num_suffix_nodes_) * bucket_bytes;
      float bytes_per_entry = static_cast<float>(total_bytes_used) / num_entries_;
      if (tile.thread_rank() == 0) {
        printf("%lu entries (per-bucket max %lu, avg %f), %lu heads + %lu aux nodes (+%lu suffix nodes) (per-bucket max %lu, avg %f); fillfactor %f\n",
          num_entries_, max_entries_per_bucket_, avg_entries_per_bucket,
          num_head_nodes_, num_aux_nodes_, num_suffix_nodes_, max_nodes_per_bucket_, avg_nodes_per_bucket,
          fill_factor);
        printf("Total Space Consumption: %lu B (%f B/entry)\n", total_bytes_used, bytes_per_entry);
      }
    }
    uint64_t num_head_nodes_ = 0, num_aux_nodes_ = 0, num_suffix_nodes_ = 0;
    uint64_t this_bucket_num_entries_ = 0, max_entries_per_bucket_ = 0, num_entries_ = 0;
    uint64_t this_bucket_num_nodes_ = 0, max_nodes_per_bucket_ = 0;
  };
  void validate() {
    validate_nodes_task task;
    traverse_nodes(task);
  }

 private:
  template <typename tile_type>
  DEVICE_QUALIFIER void initialize_bucket(size_type bucket_index,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    auto node = node_type(bucket_index, tile, allocator);
    node.initialize_empty(true);
    node.template store_to_array<false>(d_table_);
  }

  void allocate() {
    is_owner_ = true;
    cuda_try(cudaMalloc(&d_table_, bucket_bytes * num_buckets_));
    initialize();
  }

  void deallocate() {
    if (is_owner_) {
      cuda_try(cudaFree(d_table_));
    }
  }

  void initialize() {
    const uint32_t num_blocks = num_buckets_;
    const uint32_t block_size = use_subwarp ? 16 : 32;
    kernels::GpuHashtable::initialize_kernel<block_size><<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
  }

  elem_type* d_table_;
  bool is_owner_;
  size_type num_buckets_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <uint32_t tile_size, typename hashtable>
  friend __global__ void kernels::GpuHashtable::initialize_kernel(hashtable);

  template <bool do_reclaim, bool subwarp, typename device_func, typename index_type>
  friend __global__ void kernels::batch_kernel(index_type index,
                                              const device_func func,
                                              uint32_t num_requests);

};

} // namespace GpuHashtable
