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
#include <chainhashtable_node.hpp>
#include <suffix.hpp>
#include <queue>
#include <sstream>
#include <type_traits>

#include <dynamic_stack.hpp>
#include <simple_bump_alloc.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_dummy_reclaim.hpp>
#include <simple_debra_reclaim.hpp>

namespace GpuChainHashtable {

template <typename Allocator,
          typename Reclaimer>
struct gpu_chainhashtable {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using table_ptr_type = uint64_t;
  static auto constexpr bucket_size = 32;
  static std::size_t constexpr bucket_bytes = sizeof(elem_type) * bucket_size;
  static auto constexpr cg_tile_size = 32;
  using hashtable_type = gpu_chainhashtable<Allocator, Reclaimer>;

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
  void find(const key_slice_type* keys,
            const size_type max_key_length,
            const size_type* key_lengths,
            value_type* values,
            const size_type num_keys,
            cudaStream_t stream = 0,
            bool concurrent = false) {
    using find_concurrent = kernel::GpuChainHashtable::find_device_func<true, key_slice_type, size_type, value_type>;
    using find_readonly = kernel::GpuChainHashtable::find_device_func<false, key_slice_type, size_type, value_type>;
    #define find_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values
    if (concurrent) {
      find_concurrent func{find_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      find_readonly func{find_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    #undef find_args
  }

  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0,
              bool update_if_exists = false) {
    using insert_func = kernel::GpuChainHashtable::insert_device_func<key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists
    insert_func func{insert_args};
    launch_batch_kernel(func, num_keys, stream);
    #undef insert_args
  }

  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0,
             bool do_merge = true) {
    using erase_readonly = kernel::GpuChainHashtable::erase_device_func<false, key_slice_type, size_type, value_type>;
    using erase_merge = kernel::GpuChainHashtable::erase_device_func<true, key_slice_type, size_type, value_type>;
    #define erase_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths
    if (do_merge) {
      erase_merge func{erase_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      erase_readonly func{erase_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    #undef erase_args
  }

  void test_concurrent_insert_erase(const key_slice_type* insert_keys,
                                    const size_type* insert_key_lengths,
                                    const value_type* insert_values,
                                    const size_type insert_num_keys,
                                    const key_slice_type* erase_keys,
                                    const size_type* erase_key_lengths,
                                    const size_type erase_num_keys,
                                    const size_type max_key_length,
                                    cudaStream_t stream = 0,
                                    bool insert_update_if_exists = false,
                                    bool erase_do_merge = true) {
    using insert_func = kernel::GpuChainHashtable::insert_device_func<key_slice_type, size_type, value_type>;
    using erase_readonly = kernel::GpuChainHashtable::erase_device_func<false, key_slice_type, size_type, value_type>;
    using erase_merge = kernel::GpuChainHashtable::erase_device_func<true, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = insert_keys, .max_key_length = max_key_length, .d_key_lengths = insert_key_lengths, .d_values = insert_values, .update_if_exists = insert_update_if_exists
    #define erase_args .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths
    insert_func ins_func{insert_args};
    if (erase_do_merge) {
      erase_merge erase_func{erase_args};
      launch_batch_concurrent_two_funcs_kernel(ins_func, insert_num_keys, erase_func, erase_num_keys, stream);
    }
    else {
      erase_readonly erase_func{erase_args};
      launch_batch_concurrent_two_funcs_kernel(ins_func, insert_num_keys, erase_func, erase_num_keys, stream);
    }
    #undef insert_args
    #undef erase_args
  }

  // device-side APIs
  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator) {
    using node_type = chainhashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = compute_hash(key, key_length, tile) % num_buckets_;
    elem_type* bucket_ptr = d_table_ + (bucket_size * hash);
    const key_slice_type first_slice = key[0];
    const bool more_key = (key_length > 1);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    auto target_node = coop_traverse_until_found<concurrent>(
        first_slice, more_key, key, key_length, bucket_ptr, location_if_found, suffix_if_found, tile, allocator);
    if (location_if_found >= 0) { // found
      if (more_key) {
        return suffix_if_found.get_value();
      }
      else {
        return target_node.get_value_from_location(location_if_found);
      }
    }
    // not found
    return invalid_value;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           bool update_if_exists = false) {
    using node_type = chainhashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = compute_hash(key, key_length, tile) % num_buckets_;
    elem_type* bucket_ptr = d_table_ + (bucket_size * hash);
    node_type::lock(bucket_ptr, tile);
    const key_slice_type first_slice = key[0];
    const bool more_key = (key_length > 1);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    auto target_node = coop_traverse_until_found<true>(
        first_slice, more_key, key, key_length, bucket_ptr, location_if_found, suffix_if_found, tile, allocator);
    if (location_if_found >= 0) { // already exists
      if (update_if_exists) {
        if (more_key) {
          suffix_if_found.update_value(value);
          suffix_if_found.template store_head<cuda_memory_order::relaxed>();
        }
        else {
          target_node.update(location_if_found, value);
          target_node.template store<cuda_memory_order::relaxed>();
        }
      }
      node_type::unlock(bucket_ptr, tile);
      return update_if_exists;
    }
    // not exists
    value_type to_insert = value;
    if (more_key) {
      to_insert = allocator.allocate(tile);
      auto suffix = suffix_type(
          reinterpret_cast<elem_type*>(allocator.address(to_insert)), to_insert, tile, allocator);
      suffix.template create_from<cuda_memory_order::relaxed>(key + 1, key_length - 1, value);
      suffix.template store_head<cuda_memory_order::relaxed>();
      __threadfence();
    }
    if (target_node.is_full()) {
      auto next_index = allocator.allocate(tile);
      auto new_node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
      new_node.initialize_empty(false);
      new_node.insert(first_slice, to_insert, more_key);
      new_node.template store<cuda_memory_order::relaxed>();
      __threadfence();
      target_node.set_next_index(next_index);
      target_node.set_has_next();
    }
    else { // !node.is_full()
      target_node.insert(first_slice, to_insert, more_key);
    }
    target_node.template store<cuda_memory_order::relaxed>();
    node_type::unlock(bucket_ptr, tile);
    return true;
  }

  template <bool do_merge, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    using node_type = chainhashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = compute_hash(key, key_length, tile) % num_buckets_;
    elem_type* bucket_ptr = d_table_ + (bucket_size * hash);
    node_type::lock(bucket_ptr, tile);
    const key_slice_type first_slice = key[0];
    const bool more_key = (key_length > 1);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    node_type target_node(tile);
    if constexpr (do_merge) {
      target_node = coop_traverse_until_found_merge(
        first_slice, more_key, key, key_length, bucket_ptr, location_if_found, suffix_if_found, tile, allocator, reclaimer);
    }
    else {
      target_node = coop_traverse_until_found<true>(
        first_slice, more_key, key, key_length, bucket_ptr, location_if_found, suffix_if_found, tile, allocator);
    }
    if (location_if_found >= 0) { // exists
      target_node.erase(location_if_found);
      target_node.template store<cuda_memory_order::relaxed>();
      if (more_key) {
        suffix_if_found.template retire<cuda_memory_order::relaxed>(reclaimer);
      }
      node_type::unlock(bucket_ptr, tile);
      return true;
    }
    // not exists
    node_type::unlock(bucket_ptr, tile);
    return false;
  }

 private:
  // device-side helper functions
  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER chainhashtable_node<tile_type> coop_traverse_until_found(const key_slice_type& first_slice,
                                                                            bool more_key,
                                                                            const key_slice_type* key,
                                                                            const size_type& key_length,
                                                                            elem_type* bucket_ptr,
                                                                            int& location_if_found,
                                                                            suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                                            const tile_type& tile,
                                                                            device_allocator_context_type& allocator) {
    using node_type = chainhashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    static constexpr auto memory_order = concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak;
    auto node = node_type(bucket_ptr, tile);
    while (true) {
      node.template load<memory_order>();
      uint32_t to_check = node.match_key_in_node(first_slice, more_key);
      if (more_key) {
        // if length > 1, compare suffixes
        while (to_check != 0) {
          auto cur_location = __ffs(to_check) - 1;
          auto suffix_index = node.get_value_from_location(cur_location);
          auto suffix = suffix_type(
              reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
          suffix.template load_head<memory_order>();
          if (suffix.template streq<memory_order>(key + 1, key_length - 1)) {
            // found
            location_if_found = cur_location;
            suffix_if_found = suffix;
            return node;
          }
          to_check &= ~(1u << cur_location);
        }
      }
      else {
        // if length == 1, match means match
        if (to_check != 0) {
          // found
          location_if_found = __ffs(to_check) - 1;
          return node;
        }
      }
      // done searching this node, move on to next
      if (!node.has_next()) { break; }
      auto next_index = node.get_next_index();
      node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
    }
    // not found until the end
    location_if_found = -1;
    return node;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER chainhashtable_node<tile_type> coop_traverse_until_found_merge(const key_slice_type& first_slice,
                                                                                  bool more_key,
                                                                                  const key_slice_type* key,
                                                                                  const size_type& key_length,
                                                                                  elem_type* bucket_ptr,
                                                                                  int& location_if_found,
                                                                                  suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                                                  const tile_type& tile,
                                                                                  device_allocator_context_type& allocator,
                                                                                  device_reclaimer_context_type& reclaimer) {
    using node_type = chainhashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto node = node_type(bucket_ptr, tile);
    node.template load<cuda_memory_order::relaxed>();
    bool current_node_store_deferred = false;
    while (true) {
      uint32_t to_check = node.match_key_in_node(first_slice, more_key);
      if (more_key) {
        // if length > 1, compare suffixes
        while (to_check != 0) {
          auto cur_location = __ffs(to_check) - 1;
          auto suffix_index = node.get_value_from_location(cur_location);
          auto suffix = suffix_type(
              reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
          suffix.template load_head<cuda_memory_order::relaxed>();
          if (suffix.template streq<cuda_memory_order::relaxed>(key + 1, key_length - 1)) {
            // found
            location_if_found = cur_location;
            suffix_if_found = suffix;
            // current_node_store_deferred: USER SHOULD STORE the node returned
            return node;
          }
          to_check &= ~(1u << cur_location);
        }
      }
      else {
        // if length == 1, match means match
        if (to_check != 0) {
          // found
          location_if_found = __ffs(to_check) - 1;
          // current_node_store_deferred: USER SHOULD STORE the node returned
          return node;
        }
      }
      if (current_node_store_deferred) {
        node.template store<cuda_memory_order::relaxed>();
        current_node_store_deferred = false;
      }
      // done searching this node, move on to next
      if (!node.has_next()) { break; }
      auto next_index = node.get_next_index();
      auto next_node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
      next_node.template load<cuda_memory_order::relaxed>();
      if (node.is_mergeable(next_node)) {
        node.merge(next_node);
        current_node_store_deferred = true;
        __threadfence();
        reclaimer.retire(next_index, tile);
      }
      else {
        node = next_node;
      }
    }
    // not found until the end
    location_if_found = -1;
    return node;
  }

  __host__ __device__ static constexpr uint32_t _constexpr_pow(uint32_t base, uint32_t exp) {
    return (exp == 0) ? 1 : base * _constexpr_pow(base, exp - 1);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER uint32_t compute_hash(const key_slice_type* key,
                                         size_type key_length,
                                         const tile_type& tile) {
    // parallel polynomial rolling hash
    static constexpr uint32_t prime = 0x9e3779b1;
    static constexpr uint32_t prime_multiplier = _constexpr_pow(prime, cg_tile_size);
    // 1. exponent = [1, p, p^2, p^3, ..., p^31]; parallel prefix product
    uint32_t exponent = (tile.thread_rank() == 0) ? 1 : prime;
    for (uint32_t offset = 1; offset < cg_tile_size; offset <<= 1) {
      exponent *= tile.shfl_up(exponent, offset);
    }
    // 2. compute per-lane value
    uint32_t value = 0;
    while (true) {
      if (tile.thread_rank() < key_length) {
        value += key[tile.thread_rank()] * exponent;
      }
      if (key_length <= cg_tile_size) { break; }
      key += cg_tile_size;
      key_length -= cg_tile_size;
      exponent *= prime_multiplier;
    }
    // 3. reduce sum
    for (uint32_t offset = (cg_tile_size / 2); offset != 0; offset >>= 1) {
      value += tile.shfl_down(value, offset);
    }
    value = tile.shfl(value, 0);
    // 4. finalize
    value ^= (key_length * prime); // embed length
    value ^= value >> 16; // murmur3 finalizer
    value *= 0x85ebca6b;
    value ^= value >> 13;
    value *= 0xc2b2ae35;
    value ^= value >> 16;
    return value;
  }

 public:
  // device-side debug functions
  template <typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_nodes(Func& task, const tile_type& tile) {
    // debug-purpose, so inefficient implementation
    // called with single warp
    using node_type = chainhashtable_node<tile_type>;
    device_allocator_context_type allocator{allocator_, tile};
    for (size_type bucket_index = 0; bucket_index < num_buckets_; bucket_index++) {
      auto node = node_type(d_table_ + (bucket_size * bucket_index), tile);
      node.template load<cuda_memory_order::weak>();
      task.exec(node, bucket_index, tile, allocator);
      while (node.has_next()) {
        auto next_index = node.get_next_index();
        node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
        node.template load<cuda_memory_order::weak>();
        task.exec(node, -1, tile, allocator);
      }
    }
  }

  template <typename func>
  void traverse_nodes() {
    kernel::GpuChainHashtable::traverse_nodes_kernel<func><<<1, 32>>>(*this);
    cudaDeviceSynchronize();
  }

  struct print_nodes_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int head_index, const tile_type& tile, device_allocator_context_type& allocator) {
      if (head_index >= 0 && tile.thread_rank() == 0) printf("HEAD[%d] ", head_index);
      node.print(allocator);
    }
    DEVICE_QUALIFIER void fini() {}
  };
  void print() {
    traverse_nodes<print_nodes_task>();
  }

  struct validate_nodes_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {
      lead_lane_ = lead_lane;
      num_head_nodes_ = 0;
      num_aux_nodes_ = 0;
      num_suffix_nodes_ = 0;
      this_bucket_num_entries_ = 0;
      max_entries_per_bucket_ = 0;
      num_entries_ = 0;
      this_bucket_num_nodes_ = 0;
      max_nodes_per_bucket_ = 0;
    }
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
          auto suffix = suffix_node<tile_type, device_allocator_context_type>(
              reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
          suffix.template load_head<cuda_memory_order::weak>();
          num_suffix_nodes_ += suffix.get_num_nodes();
        }
      }
      this_bucket_num_entries_ += num_keys;
      num_entries_ += num_keys;
      this_bucket_num_nodes_++;
      if (head_index >= 0) { num_head_nodes_++; }
      else { num_aux_nodes_++; }
    }
    DEVICE_QUALIFIER void fini() {
      max_entries_per_bucket_ = max(max_entries_per_bucket_, this_bucket_num_entries_);
      max_nodes_per_bucket_ = max(max_nodes_per_bucket_, this_bucket_num_nodes_);
      float avg_entries_per_bucket = float(num_entries_) / num_head_nodes_;
      float avg_nodes_per_bucket = float(num_head_nodes_ + num_aux_nodes_) / num_head_nodes_;
      if (lead_lane_) {
        printf("%lu heads, %lu auxiliary nodes (+%lu suffix nodes) found; per-bucket nodes(max %lu, avg %f), entries(max %lu, avg %f)\n",
          num_head_nodes_, num_aux_nodes_, num_suffix_nodes_, max_nodes_per_bucket_, avg_nodes_per_bucket, max_entries_per_bucket_, avg_entries_per_bucket);
      }
    }
    bool lead_lane_;
    uint64_t num_head_nodes_, num_aux_nodes_, num_suffix_nodes_;
    uint64_t this_bucket_num_entries_, max_entries_per_bucket_, num_entries_;
    uint64_t this_bucket_num_nodes_, max_nodes_per_bucket_;
  };
  void validate() {
    traverse_nodes<validate_nodes_task>();
  }

 private:
  template <typename tile_type>
  DEVICE_QUALIFIER void initialize_bucket(size_type bucket_index,
                                          const tile_type& tile) {
    using node_type = chainhashtable_node<tile_type>;
    auto node = node_type(d_table_ + (bucket_index * bucket_size), tile);
    node.initialize_empty(true);
    node.template store<cuda_memory_order::weak>();
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
    const uint32_t block_size = cg_tile_size;
    kernel::GpuChainHashtable::initialize_kernel<<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
  }

  template <typename device_func>
  void launch_batch_kernel(const device_func& func, uint32_t num_requests, cudaStream_t stream) {
    static constexpr bool do_reclaim = device_func::reclaim_required;
    int block_size = host_reclaimer_type::block_size_;
    std::size_t shmem_size = sizeof(uint32_t) * device_reclaimer_context_type::required_shmem_size();
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      kernel::batch_kernel<do_reclaim, device_func, hashtable_type>,
      block_size,
      shmem_size);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    uint32_t num_blocks = num_blocks_per_sm * device_prop.multiProcessorCount;

    kernel::batch_kernel<do_reclaim><<<num_blocks, block_size, shmem_size, stream>>>(
        *this, func, num_requests);
  }

  template <typename device_func0, typename device_func1>
  void launch_batch_concurrent_two_funcs_kernel(const device_func0& func0, uint32_t num_requests0, const device_func1& func1, uint32_t num_requests1, cudaStream_t stream) {
    static constexpr bool do_reclaim = device_func0::reclaim_required || device_func1::reclaim_required;
    int block_size = host_reclaimer_type::block_size_;
    std::size_t shmem_size = sizeof(uint32_t) * device_reclaimer_context_type::required_shmem_size();
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      kernel::batch_concurrent_two_funcs_kernel<do_reclaim, device_func0, device_func1, hashtable_type>,
      block_size,
      shmem_size);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    uint32_t num_blocks = num_blocks_per_sm * device_prop.multiProcessorCount;
    
    kernel::batch_concurrent_two_funcs_kernel<do_reclaim><<<num_blocks, block_size, shmem_size, stream>>>(
        *this, func0, num_requests0, func1, num_requests1);
  }

  elem_type* d_table_;
  bool is_owner_;
  size_type num_buckets_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <typename chainhashtable>
  friend __global__ void kernel::GpuChainHashtable::initialize_kernel(chainhashtable);

  template <bool do_reclaim, typename device_func, typename index_type>
  friend __global__ void kernel::batch_kernel(index_type index,
                                              const device_func func,
                                              uint32_t num_requests);

  template <bool do_reclaim, typename device_func0, typename device_func1, typename index_type>
  friend __global__ void kernel::batch_concurrent_two_funcs_kernel(index_type tree,
                                                                   const device_func0 func0,
                                                                   uint32_t num_requests0,
                                                                   const device_func1 func1,
                                                                   uint32_t num_requests1);

};

} // namespace GpuChainHashtable
