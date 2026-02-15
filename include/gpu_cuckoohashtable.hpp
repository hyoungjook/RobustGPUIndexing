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
#include <hashtable_node.hpp>
#include <suffix.hpp>
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
          typename Reclaimer>
struct gpu_cuckoohashtable {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using table_ptr_type = uint64_t;
  static auto constexpr bucket_size = 32;
  static std::size_t constexpr bucket_bytes = sizeof(elem_type) * bucket_size;
  static auto constexpr cg_tile_size = 32;
  using hashtable_type = gpu_cuckoohashtable<Allocator, Reclaimer>;
  static auto constexpr num_hfs = 4;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  using host_allocator_type = Allocator;
  using device_allocator_instance_type = typename host_allocator_type::device_instance_type;
  using device_allocator_context_type = device_allocator_context<host_allocator_type>;

  using host_reclaimer_type = Reclaimer;
  using device_reclaimer_instance_type = typename host_reclaimer_type::device_instance_type;
  using device_reclaimer_context_type = device_reclaimer_context<host_reclaimer_type>;

  gpu_cuckoohashtable() = delete;
  gpu_cuckoohashtable(const host_allocator_type& host_allocator,
           const host_reclaimer_type& host_reclaimer,
           size_type num_buckets_per_hf)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance())
      , num_buckets_per_hf_(num_buckets_per_hf) {
    allocate();
  }
  gpu_cuckoohashtable(const host_allocator_type& host_allocator,
           const host_reclaimer_type& host_reclaimer,
           std::size_t num_elements,
           float fill_factor)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance()) {
    auto num_total_buckets = std::max(static_cast<std::size_t>(static_cast<double>(num_elements) / fill_factor / 15), 1UL);
    num_buckets_per_hf_ = (num_total_buckets + num_hfs - 1) / num_hfs;
    allocate();
  }

  gpu_cuckoohashtable& operator=(const gpu_cuckoohashtable& other) = delete;
  gpu_cuckoohashtable(const gpu_cuckoohashtable& other)
      : d_table_(other.d_table_)
      , is_owner_(false)
      , num_buckets_per_hf_(other.num_buckets_per_hf_)
      , allocator_(other.allocator_)
      , reclaimer_(other.reclaimer_) {}

  ~gpu_cuckoohashtable() {
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
            bool concurrent = false,
            bool use_hash_for_longkey = true) {
    using find_concurrent_hash4long = kernel::GpuHashtable::find_device_func<true, true, key_slice_type, size_type, value_type>;
    using find_concurrent_prfx4long = kernel::GpuHashtable::find_device_func<true, false, key_slice_type, size_type, value_type>;
    using find_readonly_hash4long = kernel::GpuHashtable::find_device_func<false, true, key_slice_type, size_type, value_type>;
    using find_readonly_prfx4long = kernel::GpuHashtable::find_device_func<false, false, key_slice_type, size_type, value_type>;
    #define find_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values
    if (concurrent) {
      if (use_hash_for_longkey) {
        find_concurrent_hash4long func{find_args};
        launch_batch_kernel(func, num_keys, stream);
      }
      else {
        find_concurrent_prfx4long func{find_args};
        launch_batch_kernel(func, num_keys, stream);
      }
    }
    else {
      if (use_hash_for_longkey) {
        find_readonly_hash4long func{find_args};
        launch_batch_kernel(func, num_keys, stream);
      }
      else {
        find_readonly_prfx4long func{find_args};
        launch_batch_kernel(func, num_keys, stream);
      }
    }
    #undef find_args
  }

  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0,
              bool update_if_exists = false,
              bool use_hash_for_longkey = true) {
    using insert_hash4long = kernel::GpuHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists
    if (use_hash_for_longkey) {
      insert_hash4long func{insert_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      insert_prfx4long func{insert_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    #undef insert_args
  }

  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0,
             bool do_merge = true,
             bool use_hash_for_longkey = true) {
    using erase_merge_hash4long = kernel::GpuHashtable::erase_device_func<true, true, key_slice_type, size_type, value_type>;
    using erase_merge_prfx4long = kernel::GpuHashtable::erase_device_func<true, false, key_slice_type, size_type, value_type>;
    using erase_nomerge_hash4long = kernel::GpuHashtable::erase_device_func<false, true, key_slice_type, size_type, value_type>;
    using erase_nomerge_prfx4long = kernel::GpuHashtable::erase_device_func<false, false, key_slice_type, size_type, value_type>;
    #define erase_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths
    if (do_merge) {
      if (use_hash_for_longkey) {
        erase_merge_hash4long func{erase_args};
        launch_batch_kernel(func, num_keys, stream);
      }
      else {
        erase_merge_prfx4long func{erase_args};
        launch_batch_kernel(func, num_keys, stream);
      }
    }
    else {
      if (use_hash_for_longkey) {
        erase_nomerge_hash4long func{erase_args};
        launch_batch_kernel(func, num_keys, stream);
      }
      else {
        erase_nomerge_prfx4long func{erase_args};
        launch_batch_kernel(func, num_keys, stream);
      }
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
                                    bool erase_do_merge = true,
                                    bool use_hash_for_longkey = true) {
    using insert_hash4long = kernel::GpuHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
    using erase_merge_hash4long = kernel::GpuHashtable::erase_device_func<true, true, key_slice_type, size_type, value_type>;
    using erase_merge_prfx4long = kernel::GpuHashtable::erase_device_func<true, false, key_slice_type, size_type, value_type>;
    using erase_nomerge_hash4long = kernel::GpuHashtable::erase_device_func<false, true, key_slice_type, size_type, value_type>;
    using erase_nomerge_prfx4long = kernel::GpuHashtable::erase_device_func<false, false, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = insert_keys, .max_key_length = max_key_length, .d_key_lengths = insert_key_lengths, .d_values = insert_values, .update_if_exists = insert_update_if_exists
    #define erase_args .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths
    if (use_hash_for_longkey) {
      insert_hash4long insert_func{insert_args};
      if (erase_do_merge) {
        erase_merge_hash4long erase_func{erase_args};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
      else {
        erase_nomerge_hash4long erase_func{erase_args};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
    }
    else {
      insert_prfx4long insert_func{insert_args};
      if (erase_do_merge) {
        erase_merge_prfx4long erase_func{erase_args};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
      else {
        erase_nomerge_prfx4long erase_func{erase_args};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
    }
    #undef insert_args
    #undef erase_args
  }

  // device-side APIs
  template <bool concurrent, bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    static constexpr auto memory_order = concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak;
    auto hash = compute_hashx2(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_for_longkey && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i = ((hash.x ^ hash.y) * hash_prime2) % num_hfs; // 2-in-d cuckoo hashing
    node_type nodes[2] = {
      node_type(d_table_ + (bucket_size * ((hash.x % num_buckets_per_hf_) + (table_i * num_buckets_per_hf_))), tile),
      node_type(d_table_ + (bucket_size * ((hash.y % num_buckets_per_hf_) + (((table_i + 1) % num_hfs) * num_buckets_per_hf_))), tile)
    };
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    while (true) {
      // TODO version check
      for (int i = 0; i < 2; i++) {
        nodes[i].template load<memory_order>();
        location_if_found = coop_get_key_location_from_node<concurrent, use_hash_for_longkey>(
            nodes[i], first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
        if (location_if_found >= 0) {
          return more_key ? suffix_if_found.get_value() : nodes[i].get_value_from_location(location_if_found);
        }
      }
      break;
    }
    // not found
    return invalid_value;
  }

  template <bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           bool update_if_exists = false) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = compute_hashx2(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_for_longkey && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i[2]; // 2-in-d cuckoo hashing
    table_i[0] = ((hash.x ^ hash.y) * hash_prime2) % num_hfs;
    table_i[1] = (table_i[0] + 1) % num_hfs;
    node_type nodes[2] = {
      node_type(d_table_ + (bucket_size * ((hash.x % num_buckets_per_hf_) + (table_i[0] * num_buckets_per_hf_))), tile),
      node_type(d_table_ + (bucket_size * ((hash.y % num_buckets_per_hf_) + (table_i[1] * num_buckets_per_hf_))), tile)
    };
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    while (true) {
      // === Phase 1. Lock all nodes, if not exists, insert ===
      if (table_i[0] < 2) {
        node_type::lock(nodes[0].get_node_ptr(), tile);
        node_type::lock(nodes[1].get_node_ptr(), tile);
      }
      else {
        node_type::lock(nodes[1].get_node_ptr(), tile);
        node_type::lock(nodes[0].get_node_ptr(), tile);
      }
      // Check if exists
      for (int i = 0; i < 2; i++) {
        nodes[i].template load<cuda_memory_order::relaxed>();
        location_if_found = coop_get_key_location_from_node<true, use_hash_for_longkey>(
            nodes[i], first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
        if (location_if_found >= 0) {
          if (update_if_exists) {
            if (more_key) {
              suffix_if_found.update_value(value);
              suffix_if_found.template store_head<cuda_memory_order::relaxed>();
            }
            else {
              nodes[i].update(location_if_found, value);
              nodes[i].template store<cuda_memory_order::relaxed>();
            }
          }
          for (int j = 0; j < 2; j++) {
            node_type::unlock(nodes[j].get_node_ptr(), tile);
          }
          return update_if_exists;
        }
      }
      // Try insert if not full
      for (int i = 0; i < 2; i++) {
        if (!nodes[i].is_full()) {
          value_type to_insert = value;
          if (more_key) {
            to_insert = allocator.allocate(tile);
            auto suffix = suffix_type(
                reinterpret_cast<elem_type*>(allocator.address(to_insert)), to_insert, tile, allocator);
            static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
            suffix.template create_from<cuda_memory_order::relaxed>(key + suffix_offset, key_length - suffix_offset, value);
            suffix.template store_head<cuda_memory_order::relaxed>();
            __threadfence(); \
          }
          nodes[i].insert(first_slice, to_insert, more_key);
          nodes[i].template store<cuda_memory_order::relaxed>();
          for (int j = 0; j < 2; j++) {
            node_type::unlock(nodes[j].get_node_ptr(), tile);
          }
          return true;
        }
      }
      // All nodes are full.
      for (int j = 0; j < 2; j++) {
        node_type::unlock(nodes[j].get_node_ptr(), tile);
      }
      // === Phase 2. Try find cuckoo path with BFS, depth=1 ===
      bool continue_to_retry = false; // if we made the empty slot
      for (int i = 0; i < 2; i++) {
        nodes[i].template load<cuda_memory_order::relaxed>();
        if (!nodes[i].is_full()) {
          continue_to_retry = true;
          break;
        }
        for (uint32_t loc = 0; loc < nodes[i].capacity; loc++) {
          // compute hash for suffix key
          uint32_t hash[2];
          if (nodes[i].get_suffix_of_location(loc)) {
            auto suffix_index = nodes[i].get_value_from_location(loc);
            auto suffix = suffix_type(
                reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
            suffix.template load_head<cuda_memory_order::relaxed>();
            compute_hashx2_for_suffix<use_hash_for_longkey, cuda_memory_order::relaxed>(
                suffix, use_hash_for_longkey ? 0 : nodes[i].get_key_from_location(loc), hash, tile);
          }
          else {
            compute_hashx2_single_slice(nodes[i].get_key_from_location(loc), hash, tile);
          }
          uint32_t target_table_i = ((hash[0] ^ hash[1]) * hash_prime2) % num_hfs;
          if (target_table_i == table_i[i]) {
            target_table_i = (target_table_i + 1) % num_hfs;
            hash[0] = hash[1];
          }
          else { assert((target_table_i + 1) % num_hfs == table_i[i]); }
          // check other node
          auto other_node = node_type(d_table_ + (bucket_size * ((hash[0] % num_buckets_per_hf_) + (target_table_i * num_buckets_per_hf_))), tile);
          other_node.template load<cuda_memory_order::relaxed>();
          if (!other_node.is_full()) {
            // found the space
            if (i < target_table_i) {
              node_type::lock(nodes[i].get_node_ptr(), tile);
              node_type::lock(other_node.get_node_ptr(), tile);
            }
            else {
              node_type::lock(other_node.get_node_ptr(), tile);
              node_type::lock(nodes[i].get_node_ptr(), tile);
            }
            nodes[i].template load<cuda_memory_order::relaxed>();
            other_node.template load<cuda_memory_order::relaxed>();
            if (!nodes[i].is_full()) {
              continue_to_retry = true;
            }
            else if (!other_node.is_full()) {
              // locked the non-full node; do the cuckoo
              key_slice_type key = nodes[i].get_key_from_location(loc);
              value_type value = nodes[i].get_value_from_location(loc);
              bool more_key = nodes[i].get_suffix_of_location(loc);
              other_node.insert(key, value, more_key);
              nodes[i].erase(loc);
              other_node.template store<cuda_memory_order::relaxed>();
              __threadfence();
              nodes[i].template store<cuda_memory_order::relaxed>();
              continue_to_retry = true;
            }
            node_type::unlock(nodes[i].get_node_ptr(), tile);
            node_type::unlock(other_node.get_node_ptr(), tile);
            if (continue_to_retry) { break; }
          }
        }
        if (continue_to_retry) { break; }
      }
      if (continue_to_retry) { continue; }
      // Phase 3: Cuckoo failed.. TODO
      assert(false);
    }
  }

  template <bool do_merge, bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = compute_hashx2(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_for_longkey && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i = ((hash.x ^ hash.y) * hash_prime2) % num_hfs; // 2-in-d cuckoo hashing
    node_type nodes[2] = {
      node_type(d_table_ + (bucket_size * ((hash.x % num_buckets_per_hf_) + (table_i * num_buckets_per_hf_))), tile),
      node_type(d_table_ + (bucket_size * ((hash.y % num_buckets_per_hf_) + (((table_i + 1) % num_hfs) * num_buckets_per_hf_))), tile)
    };
    // lock all nodes in order
    // TODO first try without lock
    if (table_i < 2) {
      node_type::lock(nodes[0].get_node_ptr(), tile);
      node_type::lock(nodes[1].get_node_ptr(), tile);
    }
    else {
      node_type::lock(nodes[1].get_node_ptr(), tile);
      node_type::lock(nodes[0].get_node_ptr(), tile);
    }
    // check nodes
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    for (int i = 0; i < 2; i++) {
      nodes[i].template load<cuda_memory_order::relaxed>();
      location_if_found = coop_get_key_location_from_node<true, use_hash_for_longkey>(
          nodes[i], first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
      if (location_if_found >= 0) {
        nodes[i].erase(location_if_found);
        nodes[i].template store<cuda_memory_order::relaxed>();
        if (more_key) {
          suffix_if_found.template retire<cuda_memory_order::relaxed>(reclaimer);
        }
        for (int j = i; j < 2; j++) {
          node_type::unlock(nodes[j].get_node_ptr(), tile);
        }
        return true;
      }
      node_type::unlock(nodes[i].get_node_ptr(), tile);
    }
    // not found
    return false;
  }

 private:
  // device-side helper functions
  template <bool concurrent, bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER int coop_get_key_location_from_node(hashtable_node<tile_type>& node,
                                                       const key_slice_type& first_slice,
                                                       bool more_key,
                                                       const key_slice_type* key,
                                                       const size_type& key_length,
                                                       suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    static constexpr auto memory_order = concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak;
    uint32_t to_check = node.match_key_in_node(first_slice, more_key);
    if (more_key) {
      // if length > 1, compare suffixes
      while (to_check != 0) {
        auto cur_location = __ffs(to_check) - 1;
        auto suffix_index = node.get_value_from_location(cur_location);
        auto suffix = suffix_type(
            reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
        suffix.template load_head<memory_order>();
        static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
        if (suffix.template streq<memory_order>(key + suffix_offset, key_length - suffix_offset)) {
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
        int location = __ffs(to_check) - 1;
        return location;
      }
    }
    // not found
    return -1;
  }

  static constexpr uint32_t hash_prime0 = 0x9e3779b1;
  static constexpr uint32_t hash_prime1 = 0x01000193;
  static constexpr uint32_t hash_prime2 = 0xfffffffb;
  static DEVICE_QUALIFIER uint32_t hash_murmur3_finalizer(uint32_t hash) {
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
  }
  template <typename tile_type>
  DEVICE_QUALIFIER uint2 compute_hashx2(const key_slice_type* key, size_type key_length, const tile_type& tile) {
    static constexpr uint32_t prime0_multiplier = utils::constexpr_pow(hash_prime0, cg_tile_size);
    static constexpr uint32_t prime1_multiplier = utils::constexpr_pow(hash_prime1, cg_tile_size);
    // 1. exponent = [1, p, p^2, ..., p^31]; parallel prefix product
    uint32_t exponent0 = (tile.thread_rank() == 0) ? 1 : hash_prime0;
    uint32_t exponent1 = (tile.thread_rank() == 0) ? 1 : hash_prime1;
    for (uint32_t offset = 1; offset < cg_tile_size; offset <<= 1) {
      exponent0 *= tile.shfl_up(exponent0, offset);
      exponent1 *= tile.shfl_up(exponent1, offset);
    }
    // 2. compute per-lane value
    uint32_t hash = 0, hash1 = 0;
    while (true) {
      if (tile.thread_rank() < key_length) {
        auto slice = key[tile.thread_rank()];
        hash += exponent0 * slice;
        hash1 += exponent1 * slice;
      }
      if (key_length < cg_tile_size) { break; }
      key += cg_tile_size;
      key_length -= cg_tile_size;
      exponent0 *= prime0_multiplier;
      exponent1 *= prime1_multiplier;
    }
    // 3. reduce sum
    for (uint32_t offset = (cg_tile_size / 2); offset != 0; offset >>= 1) {
      hash += tile.shfl_down(hash, offset);
      hash1 += tile.shfl_up(hash1, offset);
    }
    hash = ((hash * hash_prime0) + key_length) * hash_prime0;
    hash1 = ((hash1 * hash_prime1) + key_length) * hash_prime1;
    if (tile.thread_rank() == cg_tile_size - 1) { hash = hash1; }
    // 4. finalize
    hash = hash_murmur3_finalizer(hash);
    return make_uint2(tile.shfl(hash, 0), tile.shfl(hash, cg_tile_size - 1));
  }
  struct single_slice_hasher {
    DEVICE_QUALIFIER uint32_t operator()(uint32_t key) {
      // if length == 1, hash = p * (1 + p * key[0])
      uint32_t hash = prime_ * (1 + prime_ * key);
      return hash_murmur3_finalizer(hash);
    }
    uint32_t prime_;
  };
  template <typename tile_type>
  DEVICE_QUALIFIER void compute_hashx2_single_slice(const key_slice_type& key,
                                                    uint32_t out_hash[2],
                                                    const tile_type& tile) {
    // if key_length == 1, hash = murmur3(((key * p) + 1) * p)
    out_hash[0] = ((key * hash_prime0) + 1) * hash_prime0;
    out_hash[1] = ((key * hash_prime1) + 1) * hash_prime1;
    if (tile.thread_rank() == 1) { out_hash[0] = out_hash[1]; }
    out_hash[0] = hash_murmur3_finalizer(out_hash[0]);
    out_hash[1] = tile.shfl(out_hash[0], 1);
    out_hash[0] = tile.shfl(out_hash[0], 0);
  }
  template <bool use_hash_for_longkey, cuda_memory_order order, typename suffix_type, typename tile_type>
  DEVICE_QUALIFIER void compute_hashx2_for_suffix(const suffix_type& suffix,
                                                  const key_slice_type& first_slice,
                                                  uint32_t out_hash[2],
                                                  const tile_type& tile) {
    // compute polynomial
    auto polynomial = suffix.template compute_polynomial<hash_prime0, hash_prime1, order>();
    out_hash[0] = polynomial.x;
    out_hash[1] = polynomial.y;
    if constexpr (!use_hash_for_longkey) {
      out_hash[0] = (out_hash[0] * hash_prime0) + first_slice;
      out_hash[1] = (out_hash[1] * hash_prime1) + first_slice;
    }
    static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
    uint32_t key_length = suffix.get_key_length() + suffix_offset;
    out_hash[0] = ((out_hash[0] * hash_prime0) + key_length) * hash_prime0;
    out_hash[1] = ((out_hash[1] * hash_prime1) + key_length) * hash_prime1;
    // finalize
    if (tile.thread_rank() == 1) { out_hash[0] = out_hash[1]; }
    out_hash[0] = hash_murmur3_finalizer(out_hash[0]);
    out_hash[1] = tile.shfl(out_hash[0], 1);
    out_hash[0] = tile.shfl(out_hash[0], 0);
  }

 public:
  // device-side debug functions
  template <typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_nodes(Func& task, const tile_type& tile) {
    // debug-purpose, so inefficient implementation
    // called with single warp
    using node_type = hashtable_node<tile_type>;
    device_allocator_context_type allocator{allocator_, tile};
    for (int table_index = 0; table_index < num_hfs; table_index++) {
      for (size_type bucket_index = 0; bucket_index < num_buckets_per_hf_; bucket_index++) {
        size_type global_bucket_index = num_buckets_per_hf_ * table_index + bucket_index;
        auto node = node_type(d_table_ + (bucket_size * global_bucket_index), tile);
        node.template load<cuda_memory_order::weak>();
        task.exec(node, table_index, bucket_index, tile, allocator);
      }
    }
  }

  template <typename func>
  void traverse_nodes() {
    kernel::GpuHashtable::traverse_nodes_kernel<func><<<1, 32>>>(*this);
    cudaDeviceSynchronize();
  }

  struct print_nodes_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int table_index, int bucket_index, const tile_type& tile, device_allocator_context_type& allocator) {
      if (tile.thread_rank() == 0) printf("TABLE[%d].NODE[%d] ", table_index, bucket_index);
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
      num_buckets_ = 0;
      num_suffix_nodes_ = 0;
      for (int i = 0; i < num_hfs; i++) {
        entries_per_table_[i] = 0;
      }
    }
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int table_index, int bucket_index, const tile_type& tile, device_allocator_context_type& allocator) {
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
      num_buckets_++;
      entries_per_table_[table_index] += num_keys;
    }
    DEVICE_QUALIFIER void fini() {
      if (lead_lane_) {
        printf("%lu buckets, %lu suffix nodes; entries(", num_buckets_, num_suffix_nodes_);
        for (int i = 0; i < num_hfs; i++) {
          printf("%lu ", entries_per_table_[i]);
        }
        printf(")\n");
      }
    }
    bool lead_lane_;
    uint64_t num_buckets_, num_suffix_nodes_;
    uint64_t entries_per_table_[num_hfs];
  };
  void validate() {
    traverse_nodes<validate_nodes_task>();
  }

 private:
  template <typename tile_type>
  DEVICE_QUALIFIER void initialize_bucket(size_type bucket_index,
                                          const tile_type& tile) {
    using node_type = hashtable_node<tile_type>;
    auto node = node_type(d_table_ + (bucket_index * bucket_size), tile);
    node.initialize_empty(true);
    node.template store<cuda_memory_order::weak>();
  }

  void allocate() {
    is_owner_ = true;
    cuda_try(cudaMalloc(&d_table_, bucket_bytes * num_buckets_per_hf_ * num_hfs));
    initialize();
  }

  void deallocate() {
    if (is_owner_) {
      cuda_try(cudaFree(d_table_));
    }
  }

  void initialize() {
    const uint32_t num_blocks = num_buckets_per_hf_ * num_hfs;
    const uint32_t block_size = cg_tile_size;
    kernel::GpuHashtable::initialize_kernel<<<num_blocks, block_size>>>(*this);
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
  size_type num_buckets_per_hf_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <typename hashtable>
  friend __global__ void kernel::GpuHashtable::initialize_kernel(hashtable);

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

} // namespace GpuHashtable
