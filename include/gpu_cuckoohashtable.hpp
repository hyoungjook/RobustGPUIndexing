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
  static auto constexpr default_max_fill_factor = 0.9f;
  static uint32_t constexpr version_counter_size = 8192;

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
    if (fill_factor > default_max_fill_factor) { fill_factor = default_max_fill_factor; }
    auto num_total_buckets = std::max(static_cast<std::size_t>(static_cast<double>(num_elements) / fill_factor / 15), 1UL);
    num_buckets_per_hf_ = (num_total_buckets + num_hfs - 1) / num_hfs;
    allocate();
  }

  gpu_cuckoohashtable& operator=(const gpu_cuckoohashtable& other) = delete;
  gpu_cuckoohashtable(const gpu_cuckoohashtable& other)
      : d_table_(other.d_table_)
      , d_versions_(other.d_versions_)
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
             [[maybe_unused]] bool do_merge_unused = true,
             bool use_hash_for_longkey = true) {
    using erase_hash4long = kernel::GpuHashtable::erase_device_func<false, true, key_slice_type, size_type, value_type>;
    using erase_prfx4long = kernel::GpuHashtable::erase_device_func<false, false, key_slice_type, size_type, value_type>;
    #define erase_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths
    if (use_hash_for_longkey) {
      erase_hash4long func{erase_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      erase_prfx4long func{erase_args};
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
                                    [[maybe_unused]] bool erase_do_merge = true,
                                    bool use_hash_for_longkey = true) {
    using insert_hash4long = kernel::GpuHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
    using erase_hash4long = kernel::GpuHashtable::erase_device_func<false, true, key_slice_type, size_type, value_type>;
    using erase_prfx4long = kernel::GpuHashtable::erase_device_func<false, false, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = insert_keys, .max_key_length = max_key_length, .d_key_lengths = insert_key_lengths, .d_values = insert_values, .update_if_exists = insert_update_if_exists
    #define erase_args .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths
    if (use_hash_for_longkey) {
      insert_hash4long insert_func{insert_args};
      erase_hash4long erase_func{erase_args};
      launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
    }
    else {
      insert_prfx4long insert_func{insert_args};
      erase_prfx4long erase_func{erase_args};
      launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
    }
    #undef insert_args
    #undef erase_args
  }

  void test_concurrent_insert_find(const key_slice_type* insert_keys,
                                   const size_type* insert_key_lengths,
                                   const value_type* insert_values,
                                   const size_type insert_num_keys,
                                   const key_slice_type* find_keys,
                                   const size_type* find_key_lengths,
                                   value_type* find_values,
                                   const size_type find_num_keys,
                                   const size_type max_key_length,
                                   cudaStream_t stream = 0,
                                   bool insert_update_if_exists = false,
                                   bool use_hash_for_longkey = true) {
    using insert_hash4long = kernel::GpuHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
    using find_concurrent_hash4long = kernel::GpuHashtable::find_device_func<true, true, key_slice_type, size_type, value_type>;
    using find_concurrent_prfx4long = kernel::GpuHashtable::find_device_func<true, false, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = insert_keys, .max_key_length = max_key_length, .d_key_lengths = insert_key_lengths, .d_values = insert_values, .update_if_exists = insert_update_if_exists
    #define find_args .d_keys = find_keys, .max_key_length = max_key_length, .d_key_lengths = find_key_lengths, .d_values = find_values
    if (use_hash_for_longkey) {
      insert_hash4long insert_func{insert_args};
      find_concurrent_hash4long find_func{find_args};
      launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, find_func, find_num_keys, stream);
    }
    else {
      insert_prfx4long insert_func{insert_args};
      find_concurrent_prfx4long find_func{find_args};
      launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, find_func, find_num_keys, stream);
    }
    #undef insert_args
    #undef find_args
  }

  // device-side APIs
  template <bool concurrent, bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = compute_hashx2(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_for_longkey && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i = ((hash.x ^ hash.y) * hash_prime2) % num_hfs; // 2-in-d cuckoo hashing
    auto node0 = node_type(bucket_ptr_of(table_i, hash.x), tile);
    auto node1 = node_type(bucket_ptr_of((table_i + 1) % num_hfs, hash.y), tile);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    size_type version;
    if constexpr (concurrent) {
      version = utils::memory::load<size_type, true>(d_versions_ + ((first_slice * hash_prime2) % version_counter_size));
      __threadfence();
    }
    while (true) {
      #define TRY_GET_KEY_FROM_NODE(node) \
      node.template load<concurrent>(); \
      location_if_found = coop_get_key_location_from_node<concurrent, use_hash_for_longkey>( \
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
      if (location_if_found >= 0) { \
        return more_key ? suffix_if_found.get_value() : node.get_value_from_location(location_if_found); \
      }
      TRY_GET_KEY_FROM_NODE(node0)
      TRY_GET_KEY_FROM_NODE(node1)
      #undef TRY_GET_KEY_FROM_NODE
      if constexpr (concurrent) {
        __threadfence();
        auto new_version = utils::memory::load<size_type, true>(d_versions_ + ((first_slice * hash_prime2) % version_counter_size));
        if (version != new_version || (new_version % 2 != 0)) {
          version = new_version;
          __threadfence();
          continue;
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
    uint32_t table_i = ((hash.x ^ hash.y) * hash_prime2) % num_hfs; // 2-in-d cuckoo hashing
    auto node0 = node_type(bucket_ptr_of(table_i, hash.x), tile);
    auto node1 = node_type(bucket_ptr_of((table_i + 1) % num_hfs, hash.y), tile);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    while (true) {
      // === Phase 1. Check if key or space exists in one of two buckets
      bool try_lock_and_insert = false;
      #define CHECK_KEY_OR_SPACE_EXISTS_IN_NODE(node) \
      node.template load<true>(); \
      location_if_found = coop_get_key_location_from_node<true, use_hash_for_longkey>( \
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
      if (location_if_found >= 0) { \
        if (!update_if_exists) { return false; } \
        try_lock_and_insert = true; \
      } \
      else if (!node.is_full()) { \
        try_lock_and_insert = true; \
      }
      CHECK_KEY_OR_SPACE_EXISTS_IN_NODE(node0)
      CHECK_KEY_OR_SPACE_EXISTS_IN_NODE(node1)
      #undef CHECK_KEY_OR_SPACE_EXISTS_IN_NODE
      // === Phase 2. Lock all nodes, check existence, insert ===
      if (try_lock_and_insert) {
        lock_two_nodes_in_order(node0, node1, tile);
        // Check if exists
        #define CHECK_KEY_EXISTS_IN_NODE(node) \
        node.template load<true>(); \
        location_if_found = coop_get_key_location_from_node<true, use_hash_for_longkey>( \
            node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
        if (location_if_found >= 0) { \
          if (update_if_exists) { \
            if (more_key) { \
              suffix_if_found.update_value(value); \
              suffix_if_found.template store_head<true>(); \
            } \
            else { \
              node.update(location_if_found, value); \
              node.template store<true>(); \
            } \
          } \
          node_type::unlock(node0.get_node_ptr(), tile); \
          node_type::unlock(node1.get_node_ptr(), tile); \
          return update_if_exists; \
        }
        CHECK_KEY_EXISTS_IN_NODE(node0)
        CHECK_KEY_EXISTS_IN_NODE(node1)
        #undef CHECK_KEY_EXISTS_IN_NODE
        // Try insert if not full
        #define TRY_INSERT_TO_NODE_IF_NOT_FULL(node) \
        if (!node.is_full()) { \
          value_type to_insert = value; \
          if (more_key) { \
            to_insert = allocator.allocate(tile); \
            auto suffix = suffix_type( \
                reinterpret_cast<elem_type*>(allocator.address(to_insert)), to_insert, tile, allocator); \
            static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1; \
            suffix.template create_from<true>(key + suffix_offset, key_length - suffix_offset, value); \
            suffix.template store_head<true>(); \
            __threadfence(); \
          } \
          node.insert(first_slice, to_insert, more_key); \
          node.template store<true>(); \
          node_type::unlock(node0.get_node_ptr(), tile); \
          node_type::unlock(node1.get_node_ptr(), tile); \
          return true; \
        }
        TRY_INSERT_TO_NODE_IF_NOT_FULL(node0)
        TRY_INSERT_TO_NODE_IF_NOT_FULL(node1)
        #undef TRY_INSERT_TO_NODE_IF_NOT_FULL
        // All nodes are full.
        node_type::unlock(node0.get_node_ptr(), tile);
        node_type::unlock(node1.get_node_ptr(), tile);
      }
      // === Phase 3. Try make space with cuckoo, BFS depth=1 ===
      bool cuckoo_succeed = false; // if we made the empty slot
      #define TRY_MAKE_SPACE_WITH_CUCKOO_FROM_NODE(node, current_table_i) \
      assert(node.is_full()); \
      for (uint32_t loc = 0; loc < node.capacity; loc++) { \
        auto other_node = coop_get_other_bucket_of_key_in<use_hash_for_longkey>( \
            node, loc, current_table_i, tile, allocator); \
        if (!other_node.is_full()) { \
          /*found the space*/ \
          key_slice_type target_key = node.get_key_from_location(loc); \
          value_type target_value = node.get_value_from_location(loc); \
          bool target_suffix = node.get_suffix_of_location(loc); \
          lock_two_nodes_in_order(node, other_node, tile); \
          node.template load<true>(); \
          other_node.template load<true>(); \
          if (!other_node.is_full()) { \
            /*check the key still exists in node*/ \
            uint32_t to_check = node.match_key_value_in_node(target_key, target_value, target_suffix); \
            if (to_check != 0) { \
              assert(__popc(to_check) == 1); \
              /*move the element*/ \
              other_node.insert(target_key, target_value, target_suffix); \
              node.erase(__ffs(to_check) - 1); \
              if (tile.thread_rank() == 0) { \
                atomicAdd(d_versions_ + ((target_key * hash_prime2) % version_counter_size), 1); \
              } \
              __threadfence(); \
              other_node.template store<true>(); \
              __threadfence(); \
              node.template store<true>(); \
              __threadfence(); \
              if (tile.thread_rank() == 0) { \
                atomicAdd(d_versions_ + ((target_key * hash_prime2) % version_counter_size), 1); \
              } \
              cuckoo_succeed = true; \
            } \
          } \
          node_type::unlock(node.get_node_ptr(), tile); \
          node_type::unlock(other_node.get_node_ptr(), tile); \
          if (cuckoo_succeed) { break; } \
        } \
      }
      TRY_MAKE_SPACE_WITH_CUCKOO_FROM_NODE(node0, table_i)
      if (cuckoo_succeed) { continue; }
      TRY_MAKE_SPACE_WITH_CUCKOO_FROM_NODE(node1, ((table_i + 1) % num_hfs))
      #undef TRY_MAKE_SPACE_WITH_CUCKOO_FROM_NODE
      // Phase 4: Cuckoo failed on depth=1, TODO
      assert(cuckoo_succeed);
    }
  }

  template <bool _, bool use_hash_for_longkey, typename tile_type>
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
    auto node0 = node_type(bucket_ptr_of(table_i, hash.x), tile);
    auto node1 = node_type(bucket_ptr_of((table_i + 1) % num_hfs, hash.y), tile);
    // lock all nodes in order
    lock_two_nodes_in_order(node0, node1, tile);
    // check nodes
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    #define TRY_ERASE_KEY_IN_NODE(node) \
    node.template load<true>(); \
    location_if_found = coop_get_key_location_from_node<true, use_hash_for_longkey>( \
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
    if (location_if_found >= 0) { \
      node.erase(location_if_found); \
      node.template store<true>(); \
      if (more_key) { \
        suffix_if_found.template retire<true>(reclaimer); \
      } \
      node_type::unlock(node0.get_node_ptr(), tile); \
      node_type::unlock(node1.get_node_ptr(), tile); \
      return true; \
    }
    TRY_ERASE_KEY_IN_NODE(node0)
    TRY_ERASE_KEY_IN_NODE(node1)
    #undef TRY_ERASE_KEY_IN_NODE
    // not found
    node_type::unlock(node0.get_node_ptr(), tile);
    node_type::unlock(node1.get_node_ptr(), tile);
    return false;
  }

 private:
  // device-side helper functions
  DEVICE_QUALIFIER elem_type* bucket_ptr_of(uint32_t table_i, size_type bucket_index) {
    return d_table_ + (bucket_size * ((bucket_index % num_buckets_per_hf_) + (table_i * num_buckets_per_hf_)));
  }

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
    uint32_t to_check = node.match_key_in_node(first_slice, more_key);
    if (more_key) {
      // if length > 1, compare suffixes
      while (to_check != 0) {
        auto cur_location = __ffs(to_check) - 1;
        auto suffix_index = node.get_value_from_location(cur_location);
        auto suffix = suffix_type(
            reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
        suffix.template load_head<concurrent>();
        static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
        if (suffix.template streq<concurrent>(key + suffix_offset, key_length - suffix_offset)) {
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

  template <bool use_hash_for_longkey, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER hashtable_node<tile_type> coop_get_other_bucket_of_key_in(hashtable_node<tile_type>& node,
                                                                             uint32_t location,
                                                                             uint32_t current_table_i,
                                                                             const tile_type& tile,
                                                                             allocator_type& allocator) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    // compute hash for the key
    uint2 hash;
    if (node.get_suffix_of_location(location)) {
      auto suffix_index = node.get_value_from_location(location);
      auto suffix = suffix_type(
        reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
      suffix.template load_head<true>();
      hash = compute_hashx2_for_suffix<use_hash_for_longkey, true>(
        suffix, use_hash_for_longkey ? 0 : node.get_key_from_location(location), tile);
    }
    else {
      hash = compute_hashx2_single_slice(node.get_key_from_location(location), tile);
    }
    uint32_t target_table_i = ((hash.x ^ hash.y) * hash_prime2) % num_hfs;
    // two hash buckets for the key are
    //    (target_table_i,                  hash.x % num_buckets_per_hf_) and
    //    ((target_table_i + 1) % num_hfs,  hash.y % num_buckets_per_hf_)
    assert(current_table_i == target_table_i || current_table_i == (target_table_i + 1) % num_hfs);
    if (target_table_i == current_table_i) {
      target_table_i = (target_table_i + 1) % num_hfs;
      hash.x = hash.y;
    }
    auto other_node = node_type(bucket_ptr_of(target_table_i, hash.x), tile);
    other_node.template load<true>();
    return other_node;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void lock_two_nodes_in_order(hashtable_node<tile_type>& node0,
                                                hashtable_node<tile_type>& node1,
                                                const tile_type& tile) {
    // lock node with smaller address first
    using node_type = hashtable_node<tile_type>;
    assert(node0.get_node_ptr() != node1.get_node_ptr());
    if (reinterpret_cast<uint64_t>(node0.get_node_ptr()) <
        reinterpret_cast<uint64_t>(node1.get_node_ptr())) {
      node_type::lock(node0.get_node_ptr(), tile);
      node_type::lock(node1.get_node_ptr(), tile);
    }
    else {
      node_type::lock(node1.get_node_ptr(), tile);
      node_type::lock(node0.get_node_ptr(), tile);
    }
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
      auto up_exponent0 = tile.shfl_up(exponent0, offset);
      auto up_exponent1 = tile.shfl_up(exponent1, offset);
      if (tile.thread_rank() >= offset) {
        exponent0 *= up_exponent0;
        exponent1 *= up_exponent1;
      }
    }
    // 2. compute per-lane value
    const auto original_length = key_length;
    uint32_t hash = 0, hash1 = 0;
    while (true) {
      if (tile.thread_rank() < key_length) {
        auto slice = key[tile.thread_rank()];
        hash += exponent0 * slice;
        hash1 += exponent1 * slice;
      }
      if (key_length <= cg_tile_size) { break; }
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
    hash = ((hash * hash_prime0) + original_length) * hash_prime0;
    hash1 = ((hash1 * hash_prime1) + original_length) * hash_prime1;
    if (tile.thread_rank() == cg_tile_size - 1) { hash = hash1; }
    // 4. finalize
    hash = hash_murmur3_finalizer(hash);
    return make_uint2(tile.shfl(hash, 0), tile.shfl(hash, cg_tile_size - 1));
  }
  template <typename tile_type>
  DEVICE_QUALIFIER uint2 compute_hashx2_single_slice(const key_slice_type& key,
                                                     const tile_type& tile) {
    // if key_length == 1, hash = murmur3(((key * p) + 1) * p)
    uint2 hash = make_uint2(((key * hash_prime0) + 1) * hash_prime0,
                            ((key * hash_prime1) + 1) * hash_prime1);
    if (tile.thread_rank() == 1) { hash.x = hash.y; }
    hash.x = hash_murmur3_finalizer(hash.x);
    return make_uint2(tile.shfl(hash.x, 0), tile.shfl(hash.x, 1));
  }
  template <bool use_hash_for_longkey, bool atomic, typename suffix_type, typename tile_type>
  DEVICE_QUALIFIER uint2 compute_hashx2_for_suffix(const suffix_type& suffix,
                                                   const key_slice_type& first_slice,
                                                   const tile_type& tile) {
    // compute polynomial
    uint2 hash = suffix.template compute_polynomial<hash_prime0, hash_prime1, atomic>();
    if constexpr (!use_hash_for_longkey) {
      hash.x = (hash.x * hash_prime0) + first_slice;
      hash.y = (hash.y * hash_prime1) + first_slice;
    }
    static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
    uint32_t key_length = suffix.get_key_length() + suffix_offset;
    hash.x = ((hash.x * hash_prime0) + key_length) * hash_prime0;
    hash.y = ((hash.y * hash_prime1) + key_length) * hash_prime1;
    // finalize
    if (tile.thread_rank() == 1) { hash.x = hash.y; }
    hash.x = hash_murmur3_finalizer(hash.x);
    return make_uint2(tile.shfl(hash.x, 0), tile.shfl(hash.x, 1));
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
        node.template load<false>();
        task.exec(node, table_index, bucket_index, tile, allocator);
      }
    }
  }

  template <typename func>
  void traverse_nodes(func task) {
    kernel::GpuHashtable::traverse_nodes_kernel<<<1, 32>>>(*this, task);
    cudaDeviceSynchronize();
  }

  struct print_nodes_task {
    template <typename tile_type>
    DEVICE_QUALIFIER void init(const tile_type& tile) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int table_index, int bucket_index, const tile_type& tile, device_allocator_context_type& allocator) {
      if (tile.thread_rank() == 0) printf("TABLE[%d].NODE[%d] ", table_index, bucket_index);
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
    DEVICE_QUALIFIER void init(const tile_type& tile) {
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
          suffix.template load_head<false>();
          num_suffix_nodes_ += suffix.get_num_nodes();
        }
      }
      num_buckets_++;
      num_entries_ += num_keys;
      entries_per_table_[table_index] += num_keys;
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void fini(const tile_type& tile) {
      if (tile.thread_rank() == 0) {
        printf("%lu entries (", num_entries_);
        for (int i = 0; i < num_hfs; i++) {
          printf("%lu ", entries_per_table_[i]);
        }
        printf("), %lu buckets (+%lu suffix nodes)\n", num_buckets_, num_suffix_nodes_);
      }
    }
    uint64_t num_buckets_ = 0, num_suffix_nodes_ = 0;
    uint64_t num_entries_ = 0, entries_per_table_[num_hfs] = {0,};
  };
  void validate() {
    validate_nodes_task task;
    traverse_nodes(task);
  }

 private:
  template <typename tile_type>
  DEVICE_QUALIFIER void initialize_bucket(size_type bucket_index,
                                          const tile_type& tile) {
    using node_type = hashtable_node<tile_type>;
    auto node = node_type(d_table_ + (bucket_index * bucket_size), tile);
    node.initialize_empty(true);
    node.template store<false>();
  }

  void allocate() {
    is_owner_ = true;
    cuda_try(cudaMalloc(&d_table_, bucket_bytes * num_buckets_per_hf_ * num_hfs));
    cuda_try(cudaMalloc(&d_versions_, sizeof(size_type) * version_counter_size));
    initialize();
  }

  void deallocate() {
    if (is_owner_) {
      cuda_try(cudaFree(d_table_));
      cuda_try(cudaFree(d_versions_));
    }
  }

  void initialize() {
    const uint32_t num_blocks = num_buckets_per_hf_ * num_hfs;
    const uint32_t block_size = cg_tile_size;
    kernel::GpuHashtable::initialize_kernel<<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
    cuda_try(cudaMemset(d_versions_, 0, sizeof(size_type) * version_counter_size));
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
  size_type* d_versions_;
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
