/*
 *   Copyright 2026 Hyoungjoo Kim, Carnegie Mellon University
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
          uint32_t tile_size = 16>
struct gpu_cuckoohashtable {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using table_ptr_type = uint64_t;
  static constexpr uint32_t tile_size_ = tile_size;
  static_assert(tile_size == 32 || tile_size == 16);
  static auto constexpr bucket_size = 32;
  static std::size_t constexpr bucket_bytes = sizeof(elem_type) * bucket_size;
  static auto constexpr num_hfs = 4;
  static auto constexpr max_fill_factor = 0.9f;
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
    if (fill_factor > max_fill_factor) {
      fprintf(stderr, "Fill factor %f is too large for GPUCuckooHT. Max is %f\n", fill_factor, max_fill_factor);
      exit(1);
    }
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
  template <bool concurrent = false,
            bool use_hash_tag = true>
  void find(const key_slice_type* keys,
            const size_type max_key_length,
            const size_type* key_lengths,
            value_type* values,
            const size_type num_keys,
            cudaStream_t stream = 0) {
    kernels::GpuHashtable::find_device_func<gpu_cuckoohashtable, concurrent, use_hash_tag>
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
    kernels::GpuHashtable::insert_device_func<gpu_cuckoohashtable, use_hash_tag>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true, bool _ = true>
  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0) {
    kernels::GpuHashtable::erase_device_func<gpu_cuckoohashtable, use_hash_tag, true>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true, bool _ = true>
  void mixed_batch(const kernels::request_type* request_types,
                   const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   value_type* values,
                   bool* results,
                   const size_type num_requests,
                   cudaStream_t stream = 0,
                   bool insert_update_if_exists = false) {
    kernels::GpuHashtable::mixed_device_func<gpu_cuckoohashtable, use_hash_tag, true>
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
    auto hash = utils::compute_hashx2<utils::PRIME0, utils::PRIME1>(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_tag && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i = ((hash.x ^ hash.y) * utils::PRIME2) % num_hfs; // 2-in-d cuckoo hashing
    auto node0 = node_type(bucket_index_of(table_i, hash.x), tile, allocator);
    auto node1 = node_type(bucket_index_of((table_i + 1) % num_hfs, hash.y), tile, allocator);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    size_type version;
    if constexpr (concurrent) {
      version = utils::memory::load<size_type, true, true>(d_versions_ + ((first_slice * utils::PRIME2) % version_counter_size));
    }
    while (true) {
      #define TRY_GET_KEY_FROM_NODE(node) \
      node.template load_from_array<concurrent>(d_table_); \
      location_if_found = coop_get_key_location_from_node<use_hash_tag>( \
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
      if (location_if_found >= 0) { \
        return more_key ? suffix_if_found.get_value() : node.get_value_from_location(location_if_found); \
      }
      TRY_GET_KEY_FROM_NODE(node0)
      TRY_GET_KEY_FROM_NODE(node1)
      #undef TRY_GET_KEY_FROM_NODE
      if constexpr (concurrent) {
        auto new_version = utils::memory::load<size_type, true, true>(d_versions_ + ((first_slice * utils::PRIME2) % version_counter_size));
        if (version != new_version || (new_version % 2 != 0)) {
          version = new_version;
          continue;
        }
      }
      break;
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
    auto hash = utils::compute_hashx2<utils::PRIME0, utils::PRIME1>(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_tag && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i = ((hash.x ^ hash.y) * utils::PRIME2) % num_hfs; // 2-in-d cuckoo hashing
    auto node0 = node_type(bucket_index_of(table_i, hash.x), tile, allocator);
    auto node1 = node_type(bucket_index_of((table_i + 1) % num_hfs, hash.y), tile, allocator);
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    while (true) {
      // === Phase 1. Check if key or space exists in one of two buckets
      bool try_lock_and_insert = false;
      #define CHECK_KEY_OR_SPACE_EXISTS_IN_NODE(node) \
      node.template load_from_array<true>(d_table_); \
      location_if_found = coop_get_key_location_from_node<use_hash_tag>( \
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
        node.template load_from_array<true>(d_table_); \
        location_if_found = coop_get_key_location_from_node<use_hash_tag>( \
            node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
        if (location_if_found >= 0) { \
          if (update_if_exists) { \
            if (more_key) { \
              suffix_if_found.update_value(value); \
              suffix_if_found.store_head(); \
            } \
            else { \
              node.update(location_if_found, value); \
              node.template store_to_array<false>(d_table_); \
            } \
          } \
          node_type::unlock<false>(d_table_, node0.get_node_index(), tile); \
          node_type::unlock(d_table_, node1.get_node_index(), tile); \
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
            auto suffix = suffix_type(to_insert, tile, allocator); \
            static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1; \
            suffix.create_from(key + suffix_offset, key_length - suffix_offset, value); \
            suffix.store_head(); \
          } \
          node.insert(first_slice, to_insert, more_key); \
          node.template store_to_array<false>(d_table_); \
          node_type::unlock<false>(d_table_, node0.get_node_index(), tile); \
          node_type::unlock(d_table_, node1.get_node_index(), tile); \
          return true; \
        }
        TRY_INSERT_TO_NODE_IF_NOT_FULL(node0)
        TRY_INSERT_TO_NODE_IF_NOT_FULL(node1)
        #undef TRY_INSERT_TO_NODE_IF_NOT_FULL
        // All nodes are full.
        node_type::unlock<false>(d_table_, node0.get_node_index(), tile);
        node_type::unlock(d_table_, node1.get_node_index(), tile);
      }
      // === Phase 3. Try make space with cuckoo, BFS depth=1 ===
      bool cuckoo_succeed = false; // if we made the empty slot
      #define TRY_MAKE_SPACE_WITH_CUCKOO_FROM_NODE(node, current_table_i) \
      assert(node.is_full()); \
      for (uint32_t loc = 0; loc < node.capacity; loc++) { \
        auto other_node = coop_get_other_bucket_of_key_in<use_hash_tag>( \
            node, loc, current_table_i, tile, allocator); \
        if (!other_node.is_full()) { \
          /*found the space*/ \
          key_slice_type target_key = node.get_key_from_location(loc); \
          value_type target_value = node.get_value_from_location(loc); \
          bool target_suffix = node.get_suffix_of_location(loc); \
          lock_two_nodes_in_order(node, other_node, tile); \
          node.template load_from_array<true>(d_table_); /*first load use acquire*/ \
          other_node.template load_from_array<false>(d_table_); \
          if (!other_node.is_full()) { \
            /*check the key still exists in node*/ \
            uint32_t to_check = node.match_key_value_in_node(target_key, target_value, target_suffix); \
            if (to_check != 0) { \
              assert(__popc(to_check) == 1); \
              /*move the element*/ \
              other_node.insert(target_key, target_value, target_suffix); \
              node.erase(__ffs(to_check) - 1); \
              if (tile.thread_rank() == 0) { \
                cuda::atomic_ref<size_type, cuda::thread_scope_device> version_ref(d_versions_[(target_key * utils::PRIME2) % version_counter_size]); \
                version_ref.fetch_add(1, cuda::memory_order_release); \
              } \
              other_node.template store_to_array<false>(d_table_); \
              node.template store_to_array<false>(d_table_); \
              if (tile.thread_rank() == 0) { \
                cuda::atomic_ref<size_type, cuda::thread_scope_device> version_ref(d_versions_[(target_key * utils::PRIME2) % version_counter_size]); \
                version_ref.fetch_add(1, cuda::memory_order_release); \
              } \
              cuckoo_succeed = true; \
            } \
          } \
          node_type::unlock<false>(d_table_, node.get_node_index(), tile); \
          node_type::unlock(d_table_, other_node.get_node_index(), tile); \
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

  template <bool use_hash_tag, bool _, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    auto hash = utils::compute_hashx2<utils::PRIME0, utils::PRIME1>(key, key_length, tile);
    const bool more_key = (key_length > 1);
    key_slice_type first_slice = (use_hash_tag && more_key) ?
        (hash.x + hash.y) : key[0];
    uint32_t table_i = ((hash.x ^ hash.y) * utils::PRIME2) % num_hfs; // 2-in-d cuckoo hashing
    auto node0 = node_type(bucket_index_of(table_i, hash.x), tile, allocator);
    auto node1 = node_type(bucket_index_of((table_i + 1) % num_hfs, hash.y), tile, allocator);
    // lock all nodes in order
    lock_two_nodes_in_order(node0, node1, tile);
    // check nodes
    int location_if_found;
    suffix_type suffix_if_found(tile, allocator);
    #define TRY_ERASE_KEY_IN_NODE(node) \
    node.template load_from_array<true>(d_table_); \
    location_if_found = coop_get_key_location_from_node<use_hash_tag>( \
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator); \
    if (location_if_found >= 0) { \
      node.erase(location_if_found); \
      node.template store_to_array<false>(d_table_); \
      if (more_key) { \
        suffix_if_found.retire(reclaimer); \
      } \
      node_type::unlock<false>(d_table_, node0.get_node_index(), tile); \
      node_type::unlock(d_table_, node1.get_node_index(), tile); \
      return true; \
    }
    TRY_ERASE_KEY_IN_NODE(node0)
    TRY_ERASE_KEY_IN_NODE(node1)
    #undef TRY_ERASE_KEY_IN_NODE
    // not found
    node_type::unlock<false>(d_table_, node0.get_node_index(), tile);
    node_type::unlock(d_table_, node1.get_node_index(), tile);
    return false;
  }

 private:
  // device-side helper functions
  DEVICE_QUALIFIER size_type bucket_index_of(uint32_t table_i, size_type bucket_index_hash) {
    return (table_i * num_buckets_per_hf_) + (bucket_index_hash % num_buckets_per_hf_);
  }

  template <bool use_hash_tag, typename tile_type>
  DEVICE_QUALIFIER int coop_get_key_location_from_node(hashtable_node<tile_type, device_allocator_context_type>& node,
                                                       const key_slice_type& first_slice,
                                                       bool more_key,
                                                       const key_slice_type* key,
                                                       const size_type& key_length,
                                                       suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
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
        int location = __ffs(to_check) - 1;
        return location;
      }
    }
    // not found
    return -1;
  }

  template <bool use_hash_tag, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER hashtable_node<tile_type, device_allocator_context_type>
        coop_get_other_bucket_of_key_in(hashtable_node<tile_type, device_allocator_context_type>& node,
                                        uint32_t location,
                                        uint32_t current_table_i,
                                        const tile_type& tile,
                                        allocator_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    // compute hash for the key
    uint2 hash;
    if (node.get_suffix_of_location(location)) {
      auto suffix_index = node.get_value_from_location(location);
      auto suffix = suffix_type(suffix_index, tile, allocator);
      suffix.load_head();
      hash = utils::compute_hashx2_suffix<utils::PRIME0, utils::PRIME1, !use_hash_tag>(
        suffix, use_hash_tag ? 0 : node.get_key_from_location(location), tile);
    }
    else {
      hash = utils::compute_hashx2_slice<utils::PRIME0, utils::PRIME1>(node.get_key_from_location(location), tile);
    }
    uint32_t target_table_i = ((hash.x ^ hash.y) * utils::PRIME2) % num_hfs;
    // two hash buckets for the key are
    //    (target_table_i,                  hash.x % num_buckets_per_hf_) and
    //    ((target_table_i + 1) % num_hfs,  hash.y % num_buckets_per_hf_)
    assert(current_table_i == target_table_i || current_table_i == (target_table_i + 1) % num_hfs);
    if (target_table_i == current_table_i) {
      target_table_i = (target_table_i + 1) % num_hfs;
      hash.x = hash.y;
    }
    auto other_node = node_type(bucket_index_of(target_table_i, hash.x), tile, allocator);
    other_node.template load_from_array<true>(d_table_);
    return other_node;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void lock_two_nodes_in_order(hashtable_node<tile_type, device_allocator_context_type>& node0,
                                                hashtable_node<tile_type, device_allocator_context_type>& node1,
                                                const tile_type& tile) {
    // lock node with smaller address first
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    assert(node0.get_node_index() != node1.get_node_index());
    if (node0.get_node_index() < node1.get_node_index()) {
      node_type::lock(d_table_, node0.get_node_index(), tile);
      node_type::lock(d_table_, node1.get_node_index(), tile);
    }
    else {
      node_type::lock(d_table_, node1.get_node_index(), tile);
      node_type::lock(d_table_, node0.get_node_index(), tile);
    }
  }

 public:
  // device-side debug functions
  template <typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_nodes(Func& task, const tile_type& tile) {
    // debug-purpose, so inefficient implementation
    // called with single warp
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    device_allocator_context_type allocator{allocator_, tile};
    for (int table_index = 0; table_index < num_hfs; table_index++) {
      for (size_type bucket_index = 0; bucket_index < num_buckets_per_hf_; bucket_index++) {
        size_type global_bucket_index = num_buckets_per_hf_ * table_index + bucket_index;
        auto node = node_type(global_bucket_index, tile, allocator);
        node.template load_from_array<false>(d_table_);
        task.exec(node, table_index, bucket_index, tile, allocator);
      }
    }
  }

  template <typename func>
  void traverse_nodes(func task) {
    static constexpr auto block_size = tile_size_;
    kernels::GpuHashtable::traverse_nodes_kernel<block_size><<<1, block_size>>>(*this, task);
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
          auto suffix = suffix_node<tile_type, device_allocator_context_type>(suffix_index, tile, allocator);
          suffix.load_head();
          num_suffix_nodes_ += suffix.get_num_nodes();
        }
      }
      num_buckets_++;
      num_entries_ += num_keys;
      entries_per_table_[table_index] += num_keys;
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void fini(const tile_type& tile) {
      uint64_t total_bytes_used = (num_buckets_ + num_suffix_nodes_) * bucket_bytes;
      float bytes_per_entry = static_cast<float>(total_bytes_used) / num_entries_;
      if (tile.thread_rank() == 0) {
        printf("%lu entries (", num_entries_);
        for (int i = 0; i < num_hfs; i++) {
          printf("%lu ", entries_per_table_[i]);
        }
        printf("), %lu buckets (+%lu suffix nodes)\n", num_buckets_, num_suffix_nodes_);
        printf("Total Space Consumption: %lu B (%f B/entry)\n", total_bytes_used, bytes_per_entry);
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
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    auto node = node_type(bucket_index, tile, allocator);
    node.initialize_empty(true);
    node.template store_to_array<false>(d_table_);
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
    const uint32_t block_size = tile_size_;
    kernels::GpuHashtable::initialize_kernel<block_size><<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
    cuda_try(cudaMemset(d_versions_, 0, sizeof(size_type) * version_counter_size));
  }

  elem_type* d_table_;
  size_type* d_versions_;
  bool is_owner_;
  size_type num_buckets_per_hf_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <uint32_t _tile_size, typename hashtable>
  friend __global__ void kernels::GpuHashtable::initialize_kernel(hashtable);

  template <bool do_reclaim, uint32_t _tile_size, typename device_func, typename index_type>
  friend __global__ void kernels::batch_kernel(index_type index,
                                              const device_func func,
                                              uint32_t num_requests);

};

} // namespace GpuHashtable
