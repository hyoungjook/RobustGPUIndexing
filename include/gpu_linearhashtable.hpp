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
#include <simple_bump_linear_alloc.hpp>
#include <simple_slab_linear_alloc.hpp>
#include <simple_dummy_reclaim.hpp>
#include <simple_debra_reclaim.hpp>

namespace GpuLinearHashtable {

template <typename Allocator,
          typename Reclaimer>
struct gpu_linearhashtable {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using table_ptr_type = uint64_t;
  static auto constexpr bucket_size = 32;
  static std::size_t constexpr bucket_bytes = sizeof(elem_type) * bucket_size;
  static auto constexpr cg_tile_size = 32;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();
  static constexpr size_type invalid_pointer = std::numeric_limits<size_type>::max();
  static constexpr size_type max_directory_size = 128 * 1024 * 1024; // 0.5GB
  static constexpr float load_factor_threshold = 1.0f;
  static constexpr size_type check_load_factor_every = 128;

  using host_allocator_type = Allocator;
  using device_allocator_instance_type = typename host_allocator_type::device_instance_type;
  using device_allocator_context_type = device_allocator_context<host_allocator_type>;

  using host_reclaimer_type = Reclaimer;
  using device_reclaimer_instance_type = typename host_reclaimer_type::device_instance_type;
  using device_reclaimer_context_type = device_reclaimer_context<host_reclaimer_type>;

  gpu_linearhashtable() = delete;
  // resize_policy: if > 0, linear resizing by the amount (converted to int)
  //                if < 0, exponential resizing by the amount
  gpu_linearhashtable(const host_allocator_type& host_allocator,
                      const host_reclaimer_type& host_reclaimer,
                      size_type initial_directory_size,
                      float resize_policy)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance())
      , initial_directory_size_(initial_directory_size)
      , resize_policy_(resize_policy) {
    if ((resize_policy >= 0 && (resize_policy > 2.0f || resize_policy <= 1.0f)) ||
        (resize_policy < 0 && (static_cast<size_type>(-resize_policy) % cg_tile_size != 0))) {
      fprintf(stderr, "Invalid resize_policy %f for GPULinearHT: "
                      "If >0 (exponential), should be in (1, 2], "
                      "If <0 (linear), should be multiple of %u\n",
                      resize_policy, cg_tile_size);
      exit(1);
    }
    allocate();
  }

  gpu_linearhashtable& operator=(const gpu_linearhashtable& other) = delete;
  gpu_linearhashtable(const gpu_linearhashtable& other)
      : d_global_state_(other.d_global_state_)
      , is_owner_(false)
      , initial_directory_size_(other.initial_directory_size_)
      , resize_policy_(other.resize_policy_)
      , allocator_(other.allocator_)
      , reclaimer_(other.reclaimer_) {}

  ~gpu_linearhashtable() {
    deallocate();
  }

  // host-side APIs
  // if key_lengths == NULL, we use max_key_length as a fixed length
  template <bool concurrent = false,
            bool use_hash_tag = true,
            bool tag_use_same_hash = true,
            bool reuse_dirsize = true>
  void find(const key_slice_type* keys,
            const size_type max_key_length,
            const size_type* key_lengths,
            value_type* values,
            const size_type num_keys,
            cudaStream_t stream = 0) {
    kernels::GpuLinearHashtable::find_device_func<concurrent, use_hash_tag, tag_use_same_hash, reuse_dirsize, key_slice_type, size_type, value_type>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true,
            bool tag_use_same_hash = true,
            bool reuse_dirsize = true>
  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0,
              bool update_if_exists = false) {
    kernels::GpuLinearHashtable::insert_device_func<use_hash_tag, tag_use_same_hash, reuse_dirsize, key_slice_type, size_type, value_type>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true,
            bool tag_use_same_hash = true,
            bool do_merge_buckets = true,
            bool do_merge_chains = true,
            bool reuse_dirsize = true>
  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0) {
    kernels::GpuLinearHashtable::erase_device_func<use_hash_tag, tag_use_same_hash, do_merge_chains, do_merge_buckets, reuse_dirsize, key_slice_type, size_type, value_type>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_hash_tag = true,
            bool tag_use_same_hash = true,
            bool erase_do_merge_buckets = true,
            bool erase_do_merge_chains = true,
            bool reuse_dirsize = true>
  void mixed_batch(const kernels::request_type* request_types,
                   const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   value_type* values,
                   bool* results,
                   const size_type num_requests,
                   cudaStream_t stream = 0,
                   bool insert_update_if_exists = false) {
    kernels::GpuLinearHashtable::mixed_device_func<use_hash_tag, tag_use_same_hash, erase_do_merge_chains, erase_do_merge_buckets, reuse_dirsize, key_slice_type, size_type, value_type>
      func{.d_types = request_types, .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .d_results = results, .insert_update_if_exists = insert_update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_requests, stream);
  }

  // device-side APIs
  template <bool concurrent>
  DEVICE_QUALIFIER size_type cooperative_fetch_dirsize() {
    size_type directory_size = d_global_state_->template load_directory_size<concurrent>();
    return directory_size;
  }

  template <bool concurrent, bool use_hash_tag, bool tag_use_same_hash, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find_from_dirsize(size_type& directory_size,
                                                            const key_slice_type* key,
                                                            size_type key_length,
                                                            const tile_type& tile,
                                                            device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index_hash;
    const bool more_key = (key_length > 1);
    if (use_hash_tag && !tag_use_same_hash && more_key) {
      auto hash = compute_hashx2(key, key_length, tile);
      bucket_index_hash = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index_hash = compute_hash(key, key_length, tile);
      first_slice = (tag_use_same_hash && more_key) ? bucket_index_hash : key[0];
    }
    // find the bucket
    node_type node(tile, allocator);
    find_valid_bucket<concurrent>(node, bucket_index_hash, directory_size, tile, allocator);
    // search bucket
    suffix_type suffix_if_found(tile, allocator);
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

  template <bool concurrent, bool use_hash_tag, bool tag_use_same_hash, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator) {
    auto directory_size = cooperative_fetch_dirsize<concurrent>();
    return cooperative_find_from_dirsize<concurrent, use_hash_tag, tag_use_same_hash>(
        directory_size, key, key_length, tile, allocator);
  }

  template <bool use_hash_tag, bool tag_use_same_hash, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert_from_dirsize(size_type& directory_size,
                                                        const key_slice_type* key,
                                                        const size_type key_length,
                                                        const value_type& value,
                                                        const tile_type& tile,
                                                        device_allocator_context_type& allocator,
                                                        device_reclaimer_context_type& reclaimer,
                                                        bool update_if_exists = false) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index_hash;
    const bool more_key = (key_length > 1);
    if (use_hash_tag && !tag_use_same_hash && more_key) {
      auto hash = compute_hashx2(key, key_length, tile);
      bucket_index_hash = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index_hash = compute_hash(key, key_length, tile);
      first_slice = (tag_use_same_hash && more_key) ? bucket_index_hash : key[0];
    }
    const bool check_load_factor = (bucket_index_hash % check_load_factor_every == 0);
    while (true) {
      // find the bucket
      node_type node(tile, allocator);
      bool met_invalid_pointer = find_valid_bucket<true>(node, bucket_index_hash, directory_size, tile, allocator);
      auto head_index = node.get_node_index();
      node_type::lock(head_index, tile, allocator);
      node.template load_from_allocator<true>();
      if (node.is_garbage()) {
        // this bucket just splitted by other thread; retry
        node_type::unlock(head_index, tile, allocator);
        continue;
      }
      suffix_type suffix_if_found(tile, allocator);
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
            node.template store_to_allocator<false>();
          }
        }
        node_type::unlock(head_index, tile, allocator);
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
        new_node.initialize_empty(false, node.get_local_depth());
        new_node.insert(first_slice, to_insert, more_key);
        // write order: new_node -> node
        new_node.template store_to_allocator<false>();
        node.set_next_index(next_index);
        node.set_has_next();
        node.template store_to_allocator<true>();
      }
      else { // !node.is_full()
        node.insert(first_slice, to_insert, more_key);
        node.template store_to_allocator<false>();
      }
      // check if chain is too long (not one)
      if (!node.is_head()) {
        // check if split is possible
        auto local_depth = node.get_local_depth();
          // first bucket that points to this node
        auto first_bucket_index = bucket_index_hash & ((1u << local_depth) - 1);
        if ((first_bucket_index ^ (1u << local_depth)) < directory_size) {
          // do split!
          node = node_type(head_index, tile, allocator);
          node.template load_from_allocator<false>();
          // mark garbage
          node.make_garbage();
          node.template store_to_allocator<false>();
          // allocate two new node chains
          auto new_node0_index = allocator.allocate(tile);
          auto new_node1_index = allocator.allocate(tile);
          auto new_node0 = node_type(new_node0_index, tile, allocator);
          auto new_node1 = node_type(new_node1_index, tile, allocator);
          new_node0.initialize_empty(true, local_depth + 1);
          new_node1.initialize_empty(true, local_depth + 1);
          // split
          while (true) {
            for (uint32_t loc = 0; loc < node.num_keys(); loc++) {
              // decide new_node0 or new_node1
              uint32_t bucket_index_hash_at_loc;
              if (node.get_suffix_of_location(loc)) {
                if constexpr (tag_use_same_hash) {
                  bucket_index_hash_at_loc = node.get_key_from_location(loc);
                }
                else {
                  auto suffix_index = node.get_value_from_location(loc);
                  auto suffix = suffix_type(suffix_index, tile, allocator);
                  suffix.load_head();
                  bucket_index_hash_at_loc = compute_hash_for_suffix<use_hash_tag>(
                      suffix, use_hash_tag ? 0 : node.get_key_from_location(loc), tile);
                }
              }
              else {
                bucket_index_hash_at_loc = compute_hash_single_slice(node.get_key_from_location(loc));
              }
              // store to either new_node0 or new_node1
              if ((bucket_index_hash_at_loc & (1u << local_depth)) == 0) {  // new_node0
                if (new_node0.is_full()) {
                  auto new_aux_index = allocator.allocate(tile);
                  new_node0.set_next_index(new_aux_index);
                  new_node0.set_has_next();
                  new_node0.template store_to_allocator<false>();
                  new_node0 = node_type(new_aux_index, tile, allocator);
                  new_node0.initialize_empty(false, local_depth + 1);
                }
                new_node0.insert(node.get_key_from_location(loc),
                                 node.get_value_from_location(loc),
                                 node.get_suffix_of_location(loc));
              }
              else {  // new_node1
                if (new_node1.is_full()) {
                  auto new_aux_index = allocator.allocate(tile);
                  new_node1.set_next_index(new_aux_index);
                  new_node1.set_has_next();
                  new_node1.template store_to_allocator<false>();
                  new_node1 = node_type(new_aux_index, tile, allocator);
                  new_node1.initialize_empty(false, local_depth + 1);
                }
                new_node1.insert(node.get_key_from_location(loc),
                                 node.get_value_from_location(loc),
                                 node.get_suffix_of_location(loc));
              }
            }
            if (!node.has_next()) { break; }
            auto next_index = node.get_next_index();
            node = node_type(next_index, tile, allocator);
            node.template load_from_allocator<false>(); // first load after lock already did memory_order_acquire
            reclaimer.retire(next_index, tile);
          }
          // store last nodes of two new buckets
          new_node0.template store_to_allocator<false>();
          new_node1.template store_to_allocator<true>();  // last store releases before updating directory
          // publish new buckets: 
          auto local_depth_mask = (1u << local_depth);
          directory_size = d_global_state_->template load_directory_size<true>();
          for (size_type index = first_bucket_index; index < directory_size; index += local_depth_mask) {
            auto new_node_index = (index & local_depth_mask) == 0 ? new_node0_index : new_node1_index;
            directory_entry_at(index, allocator) = new_node_index;
          }
          reclaimer.retire(head_index, tile);
          met_invalid_pointer = false;
        }
      }
      if (met_invalid_pointer) {
        // copy pointers
        auto local_depth = node.get_local_depth();
        auto first_bucket_index = bucket_index_hash & ((1u << local_depth) - 1);
        auto local_depth_mask = (1u << local_depth);
        directory_size = d_global_state_->template load_directory_size<true>();
        for (size_type index = first_bucket_index; index < directory_size; index += local_depth_mask) {
          directory_entry_at(index, allocator) = head_index;
        }
      }
      node_type::unlock(head_index, tile, allocator);
      // extend if load factor is too high
      if (check_load_factor) {
        auto num_entries = d_global_state_->template increment_num_entries<1>(tile);
        if ((static_cast<float>(num_entries) / directory_size * (static_cast<float>(check_load_factor_every) / 15.0f)) > load_factor_threshold) {
          if (d_global_state_->try_lock(tile)) {
            auto curr_directory_size = d_global_state_->template load_directory_size<true>();
            if (curr_directory_size == directory_size) {
              auto new_directory_size =
                (resize_policy_ > 0) ? static_cast<size_type>(static_cast<float>(directory_size) * resize_policy_):
                                       (directory_size + static_cast<size_type>(-resize_policy_));
              new_directory_size = (new_directory_size + cg_tile_size - 1) / cg_tile_size * cg_tile_size;  // should be multiple of 32
              new_directory_size = allocator.reallocate_linear(new_directory_size, tile);
              if (new_directory_size > curr_directory_size) {
                // invalidate new pointers
                for (size_type bucket = directory_size; bucket < new_directory_size; bucket += 32) {
                  directory_entry_at(bucket + tile.thread_rank(), allocator) = invalid_pointer;
                }
                // publish new directory
                directory_size = new_directory_size;
                d_global_state_->template store_directory_size<true>(directory_size);
              }
            }
            else {
              directory_size = curr_directory_size;
            }
            d_global_state_->unlock(tile);
          }
        }
      }
      return true;
    }
    assert(false);
  }

  template <bool use_hash_tag, bool tag_use_same_hash, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           device_reclaimer_context_type& reclaimer,
                                           bool update_if_exists = false) {
    auto directory_size = cooperative_fetch_dirsize<true>();
    return cooperative_insert_from_dirsize<use_hash_tag, tag_use_same_hash>(
        directory_size, key, key_length, value, tile, allocator, reclaimer);
  }

  template <bool use_hash_tag, bool tag_use_same_hash, bool do_merge_chains, bool do_merge_buckets, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase_from_dirsize(size_type& directory_size,
                                                       const key_slice_type* key,
                                                       const size_type key_length,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator,
                                                       device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index_hash;
    const bool more_key = (key_length > 1);
    if (use_hash_tag && !tag_use_same_hash && more_key) {
      auto hash = compute_hashx2(key, key_length, tile);
      bucket_index_hash = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index_hash = compute_hash(key, key_length, tile);
      first_slice = (tag_use_same_hash && more_key) ? bucket_index_hash : key[0];
    }
    const bool check_load_factor = (bucket_index_hash % check_load_factor_every == 0);
    while (true) {
      // find the bucket
      node_type node(tile, allocator);
      find_valid_bucket<true>(node, bucket_index_hash, directory_size, tile, allocator);
      auto head_index = node.get_node_index();
      node_type::lock(head_index, tile, allocator);
      node.template load_from_allocator<true>();
      if (node.is_garbage()) {
        // this bucket just splitted by other thread; retry
        node_type::unlock(head_index, tile, allocator);
        directory_size = d_global_state_->template load_directory_size<true>();
        continue;
      }
      int location_if_found;
      suffix_type suffix_if_found(tile, allocator);
      if constexpr (do_merge_chains) {
        location_if_found = coop_traverse_until_found_merge<use_hash_tag>(
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator, reclaimer);
      }
      else {
        location_if_found = coop_traverse_until_found<false, use_hash_tag>(  // use weak load here b/c the first load did memory_order_acquire
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
      }
      if (location_if_found >= 0) { // exists
        node.erase(location_if_found);
        node.template store_to_allocator<false>();
        if (more_key) {
          suffix_if_found.retire(reclaimer);
        }
        node_type::unlock(head_index, tile, allocator);
        if (check_load_factor) {
          // Since we decrement only on successful erase,
          //  and check_load_factor is computed in the same way with insertion,
          //  this never goes to negative value even though we do random sampling
          d_global_state_->template increment_num_entries<-1>(tile);
        }
        if constexpr (do_merge_buckets) {
          if (node.num_keys() == 0 && !node.has_next()) {
            // bucket was empty, try merge with sibling
            auto local_depth = node.get_local_depth();
            auto bucket_index = bucket_index_hash & ((1u << local_depth) - 1);
            while (local_depth > compute_global_depth(initial_directory_size_)) {
              // find sibling buckets
              auto sibling_bucket_index = bucket_index ^ (1u << (local_depth - 1));
              if (sibling_bucket_index >= directory_size) { break; }
              auto head0_index = utils::memory::load<size_type, true, true>(&directory_entry_at(bucket_index, allocator));
              auto head1_index = utils::memory::load<size_type, true, true>(&directory_entry_at(sibling_bucket_index, allocator));
              if (head0_index == invalid_pointer ||
                  head1_index == invalid_pointer ||
                  head0_index == head1_index) { break; }
              // lock the nodes in global order
              if (head0_index < head1_index) {
                node_type::lock(head0_index, tile, allocator);
                node_type::lock(head1_index, tile, allocator);
              }
              else {
                node_type::lock(head1_index, tile, allocator);
                node_type::lock(head0_index, tile, allocator);
              }
              // node is the one we observed empty
              node = node_type(head0_index, tile, allocator);
              node.template load_from_allocator<true>();
              bool cascade_merge = false;
              if (!node.is_garbage() && node.get_local_depth() == local_depth &&
                  node.num_keys() == 0 && !node.has_next()) {
                auto node1 = node_type(head1_index, tile, allocator);
                node1.template load_from_allocator<true>();
                if (!node1.is_garbage() && node1.get_local_depth() == local_depth) {
                  // do merge!
                  // order: node1.decrement local depth -> update pointers -> node.make garbage
                  node1.set_local_depth(local_depth - 1);
                  node1.template store_to_allocator<true>();
                  auto local_depth_mask = (1u << local_depth);
                  directory_size = d_global_state_->template load_directory_size<true>();
                  for (size_type index = bucket_index; index < directory_size; index += local_depth_mask) {
                    directory_entry_at(index, allocator) = head1_index;
                  }
                  node.make_garbage();
                  node.template store_to_allocator<true>();
                  reclaimer.retire(head0_index, tile);
                  cascade_merge = (node1.num_keys() == 0 && !node1.has_next());
                }
              }
              node_type::unlock<false>(head0_index, tile, allocator);
              node_type::unlock(head1_index, tile, allocator);
              // if cascading
              if (cascade_merge) {
                local_depth--;
                bucket_index = sibling_bucket_index & ((1u << local_depth) - 1);
              }
              else { break; }
            }
          }
        }
        return true;
      }
      // not exists
      node_type::unlock(head_index, tile, allocator);
      return false;
    }
  }

  template <bool use_hash_tag, bool tag_use_same_hash, bool do_merge_chains, bool do_merge_buckets, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    auto directory_size = cooperative_fetch_dirsize<true>();
    return cooperative_erase_from_dirsize<use_hash_tag, tag_use_same_hash, do_merge_chains, do_merge_buckets>(
        directory_size, key, key_length, tile, allocator, reclaimer);
  }

 private:
  // device-side helper functions
  struct __align__(128) global_state {
    size_type mutex_;
    size_type directory_size_;
    uint8_t _align_buf0[128 - 2 * sizeof(size_type)];
    size_type num_entries_;
    uint8_t _align_buf1[128 - sizeof(size_type)];

    DEVICE_QUALIFIER global_state(size_type initial_directory_size)
        : directory_size_(initial_directory_size)
        , mutex_(0) {}
    
    template <bool atomic, bool acquire = true>
    DEVICE_QUALIFIER size_type load_directory_size() {
      return utils::memory::load<size_type, atomic, acquire>(&directory_size_);
    }
    template <bool atomic, bool release = true>
    DEVICE_QUALIFIER void store_directory_size(size_type directory_size) {
      utils::memory::store<size_type, atomic, release>(&directory_size_, directory_size);
    }
    template <typename tile_type>
    DEVICE_QUALIFIER bool try_lock(const tile_type& tile) {
      size_type old = 0;
      if (tile.thread_rank() == 0) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device> mutex_ref(mutex_);
        mutex_ref.compare_exchange_strong(old, static_cast<size_type>(1),
                                          cuda::memory_order_acquire,
                                          cuda::memory_order_relaxed);
      }
      return (tile.shfl(old, 0) == 0);
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void lock(const tile_type& tile) {
      while (!try_lock(tile));
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void unlock(const tile_type& tile) {
      if (tile.thread_rank() == 0) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device> mutex_ref(mutex_);
        mutex_ref.store(0, cuda::memory_order_release);
      }
    }
    DEVICE_QUALIFIER size_type get_num_entries() {
      return utils::memory::load<size_type, true, false>(&num_entries_);
    }
    template <int amount, typename tile_type>
    DEVICE_QUALIFIER size_type increment_num_entries(const tile_type& tile) {
      size_type old;
      if (tile.thread_rank() == 0) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device> num_entries_ref(num_entries_);
        if constexpr (amount > 0) {
          old = num_entries_ref.fetch_add(static_cast<size_type>(amount), cuda::memory_order_relaxed);
        }
        else {
          old = num_entries_ref.fetch_sub(static_cast<size_type>(-amount), cuda::memory_order_relaxed);
        }
      }
      return tile.shfl(old, 0);
    }
  };

  static DEVICE_QUALIFIER size_type compute_global_depth(size_type directory_size) {
    // smallest n of (2^n >= directory_size)
    return utils::bits::bfind(directory_size - 1) + 1;
  }

  DEVICE_QUALIFIER size_type& directory_entry_at(size_type index,
                                                 device_allocator_context_type& allocator) {
    return *(reinterpret_cast<size_type*>(allocator.get_linear()) - (1 + index));
  }

  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER bool find_valid_bucket(hashtable_node<tile_type, device_allocator_context_type>& node,
                                          size_type bucket_index_hash,
                                          size_type& directory_size,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    size_type global_depth = compute_global_depth(directory_size);
    size_type bucket_index = bucket_index_hash & ((1u << global_depth) - 1);
    if (bucket_index >= directory_size) {
      bucket_index ^= (1u << (global_depth - 1));
      assert(bucket_index < directory_size);
    }
    bool met_invalid_pointer = false;
    while (true) {
      size_type head_index = utils::memory::load<size_type, concurrent, true>(
          &directory_entry_at(bucket_index, allocator));
      if (head_index == invalid_pointer) {
        // retry after unsetting MSB 1
        assert(bucket_index != 0);
        bucket_index ^= (1u << utils::bits::bfind(bucket_index));
        met_invalid_pointer = true;
        continue;
      }
      node = node_type(head_index, tile, allocator);
      node.template load_from_allocator<concurrent>();
      // local_depth-masked bucket index should match with the hash
      size_type local_depth_mask = (1u << node.get_local_depth()) - 1;
      if (((bucket_index_hash ^ bucket_index) & local_depth_mask) != 0) {
        // else: other warp splitted meanwhile, retry with larger local_depth
        bucket_index = bucket_index_hash & local_depth_mask;
        if (bucket_index >= directory_size) {
          // other warp extended directory meanwhile
          directory_size = d_global_state_->template load_directory_size<concurrent>();
          assert(bucket_index < directory_size);
        }
        continue;
      }
      return met_invalid_pointer;
    }
  }

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
          // current_node_store_deferred: USER SHOULD STORE the node returned
          return __ffs(to_check) - 1;
        }
      }
      if (current_node_store_deferred) {
        node.template store_to_allocator<false>();  // future unlock will do memory_order_release
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

  static constexpr uint32_t hash_prime0 = 0x9e3779b1;
  static constexpr uint32_t hash_prime1 = 0x01000193;
  static DEVICE_QUALIFIER uint32_t hash_murmur3_finalizer(uint32_t hash) {
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
  }
  template <typename tile_type>
  DEVICE_QUALIFIER uint32_t compute_hash(const key_slice_type* key, size_type key_length, const tile_type& tile) {
    // parallel polynomial rolling hash
    static constexpr uint32_t prime_multiplier = utils::constexpr_pow(hash_prime0, cg_tile_size);
    // 1. exponent = [1, p, p^2, ..., p^31]; parallel prefix product
    uint32_t exponent = (tile.thread_rank() == 0) ? 1 : hash_prime0;
    for (uint32_t offset = 1; offset < cg_tile_size; offset <<= 1) {
      auto up_exponent = tile.shfl_up(exponent, offset);
      if (tile.thread_rank() >= offset) {
        exponent *= up_exponent;
      }
    }
    // 2. compute per-lane value
    const auto original_length = key_length;
    uint32_t hash = 0;
    while (true) {
      if (tile.thread_rank() < key_length) {
        auto slice = key[tile.thread_rank()];
        hash += exponent * slice;
      }
      if (key_length <= cg_tile_size) { break; }
      key += cg_tile_size;
      key_length -= cg_tile_size;
      exponent *= prime_multiplier;
    }
    // 3. reduce sum
    for (uint32_t offset = (cg_tile_size / 2); offset != 0; offset >>= 1) {
      hash += tile.shfl_down(hash, offset);
    }
    hash = ((hash * hash_prime0) + original_length) * hash_prime0;
    // 4. finalize
    hash = hash_murmur3_finalizer(hash);
    return tile.shfl(hash, 0);
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
  DEVICE_QUALIFIER uint32_t compute_hash_single_slice(const key_slice_type& key) {
    // if key_length == 1, hash = murmur3(((key * p) + 1) * p)
    uint32_t hash = ((key * hash_prime0) + 1) * hash_prime0;
    return hash_murmur3_finalizer(hash);
  }
  template <bool use_hash_tag, typename suffix_type, typename tile_type>
  DEVICE_QUALIFIER uint32_t compute_hash_for_suffix(const suffix_type& suffix,
                                                    const key_slice_type& first_slice,
                                                    const tile_type& tile) {
    // compute polynomial
    uint32_t hash = suffix.template compute_polynomial<hash_prime0>();
    if constexpr (!use_hash_tag) {
      hash = (hash * hash_prime0) + first_slice;
    }
    static constexpr uint32_t suffix_offset = use_hash_tag ? 0 : 1;
    uint32_t key_length = suffix.get_key_length() + suffix_offset;
    hash = ((hash * hash_prime0) + key_length) * hash_prime0;
    // finalize
    return hash_murmur3_finalizer(hash);
  }

 public:
  // device-side debug functions
  template <typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_nodes(Func& task, const tile_type& tile) {
    // debug-purpose, so inefficient implementation
    // called with single warp
    using node_type = hashtable_node<tile_type, device_allocator_context_type>;
    device_allocator_context_type allocator{allocator_, tile};
    size_type directory_size = d_global_state_->template load_directory_size<false>();
    auto global_depth = compute_global_depth(directory_size);
    for (size_type bucket_index = 0; bucket_index < directory_size; bucket_index++) {
      auto node_index = directory_entry_at(bucket_index, allocator);
      if (node_index == invalid_pointer) continue;
      auto node = node_type(node_index, tile, allocator);
      node.template load_from_allocator<false>();
      // Check if this is the first pointer to this node
      size_type global_minus_local_mask = (1u << global_depth) - 1;
      global_minus_local_mask &= ~((1u << node.get_local_depth()) - 1);
      if ((bucket_index & global_minus_local_mask) != 0) {
        // pointer to already-traversed node chain; skip
        continue;
      }
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
    kernels::GpuLinearHashtable::traverse_nodes_kernel<<<1, 32>>>(*this, task);
    cudaDeviceSynchronize();
  }

  struct print_nodes_task {
    template <typename tile_type>
    DEVICE_QUALIFIER void init(const tile_type& tile) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, int head_index, const tile_type& tile, device_allocator_context_type& allocator) {
      if (head_index >= 0 && tile.thread_rank() == 0) printf("HEAD[%d] ", head_index);
      node.print();
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
      if (tile.thread_rank() == 0) {
        printf("%lu entries (per-bucket max %lu, avg %f), %lu heads + %lu aux nodes (+%lu suffix nodes) (per-bucket max %lu, avg %f); fillfactor %f\n",
          num_entries_, max_entries_per_bucket_, avg_entries_per_bucket,
          num_head_nodes_, num_aux_nodes_, num_suffix_nodes_, max_nodes_per_bucket_, avg_nodes_per_bucket,
          fill_factor);
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
    // global state, initial array
    if (bucket_index == 0) {
      *d_global_state_ = global_state(initial_directory_size_);
      auto resized = allocator.reallocate_linear(initial_directory_size_, tile);
      assert(resized == initial_directory_size_);
    }
    // allocate node
    auto node_index = allocator.allocate(tile);
    auto node = node_type(node_index, tile, allocator);
    // initial local depth = initial global depth of initial directory size
    node.initialize_empty(true, compute_global_depth(initial_directory_size_));
    node.template store_to_allocator<false>();
    directory_entry_at(bucket_index, allocator) = node_index;
  }

  void allocate() {
    is_owner_ = true;
    cuda_try(cudaMalloc(&d_global_state_, sizeof(global_state)));
    initialize();
  }

  void deallocate() {
    if (is_owner_) {
      cuda_try(cudaFree(d_global_state_));
    }
  }

  void initialize() {
    const uint32_t num_blocks = initial_directory_size_;
    const uint32_t block_size = cg_tile_size;
    kernels::GpuLinearHashtable::initialize_kernel<<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
  }

  global_state* d_global_state_;
  bool is_owner_;
  size_type initial_directory_size_;
  float resize_policy_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <typename linearhashtable>
  friend __global__ void kernels::GpuLinearHashtable::initialize_kernel(linearhashtable);

  template <bool do_reclaim, typename device_func, typename index_type>
  friend __global__ void kernels::batch_kernel(index_type index,
                                              const device_func func,
                                              uint32_t num_requests);

};

} // namespace GpuLinearHashtable
