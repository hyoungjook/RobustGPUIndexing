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
struct gpu_linearhashtable {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using table_ptr_type = uint64_t;
  static auto constexpr bucket_size = 32;
  static std::size_t constexpr bucket_bytes = sizeof(elem_type) * bucket_size;
  static auto constexpr cg_tile_size = 32;
  using hashtable_type = gpu_linearhashtable<Allocator, Reclaimer>;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  // TODO adjust
  static constexpr size_type max_directory_size = 128 * 1024 * 1024; // 0.5GB
  static constexpr size_type directory_delta = 1;
  static constexpr float load_factor_threshold = 1.0f;

  using host_allocator_type = Allocator;
  using device_allocator_instance_type = typename host_allocator_type::device_instance_type;
  using device_allocator_context_type = device_allocator_context<host_allocator_type>;

  using host_reclaimer_type = Reclaimer;
  using device_reclaimer_instance_type = typename host_reclaimer_type::device_instance_type;
  using device_reclaimer_context_type = device_reclaimer_context<host_reclaimer_type>;

  gpu_linearhashtable() = delete;
  gpu_linearhashtable(const host_allocator_type& host_allocator,
           const host_reclaimer_type& host_reclaimer)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance()) {
    allocate();
  }

  gpu_linearhashtable& operator=(const gpu_linearhashtable& other) = delete;
  gpu_linearhashtable(const gpu_linearhashtable& other)
      : d_directory_(other.d_directory_)
      , d_global_state_(other.d_global_state_)
      , is_owner_(false)
      , allocator_(other.allocator_)
      , reclaimer_(other.reclaimer_) {}

  ~gpu_linearhashtable() {
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
    using find_concurrent_hash4long = kernel::GpuLinearHashtable::find_device_func<true, true, key_slice_type, size_type, value_type>;
    using find_concurrent_prfx4long = kernel::GpuLinearHashtable::find_device_func<true, false, key_slice_type, size_type, value_type>;
    using find_readonly_hash4long = kernel::GpuLinearHashtable::find_device_func<false, true, key_slice_type, size_type, value_type>;
    using find_readonly_prfx4long = kernel::GpuLinearHashtable::find_device_func<false, false, key_slice_type, size_type, value_type>;
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
    using insert_hash4long = kernel::GpuLinearHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuLinearHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
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
    using erase_merge_hash4long = kernel::GpuLinearHashtable::erase_device_func<true, true, key_slice_type, size_type, value_type>;
    using erase_merge_prfx4long = kernel::GpuLinearHashtable::erase_device_func<true, false, key_slice_type, size_type, value_type>;
    using erase_nomerge_hash4long = kernel::GpuLinearHashtable::erase_device_func<false, true, key_slice_type, size_type, value_type>;
    using erase_nomerge_prfx4long = kernel::GpuLinearHashtable::erase_device_func<false, false, key_slice_type, size_type, value_type>;
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
    using insert_hash4long = kernel::GpuLinearHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuLinearHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
    using erase_merge_hash4long = kernel::GpuLinearHashtable::erase_device_func<true, true, key_slice_type, size_type, value_type>;
    using erase_merge_prfx4long = kernel::GpuLinearHashtable::erase_device_func<true, false, key_slice_type, size_type, value_type>;
    using erase_nomerge_hash4long = kernel::GpuLinearHashtable::erase_device_func<false, true, key_slice_type, size_type, value_type>;
    using erase_nomerge_prfx4long = kernel::GpuLinearHashtable::erase_device_func<false, false, key_slice_type, size_type, value_type>;
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
    using insert_hash4long = kernel::GpuLinearHashtable::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_prfx4long = kernel::GpuLinearHashtable::insert_device_func<false, key_slice_type, size_type, value_type>;
    using find_concurrent_hash4long = kernel::GpuLinearHashtable::find_device_func<true, true, key_slice_type, size_type, value_type>;
    using find_concurrent_prfx4long = kernel::GpuLinearHashtable::find_device_func<true, false, key_slice_type, size_type, value_type>;
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
    key_slice_type first_slice;
    size_type bucket_index_hash;
    const bool more_key = (key_length > 1);
    if (use_hash_for_longkey && more_key) {
      auto hash = compute_hashx2(key, key_length, tile);
      bucket_index_hash = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index_hash = compute_hash(key, key_length, tile);
      first_slice = key[0];
    }
    suffix_type suffix_if_found(tile, allocator);
    size_type bucket_index = get_bucket_index(
        bucket_index_hash, d_global_state_->template load_directory_size<concurrent>());
    while (true) {
      auto head_index = utils::memory::load<size_type, concurrent, true>(d_directory_ + bucket_index);
      auto node = node_type(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
      node.template load<concurrent>();
      int location_if_found = coop_traverse_until_found<concurrent, use_hash_for_longkey>(
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
      if constexpr (concurrent) {
        auto retry_bucket_index = get_bucket_index(
          bucket_index_hash, d_global_state_->template load_directory_size<concurrent>());
        if (bucket_index != retry_bucket_index) {
          // If bucket index changed, retry
          bucket_index = retry_bucket_index;
          continue;
        }
      }
      break;
    }
    return invalid_value;
  }

  template <bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           device_reclaimer_context_type& reclaimer,
                                           bool update_if_exists = false) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index_hash;
    const bool more_key = (key_length > 1);
    if (use_hash_for_longkey && more_key) {
      auto hash = compute_hashx2(key, key_length, tile);
      bucket_index_hash = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index_hash = compute_hash(key, key_length, tile);
      first_slice = key[0];
    }
    // check load factor
    size_type directory_size = d_global_state_->template load_directory_size<true>();
    if ((load_factor_threshold < (
          static_cast<float>(d_global_state_->load_num_entries()) / 15.0f / directory_size)) &&
        (directory_size + directory_delta <= max_directory_size)) {
      // extend directory
      // if global state is already locked, someone else is already splitting so move on
      if (d_global_state_->try_lock(tile)) {
        directory_size = d_global_state_->template load_directory_size<true>();
        auto new_directory_size = directory_size + directory_delta;
        size_type copy_from = 1u << (compute_global_depth(new_directory_size) - 1);
        // copy pointers to new directory // TODO vectorize load/stores
        for (size_type bucket = directory_size; bucket < new_directory_size; bucket++) {
          d_directory_[bucket] = d_directory_[bucket - copy_from];
        }
        // publish new directory
        directory_size = new_directory_size;
        d_global_state_->template store_directory_size<true>(directory_size);
        d_global_state_->unlock(tile);
      }
    }
    suffix_type suffix_if_found(tile, allocator);
    while (true) {
      size_type bucket_index = get_bucket_index(bucket_index_hash, directory_size);
      size_type head_index = utils::memory::load<size_type, true, true>(d_directory_ + bucket_index);
      auto node = node_type(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
      node_type::lock(node.get_node_ptr(), tile);
      node.template load<true>();
      if (node.is_garbage()) {
        // this bucket just splitted by other thread; retry
        node_type::unlock(node.get_node_ptr(), tile);
        directory_size = d_global_state_->template load_directory_size<true>();
        continue;
      }
      int location_if_found = coop_traverse_until_found<true, use_hash_for_longkey>(
        node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
      if (location_if_found >= 0) { // already exists
        if (update_if_exists) {
          if (more_key) {
            suffix_if_found.update_value(value);
            suffix_if_found.store_head();
          }
          else {
            node.update(location_if_found, value);
            node.template store<true>();
          }
        }
        node_type::unlock(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
        return update_if_exists;
      }
      // not exists
      value_type to_insert = value;
      if (more_key) {
        to_insert = allocator.allocate(tile);
        auto suffix = suffix_type(
            reinterpret_cast<elem_type*>(allocator.address(to_insert)), to_insert, tile, allocator);
        static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
        suffix.create_from(key + suffix_offset, key_length - suffix_offset, value);
        suffix.store_head();
      }
      if (node.is_full()) {
        auto next_index = allocator.allocate(tile);
        auto new_node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
        new_node.initialize_empty(node.get_local_depth());
        new_node.insert(first_slice, to_insert, more_key);
        // write order: new_node -> node
        new_node.template store<true>();
        node.set_next_index(next_index);
        node.set_has_next();
      }
      else { // !node.is_full()
        node.insert(first_slice, to_insert, more_key);
      }
      node.template store<true>();
      // check if chain is too long
      if (node.get_node_ptr() != reinterpret_cast<elem_type*>(allocator.address(head_index))) {
        // check if split is possible
        auto local_depth = node.get_local_depth();
          // first bucket that points to this node
        auto first_bucket_index = bucket_index & ((1u << local_depth) - 1);
        directory_size = d_global_state_->template load_directory_size<true>();
        if ((first_bucket_index ^ (1u << local_depth)) < directory_size) {
          // do split!
          node = node_type(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
          node.template load<false>();
          // mark garbage
          node.make_garbage();
          node.template store<false>();
          // allocate two new node chains
          auto new_node0_index = allocator.allocate(tile);
          auto new_node1_index = allocator.allocate(tile);
          auto new_node0 = node_type(reinterpret_cast<elem_type*>(allocator.address(new_node0_index)), tile);
          auto new_node1 = node_type(reinterpret_cast<elem_type*>(allocator.address(new_node1_index)), tile);
          new_node0.initialize_empty(local_depth + 1);
          new_node1.initialize_empty(local_depth + 1);
          // split
          while (true) {
            for (uint32_t loc = 0; loc < node.num_keys(); loc++) {
              // decide new_node0 or new_node1
              uint32_t bucket_index_hash_at_loc;
              if (node.get_suffix_of_location(loc)) {
                auto suffix_index = node.get_value_from_location(loc);
                auto suffix = suffix_type(
                    reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
                suffix.load_head();
                bucket_index_hash_at_loc = compute_hash_for_suffix<use_hash_for_longkey>(
                    suffix, use_hash_for_longkey ? 0 : node.get_key_from_location(loc), tile);
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
                  new_node0.template store<false>();
                  new_node0 = node_type(reinterpret_cast<elem_type*>(allocator.address(new_aux_index)), tile);
                  new_node0.initialize_empty(local_depth + 1);
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
                  new_node1.template store<false>();
                  new_node1 = node_type(reinterpret_cast<elem_type*>(allocator.address(new_aux_index)), tile);
                  new_node1.initialize_empty(local_depth + 1);
                }
                new_node1.insert(node.get_key_from_location(loc),
                                 node.get_value_from_location(loc),
                                 node.get_suffix_of_location(loc));
              }
            }
            if (!node.has_next()) { break; }
            auto next_index = node.get_next_index();
            node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
            node.template load<true>();
            reclaimer.retire(next_index, tile);
          }
          // store last nodes of two new buckets
          new_node0.template store<true>();
          new_node1.template store<true>();
          // publish new buckets: 
          auto local_depth_mask = (1u << local_depth);
          d_global_state_->lock(tile);
          for (size_type index = first_bucket_index; index < directory_size; index += local_depth_mask) {
            auto new_node_index = (index & local_depth_mask) == 0 ? new_node0_index : new_node1_index;
            utils::memory::store<size_type, true, true>(d_directory_ + index, new_node_index);
          }
          d_global_state_->unlock(tile);
        }
      }
      node_type::unlock(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
      reclaimer.retire(head_index, tile);
      // increment counter
      d_global_state_->template increment_num_entries<1>(tile);
      return true;
    }
    assert(false);
  }

  template <bool do_merge, bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    key_slice_type first_slice;
    size_type bucket_index_hash;
    const bool more_key = (key_length > 1);
    if (use_hash_for_longkey && more_key) {
      auto hash = compute_hashx2(key, key_length, tile);
      bucket_index_hash = hash.x;
      first_slice = hash.y;
    }
    else {
      bucket_index_hash = compute_hash(key, key_length, tile);
      first_slice = key[0];
    }
    suffix_type suffix_if_found(tile, allocator);
    size_type bucket_index = get_bucket_index(
        bucket_index_hash, d_global_state_->template load_directory_size<true>());
    while (true) {
      size_type head_index = utils::memory::load<size_type, true, true>(d_directory_ + bucket_index);
      auto node = node_type(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
      node_type::lock(node.get_node_ptr(), tile);
      node.template load<true>();
      int location_if_found;
      if constexpr (do_merge) {
        location_if_found = coop_traverse_until_found_merge<use_hash_for_longkey>(
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator, reclaimer);
      }
      else {
        location_if_found = coop_traverse_until_found<true, use_hash_for_longkey>(
          node, first_slice, more_key, key, key_length, suffix_if_found, tile, allocator);
      }
      if (location_if_found >= 0) { // exists
        node.erase(location_if_found);
        node.template store<true>();
        if (more_key) {
          suffix_if_found.retire(reclaimer);
        }
        node_type::unlock(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
        // decrement counter
        d_global_state_->template increment_num_entries<-1>(tile);
        return true;
      }
      // not exists
      node_type::unlock(reinterpret_cast<elem_type*>(allocator.address(head_index)), tile);
      auto retry_bucket_index = get_bucket_index(
        bucket_index_hash, d_global_state_->template load_directory_size<true>());
      if (bucket_index != retry_bucket_index) {
        // If bucket index changed, retry
        bucket_index = retry_bucket_index;
        continue;
      }
      break;
    }
    return false;
  }

 private:
  // device-side helper functions
  struct __align__(128) global_state {
    size_type mutex_;
    size_type directory_size_;
    size_type num_entries_;

    DEVICE_QUALIFIER global_state(size_type initial_directory_size)
        : directory_size_(initial_directory_size)
        , mutex_(0)
        , num_entries_(0) {}
    
    template <bool atomic, bool acquire = true>
    DEVICE_QUALIFIER size_type load_directory_size() {
      return utils::memory::load<size_type, atomic, acquire>(&directory_size_);
    }
    template <bool atomic, bool release = true>
    DEVICE_QUALIFIER void store_directory_size(size_type directory_size) {
      utils::memory::store<size_type, atomic, release>(&directory_size_, directory_size);
    }
    DEVICE_QUALIFIER size_type load_num_entries() {
      return utils::memory::load<size_type, true, false>(&num_entries_);
    }
    template <int amount, typename tile_type>
    DEVICE_QUALIFIER void increment_num_entries(const tile_type& tile) {
      if (tile.thread_rank() == 0) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device> num_entries_ref(num_entries_);
        if constexpr (amount >= 0) {
          num_entries_ref.fetch_add(static_cast<size_type>(amount), cuda::memory_order_relaxed);
        }
        else {
          num_entries_ref.fetch_sub(static_cast<size_type>(-amount), cuda::memory_order_relaxed);
        }
      }
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
  };
  static DEVICE_QUALIFIER size_type compute_global_depth(size_type directory_size) {
    // smallest n of (2^n >= directory_size)
    return utils::bits::bfind(directory_size - 1) + 1;
  }
  DEVICE_QUALIFIER size_type get_bucket_index(size_type hash, size_type directory_size) {
    size_type global_depth = compute_global_depth(directory_size);
    size_type masked_hash = hash & ((1u << global_depth) - 1);
    if (masked_hash >= directory_size) {
      masked_hash ^= (1u << (global_depth - 1));
    }
    assert(masked_hash < directory_size);
    return masked_hash;
  }

  template <bool concurrent, bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER int coop_traverse_until_found(hashtable_node<tile_type>& node,
                                                 const key_slice_type& first_slice,
                                                 bool more_key,
                                                 const key_slice_type* key,
                                                 const size_type& key_length,
                                                 suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                 const tile_type& tile,
                                                 device_allocator_context_type& allocator) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
    while (true) {
      uint32_t to_check = node.match_key_in_node(first_slice, more_key);
      if (more_key) {
        // if length > 1, compare suffixes
        while (to_check != 0) {
          auto cur_location = __ffs(to_check) - 1;
          auto suffix_index = node.get_value_from_location(cur_location);
          auto suffix = suffix_type(
              reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
          suffix.load_head();
          static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
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
      node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
      node.template load<concurrent>();
    }
    // not found until the end
    return -1;
  }

  template <bool use_hash_for_longkey, typename tile_type>
  DEVICE_QUALIFIER int coop_traverse_until_found_merge(hashtable_node<tile_type>& node,
                                                       const key_slice_type& first_slice,
                                                       bool more_key,
                                                       const key_slice_type* key,
                                                       const size_type& key_length,
                                                       suffix_node<tile_type, device_allocator_context_type>& suffix_if_found,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator,
                                                       device_reclaimer_context_type& reclaimer) {
    using node_type = hashtable_node<tile_type>;
    using suffix_type = suffix_node<tile_type, device_allocator_context_type>;
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
          suffix.load_head();
          static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
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
        node.template store<true>();
        current_node_store_deferred = false;
      }
      // done searching this node, move on to next
      if (!node.has_next()) { break; }
      auto next_index = node.get_next_index();
      auto next_node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
      next_node.template load<true>();
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
  template <bool use_hash_for_longkey, typename suffix_type, typename tile_type>
  DEVICE_QUALIFIER uint32_t compute_hash_for_suffix(const suffix_type& suffix,
                                                    const key_slice_type& first_slice,
                                                    const tile_type& tile) {
    // compute polynomial
    uint32_t hash = suffix.template compute_polynomial<hash_prime0>();
    if constexpr (!use_hash_for_longkey) {
      hash = (hash * hash_prime0) + first_slice;
    }
    static constexpr uint32_t suffix_offset = use_hash_for_longkey ? 0 : 1;
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
    using node_type = hashtable_node<tile_type>;
    device_allocator_context_type allocator{allocator_, tile};
    size_type directory_size = d_global_state_->template load_directory_size<false>();
    auto global_depth = compute_global_depth(directory_size);
    for (size_type bucket_index = 0; bucket_index < directory_size; bucket_index++) {
      auto node_index = d_directory_[bucket_index];
      auto node = node_type(reinterpret_cast<elem_type*>(allocator.address(node_index)), tile);
      node.template load<false>();
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
        node = node_type(reinterpret_cast<elem_type*>(allocator.address(next_index)), tile);
        node.template load<false>();
        task.exec(node, -1, tile, allocator);
      }
    }
  }

  template <typename func>
  void traverse_nodes(func task) {
    kernel::GpuLinearHashtable::traverse_nodes_kernel<<<1, 32>>>(*this, task);
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
          auto suffix = suffix_node<tile_type, device_allocator_context_type>(
              reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
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
    using node_type = hashtable_node<tile_type>;
    // global state
    if (bucket_index == 0) {
      *d_global_state_ = global_state(directory_delta);
    }
    // allocate node
    auto node_index = allocator.allocate(tile);
    auto node = node_type(
        reinterpret_cast<elem_type*>(allocator.address(node_index)), tile);
    // initial local depth = initial global depth of initial directory size
    node.initialize_empty(compute_global_depth(directory_delta));
    node.template store<false>();
    d_directory_[bucket_index] = node_index;
  }

  void allocate() {
    is_owner_ = true;
    cuda_try(cudaMalloc(&d_directory_, sizeof(size_type) * max_directory_size));
    cuda_try(cudaMalloc(&d_global_state_, sizeof(global_state)));
    initialize();
  }

  void deallocate() {
    if (is_owner_) {
      cuda_try(cudaFree(d_directory_));
      cuda_try(cudaFree(d_global_state_));
    }
  }

  void initialize() {
    const uint32_t num_blocks = directory_delta;
    const uint32_t block_size = cg_tile_size;
    kernel::GpuLinearHashtable::initialize_kernel<<<num_blocks, block_size>>>(*this);
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

  size_type* d_directory_;
  global_state* d_global_state_;
  bool is_owner_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <typename linearhashtable>
  friend __global__ void kernel::GpuLinearHashtable::initialize_kernel(linearhashtable);

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
