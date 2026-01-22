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
#define _CG_ABI_EXPERIMENTAL  // enable experimental CGs API

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <masstree_kernels.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <masstree_node.hpp>
#include <pair_type.hpp>
#include <queue>
#include <sstream>
#include <type_traits>

#include <dynamic_stack.hpp>
#include <simple_bump_alloc.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_dummy_reclaim.hpp>
#include <simple_debra_reclaim.hpp>

namespace GpuMasstree {

template <typename Allocator,
          typename Reclaimer>
struct gpu_masstree {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  static auto constexpr branching_factor = 16;
  static auto constexpr cg_tile_size = 2 * branching_factor;
  using masstree_type = gpu_masstree<Allocator, Reclaimer>;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  using host_allocator_type = Allocator;
  using device_allocator_instance_type = typename host_allocator_type::device_instance_type;
  using device_allocator_context_type = device_allocator_context<host_allocator_type>;

  using host_reclaimer_type = Reclaimer;
  using device_reclaimer_instance_type = typename host_reclaimer_type::device_instance_type;
  using device_reclaimer_context_type = device_reclaimer_context<host_reclaimer_type>;

  gpu_masstree() = delete;
  gpu_masstree(const host_allocator_type& host_allocator,
               const host_reclaimer_type& host_reclaimer)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance()) {
    allocate();
  }

  gpu_masstree& operator=(const gpu_masstree& other) = delete;
  gpu_masstree(const gpu_masstree& other)
      : d_root_index_(other.d_root_index_)
      , is_owner_(false)
      , allocator_(other.allocator_)
      , reclaimer_(other.reclaimer_) {}

  ~gpu_masstree() {
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
    kernels::find_device_func<key_slice_type, size_type, value_type> func{
      .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths,
      .d_values = values, .concurrent = concurrent};
    launch_batch_kernel(func, num_keys, stream);
  }

  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0,
              bool update_if_exists = false) {
    kernels::insert_device_func<key_slice_type, size_type, value_type> func{
      .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths,
      .d_values = values, .update_if_exists = update_if_exists};
    launch_batch_kernel(func, num_keys, stream);
  }

  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0,
             bool do_remove_empty_root = true,
             bool do_merge = true,
             bool concurrent = true) {
    if (do_remove_empty_root) {
      kernels::erase_device_func<true, true, key_slice_type, size_type, value_type> func{
        .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths,
        .concurrent = concurrent};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      if (do_merge) {
        kernels::erase_device_func<true, false, key_slice_type, size_type, value_type> func{
          .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths,
          .concurrent = concurrent};
        launch_batch_kernel(func, num_keys, stream);
      }
      else {
        kernels::erase_device_func<false, false, key_slice_type, size_type, value_type> func{
          .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths,
          .concurrent = concurrent};
        launch_batch_kernel(func, num_keys, stream);
      }
    }
  }

  void range(const key_slice_type* lower_keys,
             const size_type* lower_key_lengths,
             const size_type max_key_length,
             const size_type max_count_per_query,
             const size_type num_queries,
             const key_slice_type* upper_keys = nullptr,
             const size_type* upper_key_lengths = nullptr,
             size_type* counts = nullptr,
             value_type* values = nullptr,
             key_slice_type* out_keys = nullptr,
             size_type* out_key_lengths = nullptr,
             cudaStream_t stream = 0,
             bool concurrent = false) {
    if (upper_keys) {
      kernels::range_device_func<true, key_slice_type, size_type, value_type> func{
        .d_lower_keys = lower_keys, .d_lower_key_lengths = lower_key_lengths,
        .max_key_length = max_key_length, .max_count_per_query = max_count_per_query,
        .d_upper_keys = upper_keys, .d_upper_key_lengths = upper_key_lengths,
        .d_counts = counts, .d_values = values, .d_out_keys = out_keys, .d_out_key_lengths = out_key_lengths,
        .concurrent = concurrent};
      launch_batch_kernel(func, num_queries, stream);
    }
    else {
      kernels::range_device_func<false, key_slice_type, size_type, value_type> func{
        .d_lower_keys = lower_keys, .d_lower_key_lengths = lower_key_lengths,
        .max_key_length = max_key_length, .max_count_per_query = max_count_per_query,
        .d_upper_keys = upper_keys, .d_upper_key_lengths = upper_key_lengths,
        .d_counts = counts, .d_values = values, .d_out_keys = out_keys, .d_out_key_lengths = out_key_lengths,
        .concurrent = concurrent};
      launch_batch_kernel(func, num_queries, stream);
    }
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
                                    bool erase_do_remove_empty_root = true,
                                    bool erase_do_merge = true) {
    kernels::insert_device_func<key_slice_type, size_type, value_type> insert_func{
      .d_keys = insert_keys, .max_key_length = max_key_length, .d_key_lengths = insert_key_lengths,
      .d_values = insert_values, .update_if_exists = insert_update_if_exists};
    if (erase_do_remove_empty_root) {
      kernels::erase_device_func<true, true, key_slice_type, size_type, value_type> erase_func{
        .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths,
        .concurrent = true};
      launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
    }
    else {
      if (erase_do_merge) {
        kernels::erase_device_func<true, false, key_slice_type, size_type, value_type> erase_func{
          .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths,
          .concurrent = true};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
      else {
        kernels::erase_device_func<false, false, key_slice_type, size_type, value_type> erase_func{
          .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths,
          .concurrent = true};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
    }
  }

  // device-side APIs
  template <typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               const size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator,
                                               bool concurrent = false) {
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length; slice++) {
      const key_slice_type key_slice = key[slice];
      const bool link_or_value = (slice < key_length - 1);
      auto border_node = coop_traverse_until_border(key_slice, current_node_index, tile, allocator, false, concurrent);
      const bool found = border_node.get_key_value_from_node(key_slice, current_node_index, link_or_value);
      if (!found) {
        // key not exists, exit early
        return invalid_value;
      }
    }
    // current_node_index has the final value
    return current_node_index;
  }

  template <bool use_upper_key, typename tile_type>
  DEVICE_QUALIFIER size_type cooperative_range(const key_slice_type* lower_key,
                                               const size_type lower_key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator,
                                               const key_slice_type* upper_key = nullptr,
                                               const size_type upper_key_length = 1,
                                               size_type out_max_count = 1,
                                               value_type* out_value = nullptr,
                                               key_slice_type* out_keys = nullptr,
                                               size_type* out_key_lengths = nullptr,
                                               const size_type out_key_max_length = 1,
                                               bool concurrent = false) {
    using node_type = masstree_node<tile_type>;
    using dynamic_stack_type_x2 = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    using dynamic_stack_type_x1 = utils::dynamic_stack_u32<1, tile_type, device_allocator_context_type>;
    if (out_max_count <= 0) { return 0; }
    assert(!use_upper_key || upper_key != nullptr);

    static constexpr key_slice_type min_key_slice = std::numeric_limits<key_slice_type>::min();
    static constexpr key_slice_type max_key_slice = std::numeric_limits<key_slice_type>::max();
    size_type current_node_index = *d_root_index_;
    size_type layer = 0, out_count = 0;
    key_slice_type lower_key_slice = lower_key[0];
    bool lower_key_lv = (lower_key_length > 1);
    [[maybe_unused]] key_slice_type upper_key_slice;
    [[maybe_unused]] bool upper_key_lv;
    [[maybe_unused]] bool ignore_upper_key;
    if constexpr (use_upper_key) {
      upper_key_slice = upper_key[0];
      upper_key_lv = (upper_key_length > 1);
      ignore_upper_key = false;
    }
    bool passed_lower_key = false;
    dynamic_stack_type_x2 key_slice_and_node_index_stack(allocator, tile);
    [[maybe_unused]] dynamic_stack_type_x1 ignore_upper_key_stack(allocator, tile);
    while (true) {
      // traverse the btree: current_node_index can be root node or border node
      auto border_node = coop_traverse_until_border(lower_key_slice, current_node_index, tile, allocator, false, concurrent, &current_node_index);
      // traverse side links, scan the nodes, and decide scan_op:
      //      >=0 (down-link location; go to next layer)
      //      -1 (continue side-link traverse)
      //      -2 (go to previous layer)
      //      -3 (end scanning the range)
      int scan_op = -1;
      while (true) {
        if (border_node.is_garbage()) { scan_op = -1; }
        else {
          // scan a node and store outputs
          uint32_t count;
          if constexpr (use_upper_key) {
            count = border_node.scan(lower_key_slice, lower_key_lv,
                                     ignore_upper_key, upper_key_slice, upper_key_lv,
                                     out_max_count, scan_op, out_value, out_keys, layer, out_key_max_length);
          }
          else {
            count = border_node.scan(lower_key_slice, lower_key_lv,
                                     out_max_count, scan_op, out_value, out_keys, layer, out_key_max_length);
          }
          assert(count <= out_max_count);
          out_count += count;
          out_max_count -= count;
          if (out_value) { out_value += count; }
          if (out_keys) {
            utils::fill_output_keys_from_key_slice_stack<0>(key_slice_and_node_index_stack, out_keys, out_key_max_length, layer, count);
            out_keys += (out_key_max_length * count);
          }
          if (out_key_lengths) {
            if (tile.thread_rank() < count) { out_key_lengths[tile.thread_rank()] = (layer + 1); }
            out_key_lengths += count;
          }
          //  if got enough outputs, end scanning
          if (out_max_count <= 0) { scan_op = -3; }
          else if (scan_op < 0) {
            // if it's end of this layer, go to prev layer
            if (!border_node.has_sibling()) { scan_op = (layer == 0) ? -3 : -2; }
            // if reached upper key, end scanning
            else if (use_upper_key && upper_key_slice <= border_node.get_high_key()) { scan_op = -3; }
            // else: continue side traversal
            else { assert(scan_op == -1); }
          }
          //  else: found down-link in range, go to next layer
        }
        if (scan_op != -1) { break; }
        current_node_index = border_node.get_sibling_index();
        border_node = node_type(
            reinterpret_cast<key_slice_type*>(allocator.address(current_node_index)),
            current_node_index,
            tile);
        if (concurrent) { border_node.load(cuda_memory_order::memory_order_relaxed); }
        else { border_node.load(); }
      }
      // switch layer
      if (scan_op >= 0) { // go to next layer
        layer++;
        // checkpoint key slice and node index to stack
        const key_slice_type checkpoint_key_slice = border_node.get_key_from_location(scan_op);
        key_slice_and_node_index_stack.push(checkpoint_key_slice, current_node_index);
        current_node_index = border_node.get_value_from_location(scan_op);
        // check if passed the lower key
        passed_lower_key = passed_lower_key ||
            (lower_key_slice < checkpoint_key_slice || layer == lower_key_length);
        // lower key for next layer
        lower_key_slice = passed_lower_key ? min_key_slice : lower_key[layer];
        lower_key_lv = !(passed_lower_key || layer == lower_key_length - 1);
        // handle upper key
        if constexpr (use_upper_key) {
          ignore_upper_key_stack.push(static_cast<size_type>(ignore_upper_key));
          // check if ignore the upper key
          ignore_upper_key = ignore_upper_key ||
              (checkpoint_key_slice < upper_key_slice || layer == upper_key_length);
          // upper key for next layer
          upper_key_slice = ignore_upper_key ? 0 : upper_key[layer];
          upper_key_lv = (layer != upper_key_length - 1);
        }
      }
      else if (scan_op == -2) { // go to prev layer
        // pop stack until checkpoint key is not 0xFFFFFFFF
        while (true) {
          if (layer == 0) {
            key_slice_and_node_index_stack.destroy();
            if constexpr (use_upper_key) { ignore_upper_key_stack.destroy(); }
            return out_count;
          }
          layer--;
          key_slice_and_node_index_stack.pop(lower_key_slice, current_node_index);
          if constexpr (use_upper_key) {
            ignore_upper_key_stack.pop(ignore_upper_key);
            assert(ignore_upper_key || layer < upper_key_length);
            if (!ignore_upper_key && lower_key_slice >= upper_key[layer]) { continue; }
          }
          if (lower_key_slice < max_key_slice) { break; }
        }
        // scan from checkpoint_key, exclusive
        lower_key_slice++;
        lower_key_lv = node_type::BORDER_ENTRY_VALUE;
        if constexpr (use_upper_key) {
          upper_key_slice = ignore_upper_key ? 0 : upper_key[layer];
          upper_key_lv = (layer != upper_key_length - 1);
          assert(ignore_upper_key || border_node.cmp_key_lv(lower_key_slice, lower_key_lv, upper_key_slice, upper_key_lv));
        }
      }
      else {
        assert(scan_op == -3);  // end scanning
        key_slice_and_node_index_stack.destroy();
        if constexpr (use_upper_key) { ignore_upper_key_stack.destroy(); }
        return out_count;
      }
    }
    assert(false);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           bool update_if_exists = false) {
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = *d_root_index_;
    size_type prev_root_index = invalid_value;
    size_type slice = 0;
    while (slice < key_length) {
      const key_slice_type key_slice = key[slice];
      const bool link_or_value = (slice < key_length - 1);
      struct split_early_exit_check {
        DEVICE_QUALIFIER bool operator()(const node_type& border_node,
                                         const key_slice_type& key_slice,
                                         bool link_or_value) const {
          // if the border node already has the key slice and it's not the last slice,
          // we just follow the same entry, no need to update the node; so we don't lock it
          return (link_or_value == node_type::BORDER_ENTRY_LINK) && border_node.key_is_in_node(key_slice, link_or_value);
        }
        DEVICE_QUALIFIER void early_exit() { early_exited_ = true; }
        bool early_exited_ = false;
      } early_exit_check;
      auto border_node = coop_traverse_until_border_split(key_slice, link_or_value, current_node_index, tile, allocator, early_exit_check);
      if (early_exit_check.early_exited_) {
        // find next layer root and continue now
        prev_root_index = current_node_index;
        border_node.get_key_value_from_node(key_slice, current_node_index, link_or_value);
        slice++;
        continue;
      }
      assert(border_node.is_locked());
      if (border_node.is_garbage()) {
        // garbage after side-traversal means (is_garbage && !has_sibling)
        // which means it's an empty root node that's collected by erasure.
        // we should retry from the previous layer
        border_node.unlock();
        if (prev_root_index == invalid_value) {
          // if it's cascading, restart from the global root
          current_node_index = *d_root_index_;
          slice = 0;
          continue;
        }
        assert(slice > 0);
        current_node_index = prev_root_index;
        prev_root_index = invalid_value;
        slice--;
        continue;
      }
      prev_root_index = current_node_index;
      if (border_node.get_key_value_from_node(key_slice, current_node_index, link_or_value)) {
        // key exists, the value is stored in current_node_index
        if (link_or_value == node_type::BORDER_ENTRY_VALUE) {
          if (update_if_exists) {
            border_node.update(key_slice, value);
          }
          else {
            // fail_if_exists
            border_node.unlock();
            return false;
          }
        }
        else {
          // continue to next layer
          border_node.unlock();
          slice++;
          continue;
        }
      }
      else {
        // key not exists
        if (link_or_value == node_type::BORDER_ENTRY_VALUE) {
          // insert value to the node
          current_node_index = value;
        }
        else {
          // allocate next layer root node and insert its index
          current_node_index = allocator.allocate(tile);
          auto next_root_node = masstree_node<tile_type>(
            reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
            current_node_index,
            tile);
          next_root_node.initialize_root();
          next_root_node.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
        }
        border_node.insert(key_slice, current_node_index, link_or_value);
      }
      border_node.store(cuda_memory_order::memory_order_relaxed);
      border_node.unlock();
      slice++;
    }
    return true;
  }

  template <bool do_merge, bool do_remove_empty_root, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer,
                                          bool concurrent = false) {
    using node_type = masstree_node<tile_type>;
    using dynamic_stack_type = utils::dynamic_stack_u32<1, tile_type, device_allocator_context_type>;
    [[maybe_unused]] dynamic_stack_type per_layer_root_indexes(allocator, tile);
    // read-only traverse before the last slice
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length - 1; slice++) {
      if constexpr (do_remove_empty_root) { per_layer_root_indexes.push(current_node_index); }
      const key_slice_type key_slice = key[slice];
      auto border_node = coop_traverse_until_border(key_slice, current_node_index, tile, allocator, false,
                                                    do_merge || do_remove_empty_root || concurrent);
      const bool found = border_node.get_key_value_from_node(key_slice, current_node_index, node_type::BORDER_ENTRY_LINK);
      if (!found) {
        // key not exists, exit early
        return false;
      }
    }
    // now erase the entry at the last_slice layer
    if constexpr (!do_merge) {
      // no-merge algorithm: just erase the element
      const key_slice_type key_slice = key[key_length - 1];
      auto border_node = coop_traverse_until_border(key_slice, current_node_index, tile, allocator, true, concurrent);
      const bool success = border_node.erase(key_slice, node_type::BORDER_ENTRY_VALUE);
      if (success) {
        border_node.store(cuda_memory_order::memory_order_relaxed);
      }
      border_node.unlock();
      return success;
    }
    else { // (do_merge)
      // merge algorithm
      key_slice_type key_slice = key[key_length - 1];
      struct merge_early_exit_check {
        DEVICE_QUALIFIER bool operator()(const node_type& border_node,
                                         const key_slice_type& key_slice,
                                         bool link_or_value) const {
          // if the border node doesn't have the key slice, no need to lock the node
          return !border_node.key_is_in_node(key_slice, link_or_value);
        }
        DEVICE_QUALIFIER void early_exit() { early_exited_ = true; }
        DEVICE_QUALIFIER void reset() { early_exited_ = false; }
        bool early_exited_ = false;
      } early_exit_check;
      bool border_node_is_root;
      auto border_node = coop_traverse_until_border_merge(key_slice, node_type::BORDER_ENTRY_VALUE, current_node_index, border_node_is_root, tile, allocator, reclaimer, early_exit_check);
      if (early_exit_check.early_exited_) {
        return false; // key not exists
      }
      assert(border_node.is_locked());
      const bool success = border_node.erase(key_slice, node_type::BORDER_ENTRY_VALUE);
      if (success) {
        border_node.store(cuda_memory_order::memory_order_relaxed);
      }
      border_node.unlock();
      if constexpr (do_remove_empty_root) {
        // collect empty roots
        int layer = static_cast<int>(key_length) - 2;
        while (layer >= 0 && border_node_is_root && border_node.num_keys() == 0) {
          key_slice = key[layer];
          per_layer_root_indexes.pop(current_node_index);
          early_exit_check.reset();
          border_node = coop_traverse_until_border_merge(key_slice, node_type::BORDER_ENTRY_LINK, current_node_index, border_node_is_root, tile, allocator, reclaimer, early_exit_check);
          if (early_exit_check.early_exited_) {
            break; // other warp removed the key
          }
          if (border_node.get_key_value_from_node(key_slice, current_node_index, node_type::BORDER_ENTRY_LINK)) {
            // TODO rewrite root check here
            // check next layer root node
            // current_node_index is next_layer_root_node_index TODO not anymore
            auto next_layer_root_node = node_type(
              reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile);
            next_layer_root_node.lock();
            next_layer_root_node.load(cuda_memory_order::memory_order_relaxed);
            if (!next_layer_root_node.is_garbage() && next_layer_root_node.num_keys() == 0) {
              // still empty, remove them
              next_layer_root_node.make_garbage_node(false);
              border_node.erase(key_slice, node_type::BORDER_ENTRY_LINK);
              next_layer_root_node.store(cuda_memory_order::memory_order_relaxed);
              __threadfence();
              border_node.store(cuda_memory_order::memory_order_relaxed);
              next_layer_root_node.unlock();
              border_node.unlock();
              reclaimer.retire(current_node_index, tile);
            }
            else {
              // other warp changed the root
              next_layer_root_node.unlock();
              border_node.unlock();
              break;
            }
          }
          else {
            // other warp changed the root
            border_node.unlock();
            break;
          }
          layer--;
        }
        per_layer_root_indexes.destroy();
      }
      return success;
    }
    assert(false);
    return false;
  }

 private:
  // device-side helper functions
  template <typename tile_type>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border(const key_slice_type& key_slice,
                                                                       const size_type& current_root_index,
                                                                       const tile_type& tile,
                                                                       device_allocator_context_type& allocator,
                                                                       bool lock_border_node,
                                                                       bool concurrent,
                                                                       size_type* node_index = nullptr) {
    // starting from a local root node in a layer, return the border node and its index
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
          current_node_index,
          tile);
      if (concurrent) {
        current_node.load(cuda_memory_order::memory_order_relaxed);
        traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);
      }
      else {
        current_node.load();
      }
      if (current_node.is_border()) {
        if (lock_border_node) {
          current_node.lock();
          current_node.load(cuda_memory_order::memory_order_relaxed);
          if (concurrent) {
            traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
          }
        }
        if (node_index != nullptr) { *node_index = current_node_index; }
        return current_node;
      }
      else {
        current_node_index = current_node.find_next(key_slice);
      }
    }
    assert(false);
  }

  template <typename tile_type, typename EarlyExitCheck>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border_split(const key_slice_type& key_slice,
                                                                             bool link_or_value,
                                                                             const size_type& current_root_index,
                                                                             const tile_type& tile,
                                                                             device_allocator_context_type& allocator,
                                                                             EarlyExitCheck& early_exit_check) {
    // starting from a local root node in a layer, return the LOCKED border node and its index
    // proactively split full nodes while traversal. also the returned border node is not full.
    // if early exit condition is met, returned node is not locked by this warp (might locked by another)
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    size_type parent_index = current_root_index;
    while (true) {
      auto current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
          current_node_index,
          tile);
      current_node.load(cuda_memory_order::memory_order_relaxed);
      bool link_traversed = traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit_check(current_node, key_slice, link_or_value)) {
        early_exit_check.early_exit();
        return current_node;
      }

      // lock the node & traverse again, if it's full or border
      // if it's full, the parent should be known
      if (current_node.is_full() || current_node.is_border()) {
        if (current_node.try_lock()) {
          current_node.load(cuda_memory_order::memory_order_relaxed);
          if (!current_node.is_full()) {
            link_traversed |= traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
          }
          if (current_node.is_full()) {
            // if parent is unknown, restart from root
            if (current_node_index != current_root_index &&
                (current_node_index == parent_index || link_traversed || current_node.traverse_required(key_slice))) {
              current_node.unlock();
              current_node_index = current_root_index;
              parent_index = current_root_index;
              continue;
            }
            assert(!current_node.traverse_required(key_slice));
          }
          else if (!current_node.is_border()) {
            // lock not needed anymore
            current_node.unlock();
          }
        }
        else {
          // try_lock failed, retry from parent
          current_node_index = parent_index;
          continue;
        }
      }
      assert((current_node.is_full() || current_node.is_border()) ? 
             (current_node.is_locked() && !current_node.traverse_required(key_slice)) : true);
      assert(current_node.is_full() ?
             (current_node_index == current_root_index || (current_node_index != parent_index && !link_traversed)) : true);

      // if the node is full, split. it's already locked if it's full.
      if (current_node.is_full()) {
        if (current_node_index != current_root_index) {
          assert(!current_node.is_root());
          auto parent_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(parent_index)), parent_index, tile);
          parent_node.lock();
          parent_node.load(cuda_memory_order::memory_order_relaxed);
          // parent should be not full, not garbage, and correct parent
          if (parent_node.is_full() ||
              parent_node.is_garbage() ||
              !parent_node.ptr_is_in_node(current_node_index)) {
            current_node.unlock();
            parent_node.unlock();
            current_node_index = current_root_index;
            parent_index = current_root_index;
            continue;
          }
          // do split
          auto sibling_index = allocator.allocate(tile);
          auto split_result = current_node.split(sibling_index,
                                                 parent_index,
                                                 reinterpret_cast<elem_type*>(allocator.address(sibling_index)),
                                                 parent_node);
          // write order: right -> left -> parent
          split_result.sibling.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          current_node.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          parent_node.store(cuda_memory_order::memory_order_relaxed);
          parent_node.unlock();
          // update current node if necessary
          if (current_node.key_is_in_upperhalf(split_result.pivot_key, key_slice)) {
            current_node.unlock();
            current_node_index = sibling_index;
            current_node = split_result.sibling;
          }
          else {
            split_result.sibling.unlock();
          }
        }
        else { // (current_node_index == root_node_index)
          assert(current_node.is_root());
          auto left_sibling_index = allocator.allocate(tile);
          auto right_sibling_index = allocator.allocate(tile);
          auto two_siblings = current_node.split_as_root(left_sibling_index,
                                                         right_sibling_index,
                                                         reinterpret_cast<elem_type*>(allocator.address(left_sibling_index)),
                                                         reinterpret_cast<elem_type*>(allocator.address(right_sibling_index)));
          // write order: right -> left -> parent
          two_siblings.right.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          two_siblings.left.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          current_node.store(cuda_memory_order::memory_order_relaxed);
          current_node.unlock();
          // update current node to left or right
          current_node_index = current_node.find_next(key_slice);
          if (current_node_index == left_sibling_index) {
            two_siblings.right.unlock();
            current_node = two_siblings.left;
          }
          else {
            two_siblings.left.unlock();
            current_node = two_siblings.right;
          }
          parent_index = current_root_index;
        }
        // now, current_node is not full. if it's not border, unlock.
        if (!current_node.is_border()) { current_node.unlock(); }
      }
      assert(!current_node.is_full());

      // now, the node is not full; if border it's locked, otherwise not locked.
      // traversal or insert
      if (current_node.is_border()) {
        return current_node;
      } else {  // traverse
        parent_index = current_node_index;
        current_node_index = current_node.find_next(key_slice);
      }
    }
    assert(false);
  }

  template <typename tile_type, typename EarlyExitCheck>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border_merge(const key_slice_type& key_slice,
                                                                             bool link_or_value,
                                                                             const size_type& current_root_index,
                                                                             bool& output_node_is_root,
                                                                             const tile_type& tile,
                                                                             device_allocator_context_type& allocator,
                                                                             device_reclaimer_context_type& reclaimer,
                                                                             EarlyExitCheck& early_exit_check) {
    // starting from a local root node in a layer, return the LOCKED border node and its index
    // proactively merge/borrow underflow nodes while traversal. also the returned border node is not underflow.
    // if early exit condition is met, returned node is not locked by this warp (might locked by another)
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    size_type parent_index = current_root_index;
    size_type sibling_index = current_root_index;
    bool sibling_at_left = false;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
          current_node_index,
          tile);
      current_node.load(cuda_memory_order::memory_order_relaxed);
      bool link_traversed = traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit_check(current_node, key_slice, link_or_value)) {
        early_exit_check.early_exit();
        return current_node;
      }

      // lock the node & traverse again, if it's underflow or border
      // if it's underflow, the parent and sibling should be known
      if (current_node.is_underflow() || current_node.is_border()) {
        if (current_node.try_lock()) {
          current_node.load(cuda_memory_order::memory_order_relaxed);
          if (!current_node.is_underflow()) {
            link_traversed |= traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
          }
          if (current_node.is_underflow()) {
            // if parent is unknown, restart from root
            if (current_node_index != current_root_index &&
                (current_node_index == parent_index || sibling_index == current_root_index ||
                 link_traversed || current_node.traverse_required(key_slice))) {
              current_node.unlock();
              current_node_index = current_root_index;
              parent_index = current_root_index;
              sibling_index = current_root_index;
              continue;
            }
            assert(!current_node.traverse_required(key_slice));
          }
          else if (!current_node.is_border()) {
            // lock not needed anymore
            current_node.unlock();
          }
        }
        else {
          // try_lock failed, retry from parent
          current_node_index = parent_index;
          sibling_index = current_root_index;
          continue;
        }
      }
      assert((current_node.is_underflow() || current_node.is_border()) ? 
             (current_node.is_locked() && !current_node.traverse_required(key_slice)) : true);
      assert(current_node.is_underflow() ?
             (current_node_index == current_root_index || (current_node_index != parent_index && sibling_index != current_root_index && !link_traversed)) : true);


      // proactively merge/borrow underflow nodes
      if (current_node.is_underflow()) {
        // lock the sibling first
        auto sibling_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(sibling_index)), sibling_index, tile);
        if (sibling_at_left) {
          sibling_node.lock();
        }
        else {
          // global lock order: right -> left; use try_lock to avoid deadlock
          if (!sibling_node.try_lock()) {
            current_node.unlock();
            current_node_index = parent_index;
            sibling_index = current_root_index;
            continue;
          }
        }
        sibling_node.load(cuda_memory_order::memory_order_relaxed);
        // check sibling validity
        if (sibling_node.is_garbage() ||
            (sibling_at_left ?
             (sibling_node.get_sibling_index() != current_node_index) :
             (current_node.get_sibling_index() != sibling_index))) {
          current_node.unlock();
          sibling_node.unlock();
          current_node_index = current_root_index;
          parent_index = current_root_index;
          sibling_index = current_root_index;
          continue;
        }
        // lock the parent
        auto parent_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(parent_index)), parent_index, tile);
        parent_node.lock();
        parent_node.load(cuda_memory_order::memory_order_relaxed);
        // make sure parent is not garbage and not underflow
        if (parent_node.is_garbage() || parent_node.is_underflow()) {
          current_node.unlock();
          sibling_node.unlock();
          parent_node.unlock();
          current_node_index = current_root_index;
          parent_index = current_root_index;
          sibling_index = current_root_index;
          continue;
        }
        // make sure parent is correct parent for both children
        auto plan = parent_node.get_merge_plan(current_node_index);
        if ((plan.left_location < 0) ||
            (plan.sibling_index != sibling_index)) {
          current_node.unlock();
          sibling_node.unlock();
          parent_node.unlock();
          current_node_index = current_root_index;
          parent_index = current_root_index;
          sibling_index = current_root_index;
          continue;
        }
        // now all three nodes are locked
        if (current_node.is_mergeable(sibling_node)) {
          // merge
          auto& left_sibling_node = plan.sibling_at_left ? sibling_node : current_node;
          auto& right_sibling_node = plan.sibling_at_left ? current_node : sibling_node;
          if (parent_index != current_root_index || parent_node.num_keys() > 2) {
            auto left_sibling_index = plan.sibling_at_left ? plan.sibling_index : current_node_index;
            left_sibling_node.merge(left_sibling_index, right_sibling_node, parent_node, plan.left_location);
            left_sibling_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            right_sibling_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            parent_node.store(cuda_memory_order::memory_order_relaxed);
            parent_node.unlock();
            right_sibling_node.unlock();
            auto right_sibling_index = plan.sibling_at_left ? current_node_index : plan.sibling_index;
            reclaimer.retire(right_sibling_index, tile);
            if (plan.sibling_at_left) {
              current_node_index = plan.sibling_index;
              current_node = sibling_node;
            }
          }
          else {
            parent_node.merge_to_root(current_root_index, left_sibling_node, right_sibling_node);
            parent_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            left_sibling_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            right_sibling_node.store(cuda_memory_order::memory_order_relaxed);
            left_sibling_node.unlock();
            right_sibling_node.unlock();
            reclaimer.retire(current_node_index, tile);
            reclaimer.retire(plan.sibling_index, tile);
            current_node_index = current_root_index;
            current_node = parent_node;
          }
        }
        else {
          // borrow
          if (plan.sibling_at_left) {
            current_node.borrow_left(sibling_node, parent_node, plan.left_location);
            current_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            sibling_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            parent_node.store(cuda_memory_order::memory_order_relaxed);
          }
          else {
            // borrow_right need additional node to ensure correct lock-free traversal
            auto new_sibling_index = allocator.allocate(tile);
            auto new_sibling_node = node_type(
                reinterpret_cast<elem_type*>(allocator.address(new_sibling_index)),
                new_sibling_index,
                tile);
            current_node.borrow_right(sibling_node,
                                      parent_node,
                                      plan.left_location,
                                      current_node_index,
                                      new_sibling_index,
                                      new_sibling_node);
            new_sibling_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            current_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            sibling_node.store(cuda_memory_order::memory_order_relaxed);
            __threadfence();
            parent_node.store(cuda_memory_order::memory_order_relaxed);
            new_sibling_node.unlock();
            reclaimer.retire(sibling_index, tile);
          }
          parent_node.unlock();
          sibling_node.unlock();
        }
        // now, current_node is not underflow. if it's not border, unlock.
        if (!current_node.is_border()) { current_node.unlock(); }
      }
      // we allow border node to be underflow (for better remove-empty-root algorithm)
      //assert(current_node.is_border() || !current_node.is_underflow()); TODO
      assert(!current_node.is_underflow());

      // now, the node is not underflow; if border it's locked, otherwise not locked.
      // traversal or erase
      if (current_node.is_border()) {
        output_node_is_root = (current_node_index == current_root_index);
        return current_node;
      }
      else { // traverse
        parent_index = current_node_index;
        current_node_index = current_node.find_next_and_sibling(
            key_slice, sibling_index, sibling_at_left);
      }
    }
    assert(false);
  }

 public:
  // device-side debug functions
  template <typename tile_type>
  DEVICE_QUALIFIER void cooperative_debug_find_varlen_print(
      const key_slice_type* key, const size_type key_length, tile_type& tile) {
    using node_type = masstree_node<tile_type>;
    device_allocator_context_type allocator{allocator_, tile};
    size_type current_node_index = *d_root_index_;
    bool lead_lane = tile.thread_rank() == 0;
    if (lead_lane) printf("coop find print\n");
    for (size_type slice = 0; slice < key_length; slice++) {
      const key_slice_type key_slice = key[slice];
      const bool last_slice = (slice == key_length - 1);
      if (lead_lane) printf("key[%u]: %u%s\n", slice, key_slice, last_slice ? " (last)" : "");
      while (true) {
        node_type current_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
            current_node_index,
            tile);
        current_node.load(cuda_memory_order::memory_order_seq_cst);
        current_node.print();
        if (current_node.is_border()) {
          const bool found = current_node.get_key_value_from_node(key_slice, current_node_index, last_slice);
          if (!found) {
            // not exists
            if (lead_lane) printf("value not found, exit\n");
            return;
          }
          else {
            // value retrieved in current_node_index. continue to next layer.
            if (lead_lane) printf("move to next layer\n");
            break;
          }
        }
        else {
          current_node_index = current_node.find_next(key_slice);
          if (lead_lane) printf("proceed to next node in layer\n");
        }
      }
    }
  }

  void debug_find_varlen_print(key_slice_type* key, size_type* length) {
    debug_find_varlen_print_kernel<<<1, 32>>>(*this, key, length);
  }

  template <typename T, int MAX_SIZE>
  struct traversal_stack {
    DEVICE_QUALIFIER traversal_stack(T* shared_stack, T* shared_meta)
      : stack_(shared_stack), meta_(shared_meta), size_(0) {}
    DEVICE_QUALIFIER bool empty() { return size_ == 0; }
    DEVICE_QUALIFIER bool full() { return size_ == MAX_SIZE; }
    DEVICE_QUALIFIER void push(T value, T metadata) {
      assert(size_ < MAX_SIZE);
      stack_[size_] = value;
      meta_[size_] = metadata;
      size_++;
      __syncthreads();
    }
    DEVICE_QUALIFIER T top() {
      assert(size_ > 0);
      return stack_[size_ - 1];
    }
    DEVICE_QUALIFIER T& top_metadata() {
      assert(size_ > 0);
      return meta_[size_ - 1];
    }
    DEVICE_QUALIFIER void pop() {
      assert(size_ > 0);
      size_--;
    }
    T* stack_;
    T* meta_;
    uint32_t size_;
  };

  template <int MAX_STACK_SIZE, typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_tree_nodes(uint32_t* shared_stack,
                                                        uint32_t* shared_metadata,
                                                        const tile_type& tile,
                                                        Func& task) {
    // debug-purpose, so inefficient implementation
    // called with single warp, BFS
    using node_type         = masstree_node<tile_type>;
    device_allocator_context_type allocator{allocator_, tile};
    // stack: stores node indexes. metadata: # of traversed children
    traversal_stack<uint32_t, MAX_STACK_SIZE> stack(shared_stack, shared_metadata);
    stack.push(*d_root_index_, 0);
    while (!stack.empty()) {
      uint32_t current_node_index = stack.top();
      uint32_t num_traversed_children = stack.top_metadata();
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
          current_node_index,
          tile);
      current_node.load();
      if (num_traversed_children == 0) {
        // first time visiting
        task.exec(current_node);
      }
      if (current_node.num_keys() == num_traversed_children) {
        // done traversing this node
        stack.pop();
      }
      else {
        // num_traversed_children++
        stack.top_metadata()++;
        if ((!current_node.is_border()) ||
            (current_node.is_location_link_or_value(num_traversed_children) == node_type::BORDER_ENTRY_LINK)) {
          // If it's a interior node, push the next node
          // If it's a border node but link entry, push the next layer root
          stack.push(current_node.get_value_from_location(num_traversed_children), 0);
        }
      }
    }
  }

  struct print_node_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {}
    template <typename node_type>
    DEVICE_QUALIFIER void exec(const node_type& node) {
      node.print();
    }
    DEVICE_QUALIFIER void fini() {}
  };
  void print_tree_nodes() {
    std::cout << "===== masstree.print_tree_nodes =====" << std::endl;
    traverse_tree_nodes_kernel<200, print_node_task><<<1, 32>>>(*this);
    cudaDeviceSynchronize();
  }

  struct validate_tree_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {
      lead_lane_ = lead_lane;
      num_entries_ = 0;
      num_nodes_ = 0;
    }
    template <typename node_type>
    DEVICE_QUALIFIER void exec(const node_type& node) {
      uint32_t num_entries = 0;
      if (node.is_border()) {
        uint16_t num_keys = node.num_keys();
        key_slice_type before_key = 0;
        bool before_link_or_value = false;
        for (uint16_t i = 0; i < num_keys; i++) {
          auto key = node.get_key_from_location(i);
          auto link_or_value = node.is_location_link_or_value(i);
          if (i > 0) {
            assert(before_key <= key);
            if (before_key == key) {
              assert((before_link_or_value == node_type::BORDER_ENTRY_VALUE && 
                      link_or_value == node_type::BORDER_ENTRY_LINK));
            }
          }
          before_key = key;
          before_link_or_value = link_or_value;
          if (node.is_location_link_or_value(i) == node_type::BORDER_ENTRY_VALUE) {
            num_entries++;
          }
        }
      }
      num_entries_ += num_entries;
      num_nodes_++;
    }
    DEVICE_QUALIFIER void fini() {
      if (lead_lane_) {
        printf("%lu entries, %lu nodes found\n", num_entries_, num_nodes_);
      }
    }
    bool lead_lane_;
    uint64_t num_entries_, num_nodes_;
  };
  void validate_tree() {
    traverse_tree_nodes_kernel<200, validate_tree_task><<<1, 32>>>(*this);
    cudaDeviceSynchronize();
  }

 private:

  // Tries to traverse the side-links without locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type>
  DEVICE_QUALIFIER bool traverse_side_links(node_type& node,
                                            size_type& node_index,
                                            const key_slice_type& key_slice,
                                            const tile_type& tile,
                                            device_allocator_context_type& allocator) {
    bool traversed = false;
    while (node.traverse_required(key_slice)) {
      node_index = node.get_sibling_index();
      node =
          node_type(reinterpret_cast<key_slice_type*>(allocator.address(node_index)), node_index, tile);
      node.load(cuda_memory_order::memory_order_relaxed);
      traversed |= true;
    }
    return traversed;
  }

  // Tries to traverse the side-links with locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type>
  DEVICE_QUALIFIER bool traverse_side_links_with_locks(node_type& node,
                                                       size_type& node_index,
                                                       const key_slice_type& key_slice,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator) {
    bool traversed = false;
    while (node.traverse_required(key_slice)) {
      node_index = node.get_sibling_index();
      node_type sibling_node =
          node_type(reinterpret_cast<key_slice_type*>(allocator.address(node_index)), node_index, tile);
      node.unlock();
      sibling_node.lock();
      node = sibling_node;
      node.load(cuda_memory_order::memory_order_relaxed);
      traversed |= true;
    }
    return traversed;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void allocate_root_node(const tile_type& tile, device_allocator_context_type& allocator) {
    auto root_index = allocator.allocate(tile);
    *d_root_index_ = root_index;
    using node_type = masstree_node<tile_type>;

    auto root_node =
        node_type(reinterpret_cast<elem_type*>(allocator.address(root_index)),
                  root_index,
                  tile);
    root_node.initialize_root();
    root_node.store();
  }

  void allocate() {
    is_owner_ = true;
    cuda_try(cudaMalloc(&d_root_index_, sizeof(size_type)));
    cuda_try(cudaMemset(d_root_index_, 0x00, sizeof(size_type)));
    initialize();
  }

  void deallocate() {
    if (is_owner_) {
      cuda_try(cudaFree(d_root_index_));
    }
  }

  void initialize() {
    const uint32_t num_blocks = 1;
    const uint32_t block_size = cg_tile_size;
    kernels::initialize_kernel<<<num_blocks, block_size>>>(*this);
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
      kernels::batch_kernel<do_reclaim, device_func, masstree_type>,
      block_size,
      shmem_size);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    uint32_t num_blocks = num_blocks_per_sm * device_prop.multiProcessorCount;

    kernels::batch_kernel<do_reclaim><<<num_blocks, block_size, shmem_size, stream>>>(
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
      kernels::batch_concurrent_two_funcs_kernel<do_reclaim, device_func0, device_func1, masstree_type>,
      block_size,
      shmem_size);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    uint32_t num_blocks = num_blocks_per_sm * device_prop.multiProcessorCount;
    
    kernels::batch_concurrent_two_funcs_kernel<do_reclaim><<<num_blocks, block_size, shmem_size, stream>>>(
        *this, func0, num_requests0, func1, num_requests1);
  }

  size_type* d_root_index_;
  bool is_owner_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <typename masstree>
  friend __global__ void kernels::initialize_kernel(masstree);

  template <bool do_reclaim, typename device_func, typename masstree>
  friend __global__ void kernels::batch_kernel(masstree tree,
                                               const device_func func,
                                               uint32_t num_requests);

  template <bool do_reclaim, typename device_func0, typename device_func1, typename masstree>
  friend __global__ void kernels::batch_concurrent_two_funcs_kernel(masstree tree,
                                                                    const device_func0 func0,
                                                                    uint32_t num_requests0,
                                                                    const device_func1 func1,
                                                                    uint32_t num_requests1);

}; // struct gpu_masstree

} // namespace GPUMasstree
