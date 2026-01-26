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
#include <masstree_suffix.hpp>
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
    using find_concurrent = kernels::find_device_func<true, key_slice_type, size_type, value_type>;
    using find_readonly = kernels::find_device_func<false, key_slice_type, size_type, value_type>;
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
              bool update_if_exists = false,
              bool enable_suffix = true) {
    using insert_suffix = kernels::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_nosuffix = kernels::insert_device_func<false, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists
    if (enable_suffix) {
      insert_suffix func{insert_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      insert_nosuffix func{insert_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    #undef insert_args
  }

  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0,
             bool do_remove_empty_root = true,
             bool do_merge = true,
             bool concurrent = true) {
    using erase_readonly = kernels::erase_device_func<false, false, false, key_slice_type, size_type, value_type>;
    using erase_concurrent = kernels::erase_device_func<true, false, false, key_slice_type, size_type, value_type>;
    using erase_merge = kernels::erase_device_func<true, true, false, key_slice_type, size_type, value_type>;
    using erase_rmroot = kernels::erase_device_func<true, true, true, key_slice_type, size_type, value_type>;
    #define erase_args .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths
    if (do_remove_empty_root) {
      erase_rmroot func{erase_args};
      launch_batch_kernel(func, num_keys, stream);
    }
    else {
      if (do_merge) {
        erase_merge func{erase_args};
        launch_batch_kernel(func, num_keys, stream);
      }
      else {
        if (concurrent) {
          erase_concurrent func{erase_args};
          launch_batch_kernel(func, num_keys, stream);
        }
        else {
          erase_readonly func{erase_args};
          launch_batch_kernel(func, num_keys, stream);
        }
      }
    }
    #undef erase_args
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
    using range_bothbound_concurrent = kernels::range_device_func<true, true, key_slice_type, size_type, value_type>;
    using range_bothbound_readonly = kernels::range_device_func<true, false, key_slice_type, size_type, value_type>;
    using range_loweronly_concurrent = kernels::range_device_func<false, true, key_slice_type, size_type, value_type>;
    using range_loweronly_readonly = kernels::range_device_func<false, false, key_slice_type, size_type, value_type>;
    #define range_args .d_lower_keys = lower_keys, .d_lower_key_lengths = lower_key_lengths,  \
                       .max_key_length = max_key_length, .max_count_per_query = max_count_per_query,  \
                       .d_upper_keys = upper_keys, .d_upper_key_lengths = upper_key_lengths, \
                       .d_counts = counts, .d_values = values, .d_out_keys = out_keys, .d_out_key_lengths = out_key_lengths
    if (upper_keys) {
      if (concurrent) {
        range_bothbound_concurrent func{range_args};
        launch_batch_kernel(func, num_queries, stream);
      }
      else {
        range_bothbound_readonly func{range_args};
        launch_batch_kernel(func, num_queries, stream);
      }
    }
    else {
      if (concurrent) {
        range_loweronly_concurrent func{range_args};
        launch_batch_kernel(func, num_queries, stream);
      }
      else {
        range_loweronly_readonly func{range_args};
        launch_batch_kernel(func, num_queries, stream);
      }
    }
    #undef range_args
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
                                    bool insert_enable_suffix = true,
                                    bool erase_do_remove_empty_root = true,
                                    bool erase_do_merge = true) {
    using insert_suffix = kernels::insert_device_func<true, key_slice_type, size_type, value_type>;
    using insert_nosuffix = kernels::insert_device_func<false, key_slice_type, size_type, value_type>;
    using erase_concurrent = kernels::erase_device_func<true, false, false, key_slice_type, size_type, value_type>;
    using erase_merge = kernels::erase_device_func<true, true, false, key_slice_type, size_type, value_type>;
    using erase_rmroot = kernels::erase_device_func<true, true, true, key_slice_type, size_type, value_type>;
    #define insert_args .d_keys = insert_keys, .max_key_length = max_key_length, .d_key_lengths = insert_key_lengths, .d_values = insert_values, .update_if_exists = insert_update_if_exists
    #define erase_args .d_keys = erase_keys, .max_key_length = max_key_length, .d_key_lengths = erase_key_lengths
    if (erase_do_remove_empty_root) {
      if (insert_enable_suffix) {
        insert_suffix insert_func{insert_args};
        erase_rmroot erase_func{erase_args};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
      else {
        insert_nosuffix insert_func{insert_args};
        erase_rmroot erase_func{erase_args};
        launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
      }
    }
    else {
      if (erase_do_merge) {
        if (insert_enable_suffix) {
          insert_suffix insert_func{insert_args};
          erase_merge erase_func{erase_args};
          launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
        }
        else {
          insert_nosuffix insert_func{insert_args};
          erase_merge erase_func{erase_args};
          launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
        }
      }
      else {
        if (insert_enable_suffix) {
          insert_suffix insert_func{insert_args};
          erase_concurrent erase_func{erase_args};
          launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
        }
        else {
          insert_nosuffix insert_func{insert_args};
          erase_concurrent erase_func{erase_args};
          launch_batch_concurrent_two_funcs_kernel(insert_func, insert_num_keys, erase_func, erase_num_keys, stream);
        }
      }
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
    using node_type = masstree_node<tile_type>;
    using suffix_type = masstree_suffix_node<tile_type, device_allocator_context_type>;
    static constexpr auto memory_order = concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak;
    size_type current_node_index = *d_root_index_;
    size_type slice = 0;
    while (slice < key_length) {
      const key_slice_type key_slice = key[slice];
      const bool more_key = (slice < key_length - 1);
      auto border_node = coop_traverse_until_border<concurrent>(key_slice, current_node_index, tile, allocator, false);
      const int found_keystate = border_node.get_key_value_from_node(key_slice, current_node_index, more_key);
      if (found_keystate < 0) {
        // key not exists, exit early
        return invalid_value;
      }
      if (found_keystate == node_type::KEYSTATE_SUFFIX) {
        auto suffix = suffix_type(
            reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile, allocator);
        suffix.template load_head<memory_order>();
        const bool suffix_eq = suffix.template streq<memory_order>(key + slice, key_length - slice);
        return suffix_eq ? suffix.get_value() : invalid_value;
      }
      else {  // keystate == LINK or VALUE
        slice++;
      }
    }
    // current_node_index has the final value
    return current_node_index;
  }

  template <bool use_upper_key, bool concurrent, typename tile_type>
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
                                               const size_type out_key_max_length = 1) {
    using node_type = masstree_node<tile_type>;
    using dynamic_stack_type_x2 = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    using dynamic_stack_type_x1 = utils::dynamic_stack_u32<1, tile_type, device_allocator_context_type>;
    static constexpr auto memory_order = concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak;
    if (out_max_count <= 0) { return 0; }
    assert(!use_upper_key || upper_key != nullptr);

    static constexpr key_slice_type min_key_slice = std::numeric_limits<key_slice_type>::min();
    static constexpr key_slice_type max_key_slice = std::numeric_limits<key_slice_type>::max();
    size_type current_node_index = *d_root_index_;
    size_type layer = 0, out_count = 0;
    key_slice_type lower_key_slice = lower_key[0];
    bool lower_key_more = (lower_key_length > 1);
    [[maybe_unused]] key_slice_type upper_key_slice;
    [[maybe_unused]] bool upper_key_more;
    [[maybe_unused]] bool ignore_upper_key;
    if constexpr (use_upper_key) {
      upper_key_slice = upper_key[0];
      upper_key_more = (upper_key_length > 1);
      ignore_upper_key = false;
    }
    bool passed_lower_key = false;
    dynamic_stack_type_x2 key_slice_and_node_index_stack(allocator, tile);
    [[maybe_unused]] dynamic_stack_type_x1 ignore_upper_key_stack(allocator, tile);
    while (true) {
      // traverse the btree: current_node_index can be root node or border node
      auto border_node = coop_traverse_until_border<concurrent>(lower_key_slice, current_node_index, tile, allocator, false, &current_node_index);
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
            count = border_node.template scan<memory_order>(lower_key_slice, lower_key_more,
                                                            ignore_upper_key, upper_key_slice, upper_key_more,
                                                            out_max_count, scan_op, out_value, out_keys,
                                                            layer, out_key_max_length);
          }
          else {
            count = border_node.template scan<memory_order>(lower_key_slice, lower_key_more,
                                                            out_max_count, scan_op, out_value, out_keys,
                                                            layer, out_key_max_length);
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
        border_node.template load<memory_order>();
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
        lower_key_more = !(passed_lower_key || layer == lower_key_length - 1);
        // handle upper key
        if constexpr (use_upper_key) {
          ignore_upper_key_stack.push(static_cast<size_type>(ignore_upper_key));
          // check if ignore the upper key
          ignore_upper_key = ignore_upper_key ||
              (checkpoint_key_slice < upper_key_slice || layer == upper_key_length);
          // upper key for next layer
          upper_key_slice = ignore_upper_key ? 0 : upper_key[layer];
          upper_key_more = (layer != upper_key_length - 1);
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
        lower_key_more = false;
        if constexpr (use_upper_key) {
          upper_key_slice = ignore_upper_key ? 0 : upper_key[layer];
          upper_key_more = (layer != upper_key_length - 1);
          assert(ignore_upper_key || border_node.cmp_key(lower_key_slice, lower_key_more, upper_key_slice, upper_key_more));
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

  template <bool enable_suffix, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           device_reclaimer_context_type& reclaimer,
                                           bool update_if_exists = false) {
    using node_type = masstree_node<tile_type>;
    using suffix_type = masstree_suffix_node<tile_type, device_allocator_context_type>;
    size_type current_node_index = *d_root_index_;
    size_type prev_root_index = invalid_value;
    size_type slice = 0;
    while (slice < key_length) {
      const key_slice_type key_slice = key[slice];
      const bool more_key = (slice < key_length - 1);
      struct split_early_exit_check {
        DEVICE_QUALIFIER bool operator()(const node_type& border_node,
                                         const key_slice_type& key_slice,
                                         bool more_key) const {
          // if the border node already has the key slice and it's not the last slice,
          // we just follow the same entry, no need to update the node; so we don't lock it
          return more_key && border_node.key_is_in_node(key_slice, node_type::KEYSTATE_LINK);
        }
        DEVICE_QUALIFIER void early_exit() { early_exited_ = true; }
        bool early_exited_ = false;
      } early_exit_check;
      auto border_node = coop_traverse_until_border_split(key_slice, more_key, current_node_index, tile, allocator, early_exit_check);
      if (early_exit_check.early_exited_) {
        // find next layer root and continue now
        prev_root_index = current_node_index;
        border_node.get_key_value_from_node(key_slice, current_node_index, more_key);
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
      auto keystate = border_node.get_key_value_from_node(key_slice, current_node_index, more_key);
      if (keystate >= 0) {
        // key exists, the value is stored in current_node_index
        if (keystate == node_type::KEYSTATE_LINK) {
          // continue to next layer
          border_node.unlock();
          slice++;
          continue;
        }
        if (keystate == node_type::KEYSTATE_VALUE) {
          // update or fail
          if (update_if_exists) {
            border_node.update(key_slice, value, node_type::KEYSTATE_VALUE, node_type::KEYSTATE_VALUE);
          }
          else { // fail_if_exists
            border_node.unlock();
            return false;
          }
        }
        else { // node_type::KEYSTATE_SUFFIX
          auto suffix = suffix_type(
              reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile, allocator);
          suffix.template load_head<cuda_memory_order::relaxed>();
          key_slice_type mismatch_suffix_slice;
          int cmp = suffix.template strcmp<cuda_memory_order::relaxed>(key + slice, key_length - slice, &mismatch_suffix_slice);
          if (cmp == 0) { // already exists
            if (update_if_exists) {
              // protected by border_node.lock()
              suffix.update_value(value);
              suffix.template store_head<cuda_memory_order::relaxed>();
              border_node.unlock();
              return true;
            }
            else {  // fail_if_exists
              border_node.unlock();
              return false;
            }
          }
          else {  // mismatch: create new node chain
            int num_matches = abs(cmp) - 1;
            auto suffix_index = current_node_index;
            current_node_index = allocator.allocate(tile);
            border_node.update(key_slice, current_node_index, node_type::KEYSTATE_SUFFIX, node_type::KEYSTATE_LINK);
            // chain of singleton nodes for matching prefix
            for (int i = 0; i < num_matches; i++) {
              slice++;
              auto singleton_node = node_type(
                  reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile);
              singleton_node.initialize_root();
              current_node_index = allocator.allocate(tile);
              singleton_node.insert(key[slice], current_node_index, node_type::KEYSTATE_LINK);
              singleton_node.template store<cuda_memory_order::relaxed>();
            }
            slice++;
            // one diverging node with two entries
            auto doubleton_node = node_type(
                reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile);
            doubleton_node.initialize_root();
            // insert suffix of suffix key
            assert(num_matches < suffix.get_key_length());
            if (num_matches == suffix.get_key_length() - 1) {
              doubleton_node.insert(mismatch_suffix_slice, suffix.get_value(), node_type::KEYSTATE_VALUE);
              suffix.template retire<cuda_memory_order::relaxed>(reclaimer, suffix_index);
            }
            else {
              suffix.template trim<cuda_memory_order::relaxed>(num_matches + 1, suffix_index, reclaimer);
              suffix.template store_head<cuda_memory_order::relaxed>();
              doubleton_node.insert(mismatch_suffix_slice, suffix_index, node_type::KEYSTATE_SUFFIX);
            }
            // insert suffix of this key
            assert(slice < key_length);
            if (slice == key_length - 1) {
              doubleton_node.insert(key[slice], value, node_type::KEYSTATE_VALUE);
            }
            else {
              current_node_index = allocator.allocate(tile);
              suffix = suffix_type(
                  reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile, allocator);
              suffix.template create_from<cuda_memory_order::relaxed>(key + slice, key_length - slice, value);
              suffix.template store_head<cuda_memory_order::relaxed>();
              doubleton_node.insert(key[slice], current_node_index, node_type::KEYSTATE_SUFFIX);
            }
            doubleton_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
          }
        }
      }
      else {
        // key not exists
        if (more_key) {
          if constexpr (enable_suffix) {
            // insert suffix entry
            current_node_index = allocator.allocate(tile);
            auto suffix = suffix_type(
                reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile, allocator);
            suffix.template create_from<cuda_memory_order::relaxed>(key + slice, key_length - slice, value);
            suffix.template store_head<cuda_memory_order::relaxed>();
            __threadfence();
            keystate = node_type::KEYSTATE_SUFFIX;
          }
          else {
            // insert link entry and continue to next layer
            current_node_index = allocator.allocate(tile);
            auto next_root_node = masstree_node<tile_type>(
              reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile);
            next_root_node.initialize_root();
            next_root_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            border_node.insert(key_slice, current_node_index, node_type::KEYSTATE_LINK);
            border_node.template store<cuda_memory_order::relaxed>();
            border_node.unlock();
            slice++;
            continue;
          }
        }
        else {
          // insert value to the node
          current_node_index = value;
          keystate = node_type::KEYSTATE_VALUE;
        }
        border_node.insert(key_slice, current_node_index, keystate);
      }
      // reaching here means we updated border node and it's done
      border_node.template store<cuda_memory_order::relaxed>();
      border_node.unlock();
      return true;
    }
    assert(false);
    return false;
  }

  template <bool concurrent, bool do_merge, bool do_remove_empty_root, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    static_assert(concurrent || (!do_merge && !do_remove_empty_root));
    static_assert(do_merge || !do_remove_empty_root);
    using node_type = masstree_node<tile_type>;
    using suffix_type = masstree_suffix_node<tile_type, device_allocator_context_type>;
    using dynamic_stack_type = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    static constexpr auto memory_order = concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak;
    struct merge_early_exit_check {
      DEVICE_QUALIFIER bool operator()(const node_type& border_node,
                                       const key_slice_type& key_slice,
                                       bool more_key) const {
        // if the border node doesn't have the key slice, no need to lock the node
        return !border_node.key_is_in_node(key_slice, more_key);
      }
      DEVICE_QUALIFIER void early_exit() { early_exited_ = true; }
      DEVICE_QUALIFIER void reset() { early_exited_ = false; }
      bool early_exited_ = false;
    };
    [[maybe_unused]] dynamic_stack_type per_layer_indexes(allocator, tile); // (root_index, border_index)
    size_type current_node_index = *d_root_index_;
    uint32_t slice = 0;
    bool retry_with_merge = false;
    while (slice < key_length) {
      key_slice_type key_slice = key[slice];
      const bool more_key = (slice < key_length - 1);
      [[maybe_unused]] size_type border_node_index;
      node_type border_node(tile);
      // traverse the layer
      bool border_node_locked_by_me = true;
      if (do_merge && retry_with_merge) {
        merge_early_exit_check early_exit_check;
        border_node = coop_traverse_until_border_merge(key_slice, more_key, current_node_index,
                                                       tile, allocator, reclaimer, early_exit_check,
                                                       do_remove_empty_root ? &border_node_index : nullptr);
        if (early_exit_check.early_exited_) {
          return false; // key not exists
        }
        retry_with_merge = false;
      }
      else {
        const bool lock_border_node = !more_key;
        border_node = coop_traverse_until_border<concurrent>(key_slice, current_node_index, tile, allocator, lock_border_node, 
                                                             do_remove_empty_root ? &border_node_index : nullptr);
        border_node_locked_by_me = lock_border_node;
      }
      if (more_key) {
        // try traverse
        size_type next_index;
        const int found_keystate = border_node.get_key_value_from_node(key_slice, next_index, true);
        if (found_keystate < 0) { // key not exists
          if (border_node_locked_by_me) { border_node.unlock(); }
          return false;
        }
        else if (found_keystate == node_type::KEYSTATE_LINK) {
          // traverse to next layer
          if (border_node_locked_by_me) { border_node.unlock(); }
          if constexpr (do_remove_empty_root) { per_layer_indexes.push(current_node_index, border_node_index); }
          current_node_index = next_index;
          slice++;
          continue;
        }
        else {  // KEYSTATE_SUFFIX
          auto suffix = suffix_type(
              reinterpret_cast<elem_type*>(allocator.address(next_index)), next_index, tile, allocator);
          suffix.template load_head<memory_order>();
          const bool suffix_eq = suffix.template streq<memory_order>(key + slice, key_length - slice);
          if (suffix_eq) {
            // key exists, erase suffix value and mark suffix nodes garbage
            if (do_merge && border_node.is_underflow()) {
              if (border_node_locked_by_me) { border_node.unlock(); }
              retry_with_merge = true;
              continue;
            }
            if (!border_node_locked_by_me) {
              border_node.lock();
              border_node.template load<cuda_memory_order::relaxed>();
              if constexpr (concurrent) {
                size_type node_index;
                traverse_side_links_with_locks(border_node, node_index, key_slice, tile, allocator);
              }
              if (do_merge && border_node.is_underflow()) {
                border_node.unlock();
                retry_with_merge = true;
                continue;
              }
            }
            if (!border_node.is_border()) {
              border_node.unlock();
              retry_with_merge = true;
              continue;
            }
            if (border_node.is_garbage()) { // this means it's collected empty root
              assert(border_node.is_root());
              border_node.unlock();
              return false;
            }
            const bool success = border_node.erase(key_slice, node_type::KEYSTATE_SUFFIX);
            if (!success) {
              // something changed after lock
              border_node.unlock();
              retry_with_merge = true;
              continue;
            }
            border_node.template store<cuda_memory_order::relaxed>();
            border_node.unlock();
            suffix.template retire<memory_order>(reclaimer, next_index);
          }
          else {
            if (border_node_locked_by_me) { border_node.unlock(); }
            return false;
          }
        }
      }
      else {  // !more_key
        // erase VALUE entry if exists
        assert(border_node_locked_by_me);
        if (do_merge && border_node.is_underflow()) {
          border_node.unlock();
          retry_with_merge = true;
          continue;
        }
        const bool success = border_node.erase(key_slice, node_type::KEYSTATE_VALUE);
        if (success) {
          border_node.template store<cuda_memory_order::relaxed>();
        }
        border_node.unlock();
        if (!success) { return false; }
      }
      // reaching here means we succeeded to erase the entry
      if constexpr (do_remove_empty_root) {
        // collect empty roots
        slice--;
        while (static_cast<int>(slice) >= 0 && border_node.is_root() && border_node.num_keys() == 0) {
          key_slice = key[slice];
          size_type layer_root_index;
          per_layer_indexes.pop(layer_root_index, current_node_index);
          border_node = coop_traverse_until_border<true>(key_slice, current_node_index, tile, allocator, true);
          if (border_node.key_is_in_node(key_slice, node_type::KEYSTATE_LINK) && border_node.is_underflow()) {
            // cannot allow underflow. retry from root with proactive merging
            border_node.unlock();
            merge_early_exit_check early_exit_check;
            border_node = coop_traverse_until_border_merge(key_slice, true, layer_root_index, tile, allocator, reclaimer, early_exit_check);
            if (early_exit_check.early_exited_) { break; }
          }
          if (border_node.get_key_value_from_node(key_slice, current_node_index, node_type::KEYSTATE_LINK)) {
            // check next layer root node
            // current_node_index is next_layer_root_node_index
            auto next_layer_root_node = node_type(
              reinterpret_cast<elem_type*>(allocator.address(current_node_index)), current_node_index, tile);
            next_layer_root_node.lock();
            next_layer_root_node.template load<cuda_memory_order::relaxed>();
            if (!next_layer_root_node.is_garbage() && next_layer_root_node.num_keys() == 0) {
              // still empty, remove them
              next_layer_root_node.make_garbage_node(false);
              border_node.erase(key_slice, node_type::KEYSTATE_LINK);
              next_layer_root_node.template store<cuda_memory_order::relaxed>();
              __threadfence();
              border_node.template store<cuda_memory_order::relaxed>();
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
          slice--;
        }
        per_layer_indexes.destroy();
      }
      return true;
    }
    assert(false);
    return false;
  }

 private:
  // device-side helper functions
  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border(const key_slice_type& key_slice,
                                                                       const size_type& current_root_index,
                                                                       const tile_type& tile,
                                                                       device_allocator_context_type& allocator,
                                                                       bool lock_border_node,
                                                                       size_type* node_index = nullptr) {
    // starting from a local root node in a layer, return the border node and its index
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
          current_node_index,
          tile);
      current_node.template load<concurrent ? cuda_memory_order::relaxed : cuda_memory_order::weak>();
      if constexpr (concurrent) {
        traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);
      }
      if (current_node.is_border()) {
        if (lock_border_node) {
          current_node.lock();
          current_node.template load<cuda_memory_order::relaxed>();
          if constexpr (concurrent) {
            traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
          }
          if (!current_node.is_border()) {
            current_node.unlock();
            continue; // retry traversal to border
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
                                                                             bool more_key,
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
      current_node.template load<cuda_memory_order::relaxed>();
      bool link_traversed = traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit_check(current_node, key_slice, more_key)) {
        early_exit_check.early_exit();
        return current_node;
      }

      // lock the node & traverse again, if it's full or border
      // if it's full, the parent should be known
      if (current_node.is_full() || current_node.is_border()) {
        if (current_node.try_lock()) {
          current_node.template load<cuda_memory_order::relaxed>();
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
          parent_node.template load<cuda_memory_order::relaxed>();
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
          split_result.sibling.template store<cuda_memory_order::relaxed>();
          __threadfence();
          current_node.template store<cuda_memory_order::relaxed>();
          __threadfence();
          parent_node.template store<cuda_memory_order::relaxed>();
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
          two_siblings.right.template store<cuda_memory_order::relaxed>();
          __threadfence();
          two_siblings.left.template store<cuda_memory_order::relaxed>();
          __threadfence();
          current_node.template store<cuda_memory_order::relaxed>();
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
                                                                             bool more_key,
                                                                             const size_type& current_root_index,
                                                                             const tile_type& tile,
                                                                             device_allocator_context_type& allocator,
                                                                             device_reclaimer_context_type& reclaimer,
                                                                             EarlyExitCheck& early_exit_check,
                                                                             size_type* node_index = nullptr) {
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
      current_node.template load<cuda_memory_order::relaxed>();
      bool link_traversed = traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit_check(current_node, key_slice, more_key)) {
        early_exit_check.early_exit();
        return current_node;
      }

      // lock the node & traverse again, if it's underflow or border
      // if it's underflow, the parent and sibling should be known
      if (current_node.is_underflow() || current_node.is_border()) {
        if (current_node.try_lock()) {
          current_node.template load<cuda_memory_order::relaxed>();
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
        sibling_node.template load<cuda_memory_order::relaxed>();
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
        parent_node.template load<cuda_memory_order::relaxed>();
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
            left_sibling_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            right_sibling_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            parent_node.template store<cuda_memory_order::relaxed>();
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
            parent_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            left_sibling_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            right_sibling_node.template store<cuda_memory_order::relaxed>();
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
            current_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            sibling_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            parent_node.template store<cuda_memory_order::relaxed>();
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
            new_sibling_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            current_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            sibling_node.template store<cuda_memory_order::relaxed>();
            __threadfence();
            parent_node.template store<cuda_memory_order::relaxed>();
            new_sibling_node.unlock();
            reclaimer.retire(sibling_index, tile);
          }
          parent_node.unlock();
          sibling_node.unlock();
        }
        // now, current_node is not underflow. if it's not border, unlock.
        if (!current_node.is_border()) { current_node.unlock(); }
      }
      assert(!current_node.is_underflow());

      // now, the node is not underflow; if border it's locked, otherwise not locked.
      if (current_node.is_border()) {
        if (node_index) { *node_index = current_node_index; }
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
  template <typename tile_type, typename Func>
  DEVICE_QUALIFIER void cooperative_traverse_tree_nodes(Func& task, const tile_type& tile) {
    // debug-purpose, so inefficient implementation
    // called with single warp, BFS
    using node_type         = masstree_node<tile_type>;
    using dynamic_stack_type = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    device_allocator_context_type allocator{allocator_, tile};
    // stack: stores node indexes. metadata: # of traversed children
    dynamic_stack_type stack(allocator, tile);
    uint32_t current_node_index = *d_root_index_;
    uint32_t num_traversed_children = 0;
    stack.push(current_node_index, num_traversed_children);
    uint32_t stack_size = 1;
    while (stack_size > 0) {
      stack.pop(current_node_index, num_traversed_children);
      stack_size--;
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(current_node_index)),
          current_node_index,
          tile);
      current_node.template load<cuda_memory_order::weak>();
      if (num_traversed_children == 0) {
        // first time visiting
        task.exec(current_node, tile, allocator);
      }
      if (num_traversed_children < current_node.num_keys()) {
        num_traversed_children++;
        stack.push(current_node_index, num_traversed_children);
        num_traversed_children--;
        stack_size++;
        if ((!current_node.is_border()) ||
            (current_node.get_keystate_from_location(num_traversed_children) == node_type::KEYSTATE_LINK)) {
          // If it's a interior node, push the next node
          // If it's a border node but link entry, push the next layer root
          current_node_index = current_node.get_value_from_location(num_traversed_children);
          num_traversed_children = 0;
          stack.push(current_node_index, num_traversed_children);
          stack_size++;
        }
      }
    }
    stack.destroy();
  }

  template <typename func>
  void traverse_tree_nodes() {
    kernels::traverse_tree_nodes_kernel<func><<<1, 32>>>(*this);
    cudaDeviceSynchronize();
  }

  struct print_node_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, const tile_type& tile, device_allocator_context_type& allocator) {
      node.print(allocator);
    }
    DEVICE_QUALIFIER void fini() {}
  };
  void print() {
    traverse_tree_nodes<print_node_task>();
  }

  struct validate_tree_task {
    DEVICE_QUALIFIER void init(bool lead_lane) {
      lead_lane_ = lead_lane;
      num_entries_ = 0;
      num_nodes_ = 0;
      num_suffix_nodes_ = 0;
    }
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, const tile_type& tile, device_allocator_context_type& allocator) {
      uint32_t num_entries = 0;
      if (node.is_border()) {
        uint16_t num_keys = node.num_keys();
        key_slice_type before_key = 0;
        uint32_t before_keystate = 0;
        for (uint16_t i = 0; i < num_keys; i++) {
          auto key = node.get_key_from_location(i);
          auto keystate = node.get_keystate_from_location(i);
          if (i > 0) {
            assert(before_key <= key);
            if (before_key == key) {
              assert((before_keystate == node_type::KEYSTATE_VALUE && 
                      keystate != node_type::KEYSTATE_VALUE));
            }
          }
          if (keystate == node_type::KEYSTATE_SUFFIX) {
            auto suffix_index = node.get_value_from_location(i);
            auto suffix = masstree_suffix_node<tile_type, device_allocator_context_type>(
                reinterpret_cast<elem_type*>(allocator.address(suffix_index)), suffix_index, tile, allocator);
            suffix.template load_head<cuda_memory_order::weak>();
            num_suffix_nodes_ += suffix.get_num_nodes();
          }
          before_key = key;
          before_keystate = keystate;
          if (keystate == node_type::KEYSTATE_VALUE ||
              keystate == node_type::KEYSTATE_SUFFIX) {
            num_entries++;
          }
        }
      }
      num_entries_ += num_entries;
      num_nodes_++;
    }
    DEVICE_QUALIFIER void fini() {
      if (lead_lane_) {
        printf("%lu entries, %lu nodes (+%lu suffix nodes) found\n", num_entries_, num_nodes_, num_suffix_nodes_);
      }
    }
    bool lead_lane_;
    uint64_t num_entries_, num_nodes_, num_suffix_nodes_;
  };
  void validate() {
    traverse_tree_nodes<validate_tree_task>();
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
      node.template load<cuda_memory_order::relaxed>();
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
      node.template load<cuda_memory_order::relaxed>();
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
    root_node.template store<cuda_memory_order::weak>();
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
