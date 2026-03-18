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
#include <kernels.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <masstree_node.hpp>
#include <masstree_node_subwarp.hpp>
#include <suffix.hpp>
#include <suffix_subwarp.hpp>
#include <queue>
#include <sstream>
#include <type_traits>

#include <dynamic_stack.hpp>
#include <simple_bump_alloc.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_dummy_reclaim.hpp>
#include <simple_debra_reclaim.hpp>

namespace GpuMasstree {

struct masstree_layout_warp {
  static constexpr uint32_t cg_tile_size = 32;
  static constexpr uint32_t node_width = 16;
  using node_lane_type = uint32_t;

  template <typename tile_type, typename allocator_type>
  using node_type = masstree_node<tile_type, allocator_type>;

  template <typename tile_type, typename allocator_type>
  using suffix_type = suffix_node<tile_type, allocator_type>;
};

struct masstree_layout_subwarp {
  static constexpr uint32_t cg_tile_size = 16;
  static constexpr uint32_t node_width = 16;
  using node_lane_type = uint64_t;

  template <typename tile_type, typename allocator_type>
  using node_type = masstree_node_subwarp<tile_type, allocator_type>;

  template <typename tile_type, typename allocator_type>
  using suffix_type = suffix_node_subwarp<tile_type, allocator_type>;
};

template <typename Allocator,
          typename Reclaimer,
          typename Layout = masstree_layout_warp>
struct gpu_masstree {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  using layout_type = Layout;
  using node_lane_type = typename layout_type::node_lane_type;
  static auto constexpr branching_factor = layout_type::node_width;
  static auto constexpr cg_tile_size = layout_type::cg_tile_size;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  using host_allocator_type = Allocator;
  using device_allocator_instance_type = typename host_allocator_type::device_instance_type;
  using device_allocator_context_type = device_allocator_context<host_allocator_type>;

  using host_reclaimer_type = Reclaimer;
  using device_reclaimer_instance_type = typename host_reclaimer_type::device_instance_type;
  using device_reclaimer_context_type = device_reclaimer_context<host_reclaimer_type>;

  template <typename tile_type>
  using node_type_t = typename layout_type::template node_type<tile_type, device_allocator_context_type>;

  template <typename tile_type>
  using suffix_type_t = typename layout_type::template suffix_type<tile_type, device_allocator_context_type>;

  gpu_masstree() = delete;
  gpu_masstree(const host_allocator_type& host_allocator,
               const host_reclaimer_type& host_reclaimer)
      : allocator_(host_allocator.get_device_instance())
      , reclaimer_(host_reclaimer.get_device_instance()) {
    allocate();
  }

  gpu_masstree& operator=(const gpu_masstree& other) = delete;
  gpu_masstree(const gpu_masstree& other)
      : root_index_(other.root_index_)
      , allocator_(other.allocator_)
      , reclaimer_(other.reclaimer_) {}

  ~gpu_masstree() {
    deallocate();
  }

  // host-side APIs
  // if key_lengths == NULL, we use max_key_length as a fixed length
  template <bool concurrent = false,
            bool reuse_root = true>
  void find(const key_slice_type* keys,
            const size_type max_key_length,
            const size_type* key_lengths,
            value_type* values,
            const size_type num_keys,
            cudaStream_t stream = 0) {
    kernels::GpuMasstree::find_device_func<concurrent, reuse_root, key_slice_type, size_type, value_type>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool enable_suffix = true,
            bool reuse_root = true>
  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0,
              bool update_if_exists = false) {
    kernels::GpuMasstree::insert_device_func<enable_suffix, reuse_root, key_slice_type, size_type, value_type>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .update_if_exists = update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool do_remove_empty_root = true,
            bool do_merge = true,
            bool concurrent = true,
            bool reuse_root = true>
  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0) {
    kernels::GpuMasstree::erase_device_func<concurrent, do_merge, do_remove_empty_root, reuse_root, key_slice_type, size_type, value_type>
      func{.d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths};
    kernels::launch_batch_kernel(*this, func, num_keys, stream);
  }

  template <bool use_upper_key = true,
            bool concurrent = false,
            bool reuse_root = true>
  void scan(const key_slice_type* lower_keys,
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
            cudaStream_t stream = 0) {
    kernels::GpuMasstree::scan_device_func<use_upper_key, concurrent, reuse_root, key_slice_type, size_type, value_type>
      func{.d_lower_keys = lower_keys, .d_lower_key_lengths = lower_key_lengths,
           .max_key_length = max_key_length, .max_count_per_query = max_count_per_query,
           .d_upper_keys = upper_keys, .d_upper_key_lengths = upper_key_lengths,
           .d_counts = counts, .d_values = values, .d_out_keys = out_keys, .d_out_key_lengths = out_key_lengths};
    kernels::launch_batch_kernel(*this, func, num_queries, stream);
  }

  template <bool enable_suffix = true,
            bool erase_do_remove_empty_root = true,
            bool erase_do_merge = true,
            bool reuse_root = true>
  void mixed_batch(const kernels::request_type* request_types,
                   const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   value_type* values,
                   bool* results,
                   const size_type num_requests,
                   cudaStream_t stream = 0,
                   bool insert_update_if_exists = false) {
    kernels::GpuMasstree::mixed_device_func<enable_suffix, erase_do_merge, erase_do_remove_empty_root, reuse_root, key_slice_type, size_type, value_type>
      func{.d_types = request_types, .d_keys = keys, .max_key_length = max_key_length, .d_key_lengths = key_lengths, .d_values = values, .d_results = results, .insert_update_if_exists = insert_update_if_exists};
    kernels::launch_batch_kernel(*this, func, num_requests, stream);
  }

  // device-side APIs
  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER node_lane_type cooperative_fetch_root(const tile_type& tile,
                                                         device_allocator_context_type& allocator) {
    using node_type = node_type_t<tile_type>;
    auto root_node = node_type(root_index_, tile, allocator);
    root_node.template load_fetchonly<concurrent>();
    return root_node.get_lane_elem();
  }

  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find_from_root(node_lane_type root_lane_elem,
                                                         const key_slice_type* key,
                                                         size_type key_length,
                                                         const tile_type& tile,
                                                         device_allocator_context_type& allocator) {
    using node_type = node_type_t<tile_type>;
    using suffix_type = suffix_type_t<tile_type>;
    dummy_early_exit_check<node_type> dummy_early_exit;
    size_type slice = 0;
    auto current_node = node_type(root_index_, root_lane_elem, tile, allocator);
    current_node.read_metadata_from_registers();
    while (slice < key_length) {
      const key_slice_type key_slice = key[slice];
      const bool more_key = (slice < key_length - 1);
      coop_traverse_until_border<concurrent>(current_node, key_slice, tile, allocator, false, dummy_early_exit);
      value_type found_value;
      const int found_keystate = current_node.get_key_value_from_node(key_slice, found_value, more_key);
      if (found_keystate < 0) {
        // key not exists, exit early
        return invalid_value;
      }
      if (found_keystate == node_type::KEYSTATE_LINK) {
        slice++;
        current_node = node_type(found_value, tile, allocator);
        current_node.template load<concurrent>();
      }
      else {// keystate == SUFFIX or VALUE
        if (found_keystate == node_type::KEYSTATE_SUFFIX) {
          auto suffix = suffix_type(found_value, tile, allocator);
          suffix.load_head();
          const bool suffix_eq = suffix.streq(key + slice + 1, key_length - slice - 1);
          found_value = suffix_eq ? suffix.get_value() : invalid_value;
        }
        return found_value;
      }
    }
    assert(false);
    return invalid_value;
  }

  template <bool concurrent, typename tile_type>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               size_type key_length,
                                               const tile_type& tile,
                                               device_allocator_context_type& allocator) {
    auto root_lane_elem = cooperative_fetch_root<concurrent>(tile, allocator);
    return cooperative_find_from_root<concurrent>(root_lane_elem, key, key_length, tile, allocator);
  }

  template <bool use_upper_key, bool concurrent, typename tile_type>
  DEVICE_QUALIFIER size_type cooperative_scan_from_root(node_lane_type root_lane_elem,
                                                        const key_slice_type* lower_key,
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
    using node_type = node_type_t<tile_type>;
    using dynamic_stack_type_x2 = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    using dynamic_stack_type_x1 = utils::dynamic_stack_u32<1, tile_type, device_allocator_context_type>;
    dummy_early_exit_check<node_type> dummy_early_exit;
    if (out_max_count <= 0) { return 0; }
    assert(!use_upper_key || upper_key != nullptr);

    static constexpr key_slice_type min_key_slice = std::numeric_limits<key_slice_type>::min();
    static constexpr key_slice_type max_key_slice = std::numeric_limits<key_slice_type>::max();
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
    auto current_node = node_type(root_index_, root_lane_elem, tile, allocator);
    current_node.read_metadata_from_registers();
    while (true) {
      // traverse the btree: current_node can be root node or border node
      coop_traverse_until_border<concurrent>(current_node, lower_key_slice, tile, allocator, false, dummy_early_exit);
      // traverse side links, scan the nodes, and decide scan_op:
      //      >=0 (down-link location; go to next layer)
      //      -1 (continue side-link traverse)
      //      -2 (go to previous layer)
      //      -3 (end scanning the range)
      int scan_op = -1;
      while (true) {
        if (current_node.is_garbage()) { scan_op = -1; }
        else {
          // scan a node and store outputs
          uint32_t count = current_node.template scan<use_upper_key>(
              lower_key_slice, lower_key_more, lower_key, lower_key_length, passed_lower_key,
              upper_key_slice, upper_key_more, upper_key, upper_key_length, ignore_upper_key,
              out_max_count, scan_op, out_value, out_keys, out_key_lengths, layer, out_key_max_length);
          if (count > 0) {
            out_count += count;
            out_max_count -= count;
            if (out_value) { out_value += count; }
            if (out_key_lengths) {out_key_lengths += count;}
            if (out_keys) {
              utils::fill_output_keys_from_key_slice_stack<0>(key_slice_and_node_index_stack, out_keys, out_key_max_length, layer, count);
              out_keys += (out_key_max_length * count);
            }
          }
          //  if got enough outputs, end scanning
          if (out_max_count <= 0) { scan_op = -3; }
          else if (scan_op < 0) {
            // if it's end of this layer, go to prev layer
            if (!current_node.has_sibling()) { scan_op = (layer == 0) ? -3 : -2; }
            // if reached upper key, end scanning
            else if (use_upper_key && upper_key_slice <= current_node.get_high_key()) { scan_op = -3; }
            // else: continue side traversal
            else { assert(scan_op == -1); }
          }
          //  else: found down-link in range, go to next layer
        }
        if (scan_op != -1) { break; }
        current_node = node_type(current_node.get_sibling_index(), tile, allocator);
        current_node.template load<concurrent>();
      }
      // switch layer
      if (scan_op >= 0) { // go to next layer
        layer++;
        // checkpoint key slice and node index to stack
        const key_slice_type checkpoint_key_slice = current_node.get_key_from_location(scan_op);
        key_slice_and_node_index_stack.push(checkpoint_key_slice, current_node.get_node_index());
        current_node = node_type(current_node.get_value_from_location(scan_op), tile, allocator);
        current_node.template load<concurrent>();
        // check if passed the lower key
        passed_lower_key = passed_lower_key ||
            (lower_key_slice < checkpoint_key_slice || layer == lower_key_length);
        // lower key for next layer
        lower_key_slice = passed_lower_key ? min_key_slice : lower_key[layer];
        lower_key_more = !(passed_lower_key || layer == lower_key_length - 1);
        // handle upper key
        if constexpr (use_upper_key) {
          ignore_upper_key_stack.push(static_cast<size_type>(ignore_upper_key));
          // check if ignore the upper key at next layer
          ignore_upper_key = ignore_upper_key ||
              (checkpoint_key_slice < upper_key_slice || layer == upper_key_length);
          // upper key for next layer
          upper_key_slice = ignore_upper_key ? 0 : upper_key[layer];
          upper_key_more = (layer != upper_key_length - 1);
        }
      }
      else if (scan_op == -2) { // go to prev layer
        // pop stack until checkpoint key is not 0xFFFFFFFF
        size_type next_node_index;
        while (true) {
          if (layer == 0) {
            key_slice_and_node_index_stack.destroy();
            if constexpr (use_upper_key) { ignore_upper_key_stack.destroy(); }
            return out_count;
          }
          layer--;
          key_slice_and_node_index_stack.pop(lower_key_slice, next_node_index);
          if constexpr (use_upper_key) {
            ignore_upper_key_stack.pop(ignore_upper_key);
            assert(ignore_upper_key || layer < upper_key_length);
            if (!ignore_upper_key && lower_key_slice >= upper_key[layer]) { continue; }
          }
          if (lower_key_slice < max_key_slice) { break; }
        }
        current_node = node_type(next_node_index, tile, allocator);
        current_node.template load<concurrent>();
        // scan from checkpoint_key, exclusive
        lower_key_slice++;
        lower_key_more = false;
        if constexpr (use_upper_key) {
          upper_key_slice = ignore_upper_key ? 0 : upper_key[layer];
          upper_key_more = (layer != upper_key_length - 1);
          assert(ignore_upper_key || node_type::cmp_key(lower_key_slice, lower_key_more, upper_key_slice, upper_key_more));
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

  template <bool use_upper_key, bool concurrent, typename tile_type>
  DEVICE_QUALIFIER size_type cooperative_scan(const key_slice_type* lower_key,
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
    auto root_lane_elem = cooperative_fetch_root<concurrent>(tile, allocator);
    return cooperative_scan_from_root<use_upper_key, concurrent>(
      root_lane_elem,
      lower_key, lower_key_length, tile, allocator, upper_key, upper_key_length,
      out_max_count, out_value, out_keys, out_key_lengths, out_key_max_length);
  }

  template <bool enable_suffix, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert_from_root(node_lane_type root_lane_elem,
                                                     const key_slice_type* key,
                                                     const size_type key_length,
                                                     const value_type& value,
                                                     const tile_type& tile,
                                                     device_allocator_context_type& allocator,
                                                     device_reclaimer_context_type& reclaimer,
                                                     bool update_if_exists = false) {
    using node_type = node_type_t<tile_type>;
    using suffix_type = suffix_type_t<tile_type>;
    struct split_early_exit_check {
      DEVICE_QUALIFIER bool check(const node_type& border_node) {
        // if the border node already has the key slice and it's not the last slice,
        // we just follow the same entry, no need to update the node; so we don't lock it
        exited_ =
            more_key_ && border_node.key_is_in_node(key_slice_, node_type::KEYSTATE_LINK);
        return exited_;
      }
      const key_slice_type& key_slice_;
      const bool& more_key_;
      bool exited_ = false;
    };
    size_type prev_root_index = invalid_value;
    size_type slice = 0;
    size_type current_root_index = root_index_;
    auto current_node = node_type(root_index_, root_lane_elem, tile, allocator);
    current_node.read_metadata_from_registers();
    while (slice < key_length) {
      const key_slice_type key_slice = key[slice];
      const bool more_key = (slice < key_length - 1);
      split_early_exit_check early_exit{key_slice, more_key};
      coop_traverse_until_border_split(current_node, key_slice, tile, allocator, early_exit);
      if (early_exit.exited_) {
        // find next layer root and continue now
        prev_root_index = current_root_index;
        current_node.get_key_value_from_node(key_slice, current_root_index, more_key);
        current_node = node_type(current_root_index, tile, allocator);
        current_node.template load<true>();
        slice++;
        continue;
      }
      assert(current_node.is_locked());
      if (current_node.is_garbage()) {
        // garbage after side-traversal means (is_garbage && !has_sibling)
        // which means it's an empty root node that's collected by erasure.
        // we should retry from the previous layer
        current_node.unlock();
        if (prev_root_index == invalid_value) {
          // if it's cascading, restart from the global root
          current_node = node_type(root_index_, tile, allocator);
          current_node.template load<true>();
          slice = 0;
          continue;
        }
        assert(slice > 0);
        current_node = node_type(prev_root_index, tile, allocator);
        current_node.template load<true>();
        prev_root_index = invalid_value;
        slice--;
        continue;
      }
      value_type found_value;
      auto keystate = current_node.get_key_value_from_node(key_slice, found_value, more_key);
      if (keystate >= 0) {
        // key exists, the value is stored in found_value
        if (keystate == node_type::KEYSTATE_LINK) {
          // continue to next layer
          current_node.unlock();
          prev_root_index = current_root_index;
          current_root_index = found_value;
          current_node = node_type(current_root_index, tile, allocator);
          current_node.template load<true>();
          slice++;
          continue;
        }
        if (keystate == node_type::KEYSTATE_VALUE) {
          // update or fail
          if (update_if_exists) {
            current_node.update(key_slice, value, node_type::KEYSTATE_VALUE, node_type::KEYSTATE_VALUE);
          }
          else { // fail_if_exists
            current_node.unlock();
            return false;
          }
        }
        else if constexpr (enable_suffix) { // node_type::KEYSTATE_SUFFIX
          auto suffix = suffix_type(found_value, tile, allocator);
          suffix.load_head();
          key_slice_type mismatch_suffix_slice;
          int cmp = suffix.strcmp(key + slice + 1, key_length - slice - 1, &mismatch_suffix_slice);
          if (cmp == 0) { // already exists
            if (update_if_exists) {
              // protected by current_node.lock()
              suffix.update_value(value);
              suffix.store_head();
              current_node.unlock();
              return true;
            }
            else {  // fail_if_exists
              current_node.unlock();
              return false;
            }
          }
          else {  // mismatch: create new node chain
            int num_matches = abs(cmp) - 1;
            auto node_chain_index = allocator.allocate(tile);
            current_node.update(key_slice, node_chain_index, node_type::KEYSTATE_SUFFIX, node_type::KEYSTATE_LINK);
            // chain of singleton nodes for matching prefix
            for (int i = 0; i < num_matches; i++) {
              slice++;
              auto singleton_node = node_type(node_chain_index, tile, allocator);
              singleton_node.initialize_root();
              node_chain_index = allocator.allocate(tile);
              singleton_node.insert(key[slice], node_chain_index, node_type::KEYSTATE_LINK);
              singleton_node.template store<true, false>();
            }
            slice++;
            // one diverging node with two entries
            auto doubleton_node = node_type(node_chain_index, tile, allocator);
            doubleton_node.initialize_root();
            // insert suffix of suffix key
            assert(num_matches < suffix.get_key_length());
            if (num_matches == suffix.get_key_length() - 1) {
              doubleton_node.insert(mismatch_suffix_slice, suffix.get_value(), node_type::KEYSTATE_VALUE);
              suffix.retire(reclaimer);
            }
            else {
              auto new_suffix_index = allocator.allocate(tile);
              auto new_suffix = suffix_type(new_suffix_index, tile, allocator);
              new_suffix.move_from(suffix, num_matches + 1, reclaimer);
              new_suffix.store_head();
              doubleton_node.insert(mismatch_suffix_slice, new_suffix_index, node_type::KEYSTATE_SUFFIX);
            }
            // insert suffix of this key
            assert(slice < key_length);
            if (slice == key_length - 1) {
              doubleton_node.insert(key[slice], value, node_type::KEYSTATE_VALUE);
            }
            else {
              node_chain_index = allocator.allocate(tile);
              suffix = suffix_type(node_chain_index, tile, allocator);
              suffix.create_from(key + slice + 1, key_length - slice - 1, value);
              suffix.store_head();
              doubleton_node.insert(key[slice], node_chain_index, node_type::KEYSTATE_SUFFIX);
            }
            doubleton_node.template store<true, false>();
          }
        }
      }
      else {
        // key not exists
        if (more_key) {
          if constexpr (enable_suffix) {
            // insert suffix entry
            found_value = allocator.allocate(tile);
            auto suffix = suffix_type(found_value, tile, allocator);
            suffix.create_from(key + slice + 1, key_length - slice - 1, value);
            suffix.store_head();
            keystate = node_type::KEYSTATE_SUFFIX;
          }
          else {
            // insert link entry and continue to next layer
            prev_root_index = current_root_index;
            current_root_index = allocator.allocate(tile);
            auto next_root_node = node_type(current_root_index, tile, allocator);
            next_root_node.initialize_root();
            next_root_node.template store<true, false>();
            current_node.insert(key_slice, current_root_index, node_type::KEYSTATE_LINK);
            current_node.template store_unlock<true>();
            current_node = next_root_node;
            slice++;
            continue;
          }
        }
        else {
          // insert value to the node
          found_value = value;
          keystate = node_type::KEYSTATE_VALUE;
        }
        current_node.insert(key_slice, found_value, keystate);
      }
      // reaching here means we updated border node and it's done
      current_node.template store_unlock<true>();
      return true;
    }
    assert(false);
    return false;
  }

  template <bool enable_suffix, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           device_allocator_context_type& allocator,
                                           device_reclaimer_context_type& reclaimer,
                                           bool update_if_exists = false) {
    auto root_lane_elem = cooperative_fetch_root<true>(tile, allocator);
    return cooperative_insert_from_root<enable_suffix>(root_lane_elem, key, key_length, value, tile, allocator, reclaimer, update_if_exists);
  }

  template <bool concurrent, bool do_merge, bool do_remove_empty_root, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase_from_root(node_lane_type root_lane_elem,
                                                    const key_slice_type* key,
                                                    const size_type key_length,
                                                    const tile_type& tile,
                                                    device_allocator_context_type& allocator,
                                                    device_reclaimer_context_type& reclaimer) {
    static_assert(concurrent || (!do_merge && !do_remove_empty_root));
    static_assert(do_merge || !do_remove_empty_root);
    using node_type = node_type_t<tile_type>;
    using suffix_type = suffix_type_t<tile_type>;
    using dynamic_stack_type = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    struct merge_early_exit_check {
      DEVICE_QUALIFIER bool check(const node_type& border_node) {
        // if the border node doesn't have the key slice, no need to lock the node
        exited_ = !border_node.key_is_in_node(key_slice_, more_key_);
        return exited_;
      }
      const key_slice_type& key_slice_;
      const bool more_key_;
      bool exited_ = false;
    };
    [[maybe_unused]] dynamic_stack_type per_layer_indexes(allocator, tile); // (root_index, border_index)
    uint32_t slice = 0;
    bool retry_with_merge = false;
    size_type current_root_index = root_index_;
    auto current_node = node_type(root_index_, root_lane_elem, tile, allocator);
    current_node.read_metadata_from_registers();
    while (slice < key_length) {
      key_slice_type key_slice = key[slice];
      const bool more_key = (slice < key_length - 1);
      // traverse the layer
      bool border_node_locked_by_me = true;
      {
        merge_early_exit_check early_exit{key_slice, more_key};
        if (do_merge && retry_with_merge) {
          coop_traverse_until_border_merge(current_node, key_slice, tile, allocator, reclaimer, early_exit);
          retry_with_merge = false;
        }
        else {
          const bool lock_border_node = !more_key;
          coop_traverse_until_border<concurrent>(current_node, key_slice, tile, allocator, lock_border_node, early_exit);
          border_node_locked_by_me = lock_border_node;
        }
        if (early_exit.exited_) {
          return false; // key not exists
        }
      }
      if (current_node.is_garbage()) {
        // garbage after side-traversal means it's empty root garbage
        if (border_node_locked_by_me) { current_node.unlock(); }
        return false;
      }
      if (more_key) {
        // try traverse
        value_type found_value;
        int found_keystate = current_node.get_key_value_from_node(key_slice, found_value, true);
        if (found_keystate < 0) { // key not exists
          if (border_node_locked_by_me) { current_node.unlock(); }
          return false;
        }
        else if (found_keystate == node_type::KEYSTATE_LINK) {
          // traverse to next layer
          if (border_node_locked_by_me) { current_node.unlock(); }
          if constexpr (do_remove_empty_root) {
            per_layer_indexes.push(current_root_index, current_node.get_node_index());
          }
          current_root_index = found_value;
          current_node = node_type(current_root_index, tile, allocator);
          current_node.template load<true>();
          slice++;
          continue;
        }
        else {  // KEYSTATE_SUFFIX
          auto suffix = suffix_type(found_value, tile, allocator);
          suffix.load_head();
          const bool suffix_eq = suffix.streq(key + slice + 1, key_length - slice - 1);
          if (suffix_eq) {
            // key exists, erase suffix value and mark suffix nodes garbage
            if (do_merge && current_node.is_underflow()) {
              if (border_node_locked_by_me) { current_node.unlock(); }
              retry_with_merge = true;
              current_node = node_type(current_root_index, tile, allocator);
              current_node.template load<true>();
              continue;
            }
            if (!border_node_locked_by_me) {
              current_node.lock_load();
              if constexpr (concurrent) {
                traverse_side_links_with_locks(current_node, key_slice, tile, allocator);
              }
              if (do_merge && current_node.is_underflow()) {
                current_node.unlock();
                retry_with_merge = true;
                current_node = node_type(current_root_index, tile, allocator);
                current_node.template load<true>();
                continue;
              }
            }
            if (current_node.is_garbage()) {
              current_node.unlock();
              return false;
            }
            if (!current_node.is_border()) {
              current_node.unlock();
              retry_with_merge = true;
              current_node = node_type(current_root_index, tile, allocator);
              current_node.template load<true>();
              continue;
            }
            const bool success = current_node.erase(key_slice, node_type::KEYSTATE_SUFFIX);
            if (!success) {
              // something changed after lock: retry from get_key_value_from_node() above
              found_keystate = current_node.get_key_value_from_node(key_slice, found_value, true);
              current_node.unlock();
              if (found_keystate < 0) {
                return false;
              }
              else {
                assert(found_keystate == node_type::KEYSTATE_LINK);
                if constexpr (do_remove_empty_root) {
                  per_layer_indexes.push(current_root_index, current_node.get_node_index());
                }
                current_root_index = found_value;
                current_node = node_type(current_root_index, tile, allocator);
                current_node.template load<true>();
                slice++;
                continue;
              }
            }
            current_node.template store_unlock<true>();
            suffix.retire(reclaimer);
          }
          else {
            if (border_node_locked_by_me) { current_node.unlock(); }
            return false;
          }
        }
      }
      else {  // !more_key
        // erase VALUE entry if exists
        assert(border_node_locked_by_me);
        if (do_merge && current_node.is_underflow()) {
          current_node.unlock();
          retry_with_merge = true;
          current_node = node_type(current_root_index, tile, allocator);
          current_node.template load<true>();
          continue;
        }
        const bool success = current_node.erase(key_slice, node_type::KEYSTATE_VALUE);
        if (success) {
          current_node.template store_unlock<true>();
        }
        else {
          current_node.unlock();
          return false;
        }
      }
      // reaching here means we succeeded to erase the entry
      if constexpr (do_remove_empty_root) {
        // collect empty roots
        slice--;
        while (static_cast<int>(slice) >= 0 && current_node.is_root() && current_node.num_keys() == 0) {
          key_slice = key[slice];
          size_type layer_root_index, layer_border_index;
          per_layer_indexes.pop(layer_root_index, layer_border_index);
          merge_early_exit_check early_exit{key_slice, true};
          current_node = node_type(layer_border_index, tile, allocator);
          current_node.template load<true>();
          coop_traverse_until_border<true>(current_node, key_slice, tile, allocator, true, early_exit);
          if (early_exit.exited_) { break; }
          if (current_node.key_is_in_node(key_slice, node_type::KEYSTATE_LINK) && current_node.is_underflow()) {
            // cannot allow underflow. retry from root with proactive merging
            current_node.unlock();
            current_node = node_type(layer_root_index, tile, allocator);
            current_node.template load<true>();
            coop_traverse_until_border_merge(current_node, key_slice, tile, allocator, reclaimer, early_exit);
            if (early_exit.exited_) { break; }
          }
          value_type found_value;
          if (current_node.get_key_value_from_node(key_slice, found_value, node_type::KEYSTATE_LINK)) {
            // check next layer root node
            // found_value is next_layer_root_node_index
            auto next_layer_root_node = node_type(found_value, tile, allocator);
            next_layer_root_node.lock_load();
            if (!next_layer_root_node.is_garbage() && next_layer_root_node.num_keys() == 0) {
              // still empty, remove them
              next_layer_root_node.make_garbage_node(false);
              current_node.erase(key_slice, node_type::KEYSTATE_LINK);
              next_layer_root_node.template store_unlock<false>();
              current_node.template store_unlock<true>();
              reclaimer.retire(next_layer_root_node.get_node_index(), tile);
            }
            else {
              // other warp changed the root
              next_layer_root_node.unlock();
              current_node.unlock();
              break;
            }
          }
          else {
            // other warp changed the root
            current_node.unlock();
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

  template <bool concurrent, bool do_merge, bool do_remove_empty_root, typename tile_type>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          device_allocator_context_type& allocator,
                                          device_reclaimer_context_type& reclaimer) {
    auto root_lane_elem = cooperative_fetch_root<concurrent>(tile, allocator);
    return cooperative_erase_from_root<concurrent, do_merge, do_remove_empty_root>(
      root_lane_elem, key, key_length, tile, allocator, reclaimer);
  }

 private:
  // device-side helper functions
  template <typename node_type>
  struct dummy_early_exit_check {
    DEVICE_QUALIFIER constexpr bool check(const node_type& border_node) const noexcept {
      return false;
    }
  };

  template <bool concurrent, typename tile_type, typename early_exit_check>
  DEVICE_QUALIFIER void coop_traverse_until_border(node_type_t<tile_type>& current_node,
                                                   const key_slice_type& key_slice,
                                                   const tile_type& tile,
                                                   device_allocator_context_type& allocator,
                                                   bool lock_border_node,
                                                   early_exit_check& early_exit) {
    // starting from a local root node in a layer, return the border node and its index
    using node_type = node_type_t<tile_type>;
    while (true) {
      if constexpr (concurrent) {
        traverse_side_links(current_node, key_slice, tile, allocator);
      }
      if (current_node.is_border()) {
        if (lock_border_node) {
          if (early_exit.check(current_node)) {
            return;
          }
          current_node.lock_load();
          if constexpr (concurrent) {
            traverse_side_links_with_locks(current_node, key_slice, tile, allocator);
          }
          if (!current_node.is_border()) {
            current_node.unlock();
            continue; // retry traversal to border
          }
        }
        return;
      }
      else {
        auto next_index = current_node.find_next(key_slice);
        current_node = node_type(next_index, tile, allocator);
        current_node.template load<concurrent>();
      }
    }
    assert(false);
  }

  template <typename tile_type, typename early_exit_check>
  DEVICE_QUALIFIER void coop_traverse_until_border_split(node_type_t<tile_type>& current_node,
                                                         const key_slice_type& key_slice,
                                                         const tile_type& tile,
                                                         device_allocator_context_type& allocator,
                                                         early_exit_check& early_exit) {
    // starting from a local root node in a layer, return the LOCKED border node and its index
    // proactively split full nodes while traversal. also the returned border node is not full.
    // if early exit condition is met, returned node is not locked by this warp (might locked by another)
    using node_type = node_type_t<tile_type>;
    const size_type root_index = current_node.get_node_index();
    size_type parent_index = root_index;
    while (true) {
      bool link_traversed = traverse_side_links(current_node, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit.check(current_node)) {
        return;
      }

      // lock the node & traverse again, if it's full or border
      // if it's full, the parent should be known
      if (current_node.is_full() || current_node.is_border()) {
        if (current_node.try_lock_load()) {
          if (!current_node.is_full()) {
            link_traversed |= traverse_side_links_with_locks(current_node, key_slice, tile, allocator);
          }
          if (current_node.is_full()) {
            // if parent is unknown, restart from root
            if (current_node.get_node_index() != root_index &&
                (current_node.get_node_index() == parent_index || link_traversed || current_node.traverse_required(key_slice))) {
              current_node.unlock();
              current_node = node_type(root_index, tile, allocator);
              current_node.template load<true>();
              parent_index = root_index;
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
          current_node = node_type(parent_index, tile, allocator);
          current_node.template load<true>();
          continue;
        }
      }
      assert((current_node.is_full() || current_node.is_border()) ? 
             (current_node.is_locked() && !current_node.traverse_required(key_slice)) : true);
      assert(current_node.is_full() ?
             (current_node.get_node_index() == root_index || (current_node.get_node_index() != parent_index && !link_traversed)) : true);

      // if the node is full, split. it's already locked if it's full.
      if (current_node.is_full()) {
        if (current_node.get_node_index() != root_index) {
          assert(!current_node.is_root());
          auto parent_node = node_type(parent_index, tile, allocator);
          parent_node.lock_load();
          // parent should be not full, not garbage, and correct parent
          if (parent_node.is_full() ||
              parent_node.is_garbage() ||
              !parent_node.ptr_is_in_node(current_node.get_node_index())) {
            current_node.unlock();
            parent_node.unlock();
            current_node = node_type(root_index, tile, allocator);
            current_node.template load<true>();
            parent_index = root_index;
            continue;
          }
          // do split
          auto right_sibling_node = node_type(allocator.allocate(tile), tile, allocator);
          current_node.split(right_sibling_node, parent_index, parent_node);
          // write order: right -> left -> parent
          if (current_node.get_high_key() < key_slice) {
            // continue traverse with right_sibling_node
            right_sibling_node.template store<false>();
            current_node.template store_unlock<true>();
            parent_node.template store_unlock<true>();
            current_node = right_sibling_node;
          }
          else {  // continue traverse with current_node 
            right_sibling_node.template store_unlock<false>();
            current_node.template store<true>();
            parent_node.template store_unlock<true>();
          }
        }
        else { // (current_node.get_node_index() == root_node_index)
          assert(current_node.is_root());
          auto left_child_node = node_type(allocator.allocate(tile), tile, allocator);
          auto right_child_node = node_type(allocator.allocate(tile), tile, allocator);
          current_node.split_as_root(left_child_node, right_child_node);
          // write order: right -> left -> parent
          if (current_node.find_next(key_slice) == left_child_node.get_node_index()) {
            // continue traversal with left_child_node
            right_child_node.template store_unlock<false>();
            left_child_node.template store<false>();
            current_node.template store_unlock<true>();
            current_node = left_child_node;
          }
          else {  // continue traversal with right_child_node
            right_child_node.template store<false>();
            left_child_node.template store_unlock<false>();
            current_node.template store_unlock<true>();
            current_node = right_child_node;
          }
          parent_index = root_index;
        }
        // now, current_node is not full. if it's not border, unlock.
        if (!current_node.is_border()) { current_node.unlock(); }
      }
      assert(!current_node.is_full());

      // now, the node is not full; if border it's locked, otherwise not locked.
      // traversal or insert
      if (current_node.is_border()) {
        return;
      } else {  // traverse
        parent_index = current_node.get_node_index();
        auto next_index = current_node.find_next(key_slice);
        current_node = node_type(next_index, tile, allocator);
        current_node.template load<true>();
      }
    }
    assert(false);
  }

  template <typename tile_type, typename early_exit_check>
  DEVICE_QUALIFIER void coop_traverse_until_border_merge(node_type_t<tile_type>& current_node,
                                                         const key_slice_type& key_slice,
                                                         const tile_type& tile,
                                                         device_allocator_context_type& allocator,
                                                         device_reclaimer_context_type& reclaimer,
                                                         early_exit_check& early_exit) {
    // starting from a local root node in a layer, return the LOCKED border node and its index
    // proactively merge/borrow underflow nodes while traversal. also the returned border node is not underflow.
    // if early exit condition is met, returned node is not locked by this warp (might locked by another)
    using node_type = node_type_t<tile_type>;
    const size_type root_index = current_node.get_node_index();
    size_type parent_index = root_index;
    size_type sibling_index = root_index;
    bool sibling_at_left = false;
    while (true) {
      bool link_traversed = traverse_side_links(current_node, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit.check(current_node)) {
        return;
      }

      // lock the node & traverse again, if it's underflow or border
      // if it's underflow, the parent and sibling should be known
      if (current_node.is_underflow() || current_node.is_border()) {
        if (current_node.try_lock_load()) {
          if (!current_node.is_underflow()) {
            link_traversed |= traverse_side_links_with_locks(current_node, key_slice, tile, allocator);
          }
          if (current_node.is_underflow()) {
            // if parent is unknown, restart from root
            if (current_node.get_node_index() != root_index &&
                (current_node.get_node_index() == parent_index || sibling_index == root_index ||
                 link_traversed || current_node.traverse_required(key_slice))) {
              current_node.unlock();
              current_node = node_type(root_index, tile, allocator);
              current_node.template load<true>();
              parent_index = root_index;
              sibling_index = root_index;
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
          current_node = node_type(parent_index, tile, allocator);
          current_node.template load<true>();
          sibling_index = root_index;
          continue;
        }
      }
      assert((current_node.is_underflow() || current_node.is_border()) ? 
             (current_node.is_locked() && !current_node.traverse_required(key_slice)) : true);
      assert(current_node.is_underflow() ?
             (current_node.get_node_index() == root_index || (current_node.get_node_index() != parent_index && sibling_index != root_index && !link_traversed)) : true);


      // proactively merge/borrow underflow nodes
      if (current_node.is_underflow()) {
        // lock the sibling first
        auto sibling_node = node_type(sibling_index, tile, allocator);
        if (sibling_at_left) {
          sibling_node.lock_load();
        }
        else {
          // global lock order: right -> left; use try_lock to avoid deadlock
          if (!sibling_node.try_lock_load()) {
            current_node.unlock();
            current_node = node_type(parent_index, tile, allocator);
            current_node.template load<true>();
            sibling_index = root_index;
            continue;
          }
        }
        // check sibling validity
        if (sibling_node.is_garbage() ||
            (sibling_at_left ?
             (sibling_node.get_sibling_index() != current_node.get_node_index()) :
             (current_node.get_sibling_index() != sibling_node.get_node_index()))) {
          current_node.unlock();
          sibling_node.unlock();
          current_node = node_type(root_index, tile, allocator);
          current_node.template load<true>();
          parent_index = root_index;
          sibling_index = root_index;
          continue;
        }
        // lock the parent
        auto parent_node = node_type(parent_index, tile, allocator);
        parent_node.lock_load();
        // make sure parent is not garbage and not underflow
        if (parent_node.is_garbage() || parent_node.is_underflow()) {
          current_node.unlock();
          sibling_node.unlock();
          parent_node.unlock();
          current_node = node_type(root_index, tile, allocator);
          current_node.template load<true>();
          parent_index = root_index;
          sibling_index = root_index;
          continue;
        }
        // make sure parent is correct parent for both children
        int sibling_at_left = parent_node.check_valid_merge_siblings(
          current_node.get_node_index(), sibling_node.get_node_index());
        if (sibling_at_left < 0) {
          current_node.unlock();
          sibling_node.unlock();
          parent_node.unlock();
          current_node = node_type(root_index, tile, allocator);
          current_node.template load<true>();
          parent_index = root_index;
          sibling_index = root_index;
          continue;
        }
        // now all three nodes are locked
        if (current_node.is_mergeable(sibling_node)) {
          // merge
          if (parent_node.get_node_index() != root_index || parent_node.num_keys() > 2) {
            if (sibling_at_left) { // left_node = sibling_node, right_node = current_node
              sibling_node.merge(current_node, parent_node);
              // write order: left -> right -> parent
              sibling_node.template store<false>();
              current_node.template store_unlock<true>();
              parent_node.template store_unlock<true>();
              reclaimer.retire(current_node.get_node_index(), tile);
              current_node = sibling_node;
            }
            else { // left_node = current_node, right_node = sibling_node
              current_node.merge(sibling_node, parent_node);
              // write order: left -> right -> parent
              current_node.template store<false>();
              sibling_node.template store_unlock<true>();
              parent_node.template store_unlock<true>();
              reclaimer.retire(sibling_node.get_node_index(), tile);
            }
          }
          else {
            if (sibling_at_left) { // left_node = sibling_node, right_node = current_node
              parent_node.merge_to_root(root_index, sibling_node, current_node);
              // write order: parent -> left -> right
              parent_node.template store<false>();
              sibling_node.template store_unlock<true>();
              current_node.template store_unlock<true>();
            }
            else { // left_node = current_node, right_node = sibling_node
              parent_node.merge_to_root(root_index, current_node, sibling_node);
              // write order: parent -> left -> right
              parent_node.template store<false>();
              current_node.template store_unlock<true>();
              sibling_node.template store_unlock<true>();
            }
            reclaimer.retire(current_node.get_node_index(), tile);
            reclaimer.retire(sibling_node.get_node_index(), tile);
            current_node = parent_node;
          }
        }
        else {
          // borrow
          if (sibling_at_left) { // left_node = sibling_node, right_node = current_node
            current_node.borrow_left(sibling_node, parent_node);
            // write order: right -> left -> parent
            current_node.template store<false>();
            sibling_node.template store_unlock<true>();
            parent_node.template store_unlock<true>();
          }
          else { // left_node = current_node, right_node = sibling_node
            // borrow_right need additional node to ensure correct lock-free traversal
            auto new_sibling_node = node_type(allocator.allocate(tile), tile, allocator);
            current_node.borrow_right(sibling_node, parent_node, new_sibling_node);
            // write order: new_right -> left -> right -> parent
            new_sibling_node.template store_unlock<false>();
            current_node.template store<true>();
            sibling_node.template store_unlock<true>();
            parent_node.template store_unlock<true>();
            reclaimer.retire(sibling_node.get_node_index(), tile);
          }
        }
        // now, current_node is not underflow. if it's not border, unlock.
        if (!current_node.is_border()) { current_node.unlock(); }
      }
      assert(!current_node.is_underflow());

      // now, the node is not underflow; if border it's locked, otherwise not locked.
      if (current_node.is_border()) {
        return;
      }
      else { // traverse
        parent_index = current_node.get_node_index();
        auto next_index = current_node.find_next_and_sibling(key_slice, sibling_index, sibling_at_left);
        current_node = node_type(next_index, tile, allocator);
        current_node.template load<true>();
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
    using node_type = node_type_t<tile_type>;
    using dynamic_stack_type = utils::dynamic_stack_u32<2, tile_type, device_allocator_context_type>;
    device_allocator_context_type allocator{allocator_, tile};
    // stack: stores node indexes. metadata: # of traversed children
    dynamic_stack_type stack(allocator, tile);
    uint32_t current_node_index = root_index_;
    uint32_t num_traversed_children = 0;
    stack.push(current_node_index, num_traversed_children);
    uint32_t stack_size = 1;
    while (stack_size > 0) {
      stack.pop(current_node_index, num_traversed_children);
      stack_size--;
      node_type current_node = node_type(current_node_index, tile, allocator);
      current_node.template load<false>();
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
  void traverse_tree_nodes(func task) {
    kernels::GpuMasstree::traverse_tree_nodes_kernel<<<1, cg_tile_size>>>(*this, task);
    cudaDeviceSynchronize();
  }

  struct print_node_task {
    template <typename tile_type>
    DEVICE_QUALIFIER void init(const tile_type& tile) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, const tile_type& tile, device_allocator_context_type& allocator) {
      node.print();
    }
    template <typename tile_type>
    DEVICE_QUALIFIER void fini(const tile_type& tile) {}
  };
  void print() {
    print_node_task task;
    traverse_tree_nodes(task);
  }

  struct validate_tree_task {
    template <typename tile_type>
    DEVICE_QUALIFIER void init(const tile_type& tile) {}
    template <typename node_type, typename tile_type>
    DEVICE_QUALIFIER void exec(const node_type& node, const tile_type& tile, device_allocator_context_type& allocator) {
      using suffix_type = suffix_type_t<tile_type>;
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
            auto suffix = suffix_type(suffix_index, tile, allocator);
            suffix.load_head();
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
    template <typename tile_type>
    DEVICE_QUALIFIER void fini(const tile_type& tile) {
      uint64_t total_bytes_used = (num_nodes_ + num_suffix_nodes_) * (2 * branching_factor * sizeof(elem_type));
      float bytes_per_entry = static_cast<float>(total_bytes_used) / num_entries_;
      if (tile.thread_rank() == 0) {
        printf("%lu entries, %lu nodes (+%lu suffix nodes) found\n", num_entries_, num_nodes_, num_suffix_nodes_);
        printf("Total Space Consumption: %lu B (%f B/entry)\n", total_bytes_used, bytes_per_entry);
      }
    }
    uint64_t num_entries_ = 0, num_nodes_ = 0, num_suffix_nodes_ = 0;
  };
  void validate() {
    validate_tree_task task;
    traverse_tree_nodes(task);
  }

 private:

  // Tries to traverse the side-links without locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type>
  DEVICE_QUALIFIER bool traverse_side_links(node_type& node,
                                            const key_slice_type& key_slice,
                                            const tile_type& tile,
                                            device_allocator_context_type& allocator) {
    bool traversed = false;
    while (node.traverse_required(key_slice)) {
      auto node_index = node.get_sibling_index();
      node = node_type(node_index, tile, allocator);
      node.template load<true>();
      traversed |= true;
    }
    return traversed;
  }

  // Tries to traverse the side-links with locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type>
  DEVICE_QUALIFIER bool traverse_side_links_with_locks(node_type& node,
                                                       const key_slice_type& key_slice,
                                                       const tile_type& tile,
                                                       device_allocator_context_type& allocator) {
    bool traversed = false;
    while (node.traverse_required(key_slice)) {
      auto node_index = node.get_sibling_index();
      node_type sibling_node = node_type(node_index, tile, allocator);
      node.unlock();
      sibling_node.lock_load();
      node = sibling_node;
      traversed |= true;
    }
    return traversed;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void allocate_root_node(size_type* d_root_index, const tile_type& tile, device_allocator_context_type& allocator) {
    auto root_index = allocator.allocate(tile);
    *d_root_index = root_index;
    using node_type = node_type_t<tile_type>;
    auto root_node = node_type(root_index, tile, allocator);
    root_node.initialize_root();
    root_node.template store<false>();
  }

  void allocate() {
    initialize();
  }

  void deallocate() {}

  void initialize() {
    const uint32_t num_blocks = 1;
    const uint32_t block_size = cg_tile_size;
    size_type* d_root_index;
    cuda_try(cudaMalloc(&d_root_index, sizeof(size_type)));
    kernels::GpuMasstree::initialize_kernel<<<num_blocks, block_size>>>(*this, d_root_index);
    cuda_try(cudaDeviceSynchronize());
    cuda_try(cudaMemcpy(&root_index_, d_root_index, sizeof(size_type), cudaMemcpyDeviceToHost));
    cuda_try(cudaFree(d_root_index));
  }

  size_type root_index_;
  device_allocator_instance_type allocator_;
  device_reclaimer_instance_type reclaimer_;

  template <typename masstree, typename size_type>
  friend __global__ void kernels::GpuMasstree::initialize_kernel(masstree, size_type*);

  template <bool do_reclaim, typename device_func, typename index_type>
  friend __global__ void kernels::batch_kernel(index_type index,
                                              const device_func func,
                                              uint32_t num_requests);

}; // struct gpu_masstree

template <typename Allocator, typename Reclaimer>
using gpu_masstree_subwarp = gpu_masstree<Allocator, Reclaimer, masstree_layout_subwarp>;

} // namespace GPUMasstree
