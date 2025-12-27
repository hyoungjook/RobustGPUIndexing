/*
 *   Copyright 2022 The Regents of the University of California, Davis
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
#include <btree_kernels.hpp>
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

#include <device_bump_allocator.hpp>
#include <slab_alloc.hpp>

namespace GpuBTree {

template <typename Allocator>
struct gpu_masstree {
  using size_type = uint32_t;
  using elem_type = uint32_t;
  using key_slice_type = elem_type;
  using value_type = elem_type;
  static auto constexpr branching_factor = 16;
  static auto constexpr cg_tile_size = 2 * branching_factor;

  static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

  using allocator_type = Allocator;
  using device_allocator_context_type = device_allocator_context<allocator_type>;

  gpu_masstree() : allocator_{} {
    allocate();
  }
  
  gpu_masstree(const gpu_masstree& other)
      : d_root_index_(other.d_root_index_)
      , allocator_(other.allocator_) {}

  ~gpu_masstree() {}

  // host-side APIs
  // if key_lengths == NULL, we use max_key_length as a fixed length
  void insert(const key_slice_type* keys,
              const size_type max_key_length,
              const size_type* key_lengths,
              const value_type* values,
              const size_type num_keys,
              cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::masstree_insert_kernel<<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, values, num_keys, *this);
  }

  void find(const key_slice_type* keys,
            const size_type max_key_length,
            const size_type* key_lengths,
            value_type* values,
            const size_type num_keys,
            cudaStream_t stream = 0,
            bool concurrent = false) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::masstree_find_kernel<<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, values, num_keys, *this, concurrent);
  }

  void erase(const key_slice_type* keys,
             const size_type max_key_length,
             const size_type* key_lengths,
             const size_type num_keys,
             cudaStream_t stream = 0,
             bool concurrent = false) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::masstree_erase_kernel<false, false><<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, num_keys, *this, concurrent);
  }

  void erase_merge(const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   const size_type num_keys,
                   cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::masstree_erase_kernel<true, false><<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, num_keys, *this, true);
  }

  void erase_merge_rmroot(const key_slice_type* keys,
                          const size_type max_key_length,
                          const size_type* key_lengths,
                          const size_type num_keys,
                          cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::masstree_erase_kernel<true, true><<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, num_keys, *this, true);
  }

  // device-side APIs
  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER value_type cooperative_find(const key_slice_type* key,
                                               const size_type key_length,
                                               const tile_type& tile,
                                               DeviceAllocator& allocator,
                                               bool concurrent = false) {
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length; slice++) {
      const key_slice_type key_slice = key[slice];
      const bool key_end = (slice == key_length - 1);
      auto border_node = coop_traverse_until_border(key_slice, current_node_index, tile, allocator, false, concurrent);
      const bool found = border_node.get_key_value_from_node(key_slice, current_node_index, key_end);
      if (!found) {
        // key not exists, exit early
        return invalid_value;
      }
    }
    // current_node_index has the final value
    return current_node_index;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_insert(const key_slice_type* key,
                                           const size_type key_length,
                                           const value_type& value,
                                           const tile_type& tile,
                                           DeviceAllocator& allocator) {
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length; slice++) {
      const key_slice_type key_slice = key[slice];
      const bool key_end = (slice == key_length - 1);
      struct split_early_exit_check {
        DEVICE_QUALIFIER bool operator()(const node_type& border_node,
                                         const key_slice_type& key_slice,
                                         bool key_end) const {
          // if the border node already has the key slice and it's not the last slice,
          // we just follow the same entry, no need to update the node; so we don't lock it
          return !key_end && border_node.key_is_in_node(key_slice, key_end);
        }
        DEVICE_QUALIFIER void early_exit() { early_exited_ = true; }
        bool early_exited_ = false;
      } early_exit_check;
      auto border_node = coop_traverse_until_border_split(key_slice, key_end, current_node_index, tile, allocator, early_exit_check);
      if (early_exit_check.early_exited_) {
        // find next layer root and continue now
        border_node.get_key_value_from_node(key_slice, current_node_index, key_end);
        continue;
      }
      assert(border_node.is_locked());
      value_type value_to_insert;
      if (key_end) {
        value_to_insert = value;
      }
      else { // find the next layer root
        if (border_node.get_key_value_from_node(key_slice, current_node_index, false)) {
          // already exists, continue to next layer
          border_node.unlock();
          continue;
        }
        else {
          // not found: allocate next root node and insert it
          current_node_index = allocator.allocate(allocator_, 1, tile);
          auto next_root_node = masstree_node<tile_type>(
            reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
            current_node_index,
            tile);
          next_root_node.initialize_root();
          next_root_node.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          value_to_insert = current_node_index;
        }
      }
      border_node.insert(key_slice, value_to_insert, key_end);
      border_node.store(cuda_memory_order::memory_order_relaxed);
      border_node.unlock();
    }
    return true;
  }

  struct dynamic_stack {
    static constexpr uint32_t elems_per_node_ = 2 * branching_factor - 1;
    struct stack_node {
      size_type elems_[elems_per_node_];
      size_type next_node_index_;
    };
    static constexpr size_type invalid_index = std::numeric_limits<size_type>::max();
    template <typename DeviceAllocator, typename tile_type>
    DEVICE_QUALIFIER void push(const size_type& value, DeviceAllocator& allocator, allocator_type& allocator_, const tile_type& tile) {
      stack_node* node_ptr;
      if (head_node_top_ == (elems_per_node_ - 1) || head_node_index_ == invalid_index) {
        auto new_head_node_index = allocator.allocate(allocator_, 1, tile);
        node_ptr = reinterpret_cast<stack_node*>(allocator.address(allocator_, new_head_node_index));
        node_ptr->next_node_index_ = head_node_index_;
        head_node_index_ = new_head_node_index;
        head_node_top_ = -1;
      }
      else {
        node_ptr = reinterpret_cast<stack_node*>(allocator.address(allocator_, head_node_index_));
      }
      head_node_top_++;
      node_ptr->elems_[head_node_top_] = value;
    }
    template <typename DeviceAllocator>
    DEVICE_QUALIFIER size_type pop(DeviceAllocator& allocator, allocator_type& allocator_) {
      assert(head_node_index_ != invalid_index);
      if (head_node_top_ < 0) {
        auto node_ptr = reinterpret_cast<stack_node*>(allocator.address(allocator_, head_node_index_));
        auto new_head_node_index = node_ptr->next_node_index_;
        allocator.deallocate(allocator_, head_node_index_, 1);
        head_node_index_ = new_head_node_index;
        head_node_top_ = elems_per_node_ - 1;
      }
      assert(head_node_index_ != invalid_index);
      assert(head_node_top_ >= 0);
      auto node_ptr = reinterpret_cast<stack_node*>(allocator.address(allocator_, head_node_index_));
      size_type value = node_ptr->elems_[head_node_top_];
      head_node_top_--;
      return value;
    }
    template <typename DeviceAllocator>
    DEVICE_QUALIFIER void destroy(DeviceAllocator& allocator, allocator_type& allocator_) {
      while (head_node_index_ != invalid_index) {
        auto node_ptr = reinterpret_cast<stack_node*>(allocator.address(allocator_, head_node_index_));
        auto next_node_index = node_ptr->next_node_index_;
        allocator.deallocate(allocator_, head_node_index_, 1);
        head_node_index_ = next_node_index;
      }
    }
    int head_node_top_ = -1;
    size_type head_node_index_ = invalid_index;
  };

  template <bool do_merge, bool do_remove_empty_root, typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          DeviceAllocator& allocator,
                                          bool concurrent = false) {
    using node_type = masstree_node<tile_type>;
    [[maybe_unused]] dynamic_stack per_layer_root_indexes; // used for remove_empty_root
    // read-only traverse before the last slice
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length - 1; slice++) {
      if constexpr (do_remove_empty_root) {
        // record root indexes along the trace for later empty-root collection
        per_layer_root_indexes.push(current_node_index, allocator, allocator_, tile);
      }
      const key_slice_type key_slice = key[slice];
      auto border_node = coop_traverse_until_border(key_slice, current_node_index, tile, allocator, false,
                                                    do_merge || do_remove_empty_root || concurrent);
      const bool found = border_node.get_key_value_from_node(key_slice, current_node_index, false);
      if (!found) {
        // key not exists, exit early
        return false;
      }
    }

    // now erase (and merge) the entry at the last_slice layer
    const key_slice_type key_slice = key[key_length - 1];
    if constexpr (!do_merge) {
      // no-merge algorithm: just erase the element
      auto border_node = coop_traverse_until_border(key_slice, current_node_index, tile, allocator, true, concurrent);
      const bool success = border_node.erase(key_slice, true);
      if (success) {
        border_node.store(cuda_memory_order::memory_order_relaxed);
      }
      border_node.unlock();
      return success;
    }
    else { // if (do_merge)
      // merge algorithm
      bool border_node_is_root;
      auto border_node = coop_traverse_until_border_merge(key_slice, current_node_index, border_node_is_root, tile, allocator);
      assert(border_node.is_locked());
      const bool success = border_node.erase(key_slice, true);
      if (success) {
        border_node.store(cuda_memory_order::memory_order_relaxed);
      }
      border_node.unlock();
      if constexpr (do_remove_empty_root) {
        // collect empty roots
        bool cascade_empty_roots = (border_node_is_root && border_node.num_keys() == 0);
        for (int layer = static_cast<int>(key_length) - 2; layer >= 0; layer--) {
          if (!cascade_empty_roots) break;
          current_node_index = per_layer_root_indexes.pop(allocator, allocator_);
          {
            // check root not garbage
            node_type root_node = node_type(
                reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
                current_node_index,
                tile);
            root_node.load(cuda_memory_order::memory_order_relaxed);
            if (root_node.is_garbage()) {
              cascade_empty_roots = false;
              continue;
            }
          }
          border_node = coop_traverse_until_border_merge(key_slice, current_node_index, border_node_is_root, tile, allocator);
          // current_node_index <- next_layer_root_node_index
          const bool found = border_node.get_key_value_from_node(key_slice, current_node_index, false);
          if (found) {
            // check whether next layer root is still empty
            auto next_layer_root_node = node_type(
              reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)), current_node_index, tile);
            next_layer_root_node.lock();
            next_layer_root_node.load(cuda_memory_order::memory_order_relaxed);
            if (!next_layer_root_node.is_garbage() && next_layer_root_node.num_keys() == 0) {
              // still empty, remove them
              next_layer_root_node.make_garbage_node(false);
              border_node.erase(key_slice, false);
              next_layer_root_node.store(cuda_memory_order::memory_order_relaxed);
              __threadfence();
              border_node.store(cuda_memory_order::memory_order_relaxed);
              next_layer_root_node.unlock();
              border_node.unlock();
              // TODO mark next_layer_root_node to be GCed
              cascade_empty_roots = (border_node_is_root && border_node.num_keys() == 0);
            }
            else {
              // other warp removed / inserted into the root
              next_layer_root_node.unlock();
              border_node.unlock();
              cascade_empty_roots = false;
            }
          }
          else {
            // already removed by other warp
            border_node.unlock();
            cascade_empty_roots = false;
          }
        }
        per_layer_root_indexes.destroy(allocator, allocator_);
      }
      return success;
    }
    assert(false);
    return false;
  }

  // device-side helper functions
  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border(const key_slice_type& key_slice,
                                                                       const size_type& current_root_index,
                                                                       const tile_type& tile,
                                                                       DeviceAllocator& allocator,
                                                                       bool lock_border_node,
                                                                       bool concurrent) {
    // starting from a local root node in a layer, return the border node and its index
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
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
        return current_node;
      }
      else {
        current_node_index = current_node.find_next(key_slice);
      }
    }
    assert(false);
  }

  template <typename tile_type, typename DeviceAllocator, typename EarlyExitCheck>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border_split(const key_slice_type& key_slice,
                                                                             bool key_end,
                                                                             const size_type& current_root_index,
                                                                             const tile_type& tile,
                                                                             DeviceAllocator& allocator,
                                                                             EarlyExitCheck& early_exit_check) {
    // starting from a local root node in a layer, return the LOCKED border node and its index
    // proactively split full nodes while traversal. also the returned border node is not full.
    // if early exit condition is met, returned node is not locked by this warp (might locked by another)
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    size_type parent_index = current_root_index;
    while (true) {
      auto current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
          current_node_index,
          tile);
      current_node.load(cuda_memory_order::memory_order_relaxed);
      bool link_traversed = traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

      // early exit condition
      if (current_node.is_border() && early_exit_check(current_node, key_slice, key_end)) {
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
          auto parent_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(allocator_, parent_index)), parent_index, tile);
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
          auto sibling_index = allocator.allocate(allocator_, 1, tile);
          auto split_result = current_node.split(sibling_index,
                                                 parent_index,
                                                 reinterpret_cast<elem_type*>(allocator.address(allocator_, sibling_index)),
                                                 parent_node,
                                                 true);
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
          auto left_sibling_index = allocator.allocate(allocator_, 1, tile);
          auto right_sibling_index = allocator.allocate(allocator_, 1, tile);
          auto two_siblings = current_node.split_as_root(left_sibling_index,
                                                         right_sibling_index,
                                                         reinterpret_cast<elem_type*>(allocator.address(allocator_, left_sibling_index)),
                                                         reinterpret_cast<elem_type*>(allocator.address(allocator_, right_sibling_index)),
                                                         true);
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

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER masstree_node<tile_type> coop_traverse_until_border_merge(const key_slice_type& key_slice,
                                                                             const size_type& current_root_index,
                                                                             bool& output_node_is_root,
                                                                             const tile_type& tile,
                                                                             DeviceAllocator& allocator) {
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = current_root_index;
    size_type parent_index = current_root_index;
    size_type sibling_index = current_root_index;
    bool sibling_at_left = false;
    while (true) {
      node_type current_node = node_type(
          reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
          current_node_index,
          tile);
      current_node.load(cuda_memory_order::memory_order_relaxed);
      bool link_traversed = traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

      // lock the node & traverse again, if it's underflow or border
      // if it's underflow, the parent and sibling should be known
      #define is_current_node_underflow (current_node.is_underflow() && current_node_index != current_root_index)
      if (is_current_node_underflow || current_node.is_border()) {
        if (current_node.try_lock()) {
          current_node.load(cuda_memory_order::memory_order_relaxed);
          if (!is_current_node_underflow) {
            link_traversed |= traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
          }
          if (is_current_node_underflow) {
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
      assert((is_current_node_underflow || current_node.is_border()) ? 
             (current_node.is_locked() && !current_node.traverse_required(key_slice)) : true);
      assert(is_current_node_underflow ?
             (current_node_index == current_root_index || (current_node_index != parent_index && sibling_index != current_root_index && !link_traversed)) : true);


      // proactively merge/borrow underflow nodes
      if (is_current_node_underflow) {
        // lock the sibling first
        auto sibling_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(allocator_, sibling_index)), sibling_index, tile);
        if (sibling_at_left) {
          // if sibling is at left, use try_lock and retry to avoid deadlock
          if (!sibling_node.try_lock()) {
            current_node.unlock();
            current_node_index = parent_index;
            sibling_index = current_root_index;
            continue;
          }
        }
        else {
          sibling_node.lock();
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
            reinterpret_cast<elem_type*>(allocator.address(allocator_, parent_index)), parent_index, tile);
        parent_node.lock();
        parent_node.load(cuda_memory_order::memory_order_relaxed);
        // make sure parent is not garbage and not underflow
        if ((parent_node.is_garbage()) ||
            (parent_node.is_underflow() && parent_index != current_root_index)) {
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
            // TODO mark right_sibling_node as to-garbage-collect
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
            // two sibling nodes are not modified
            // TODO mark left_sibling_node and right_sibling_node as to-garbage-collect
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
            auto new_sibling_index = allocator.allocate(allocator_, 1, tile);
            auto new_sibling_node = node_type(
                reinterpret_cast<elem_type*>(allocator.address(allocator_, new_sibling_index)),
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
            // TODO mark sibling_node to be garbage collected
          }
          parent_node.unlock();
          sibling_node.unlock();
        }
        // now, current_node is not underflow. if it's not border, unlock.
        if (!current_node.is_border()) { current_node.unlock(); }
      }
      assert(!is_current_node_underflow);

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
      #undef is_current_node_underflow
    }
    assert(false);
  }

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
            reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
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
          reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
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
            (!current_node.get_key_end_bit_from_location(num_traversed_children))) {
          // If it's a non-leaf node, push the next node
          // If it's a leaf node but non-last-slice entry, push the next layer root
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
        bool before_key_end_bit = false;
        for (uint16_t i = 0; i < num_keys; i++) {
          auto key = node.get_key_from_location(i);
          auto key_end_bit = node.get_key_end_bit_from_location(i);
          if (i > 0) {
            assert(before_key <= key);
            if (before_key == key) {
              assert(
                (before_key_end_bit == false && key_end_bit == true) ||
                (before_key_end_bit == true && key_end_bit == true)
              );
            }
          }
          before_key = key;
          before_key_end_bit = key_end_bit;
          if (node.get_key_end_bit_from_location(i)) {
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
  template <typename tile_type, typename node_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool traverse_side_links(node_type& node,
                                            size_type& node_index,
                                            const key_slice_type& key_slice,
                                            const tile_type& tile,
                                            DeviceAllocator& allocator) {
    bool traversed = false;
    while (node.traverse_required(key_slice)) {
      node_index = node.get_sibling_index();
      node =
          node_type(reinterpret_cast<key_slice_type*>(allocator.address(allocator_, node_index)), node_index, tile);
      node.load(cuda_memory_order::memory_order_relaxed);
      traversed |= true;
    }
    return traversed;
  }

  // Tries to traverse the side-links with locks
  // Return true if a side-link was traversed
  template <typename tile_type, typename node_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool traverse_side_links_with_locks(node_type& node,
                                                       size_type& node_index,
                                                       const key_slice_type& key_slice,
                                                       const tile_type& tile,
                                                       DeviceAllocator& allocator) {
    bool traversed = false;
    while (node.traverse_required(key_slice)) {
      node_index = node.get_sibling_index();
      node_type sibling_node =
          node_type(reinterpret_cast<key_slice_type*>(allocator.address(allocator_, node_index)), node_index, tile);
      sibling_node.lock();
      node.unlock();
      node = sibling_node;
      node.load(cuda_memory_order::memory_order_relaxed);
      traversed |= true;
    }
    return traversed;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER void allocate_root_node(const tile_type& tile, DeviceAllocator& allocator) {
    auto root_index = allocator.allocate(allocator_, 1, tile);
    *d_root_index_ = root_index;
    using node_type = masstree_node<tile_type>;

    auto root_node =
        node_type(reinterpret_cast<elem_type*>(allocator.address(allocator_, root_index)),
                  root_index,
                  tile);
    root_node.initialize_root();
    root_node.store();
  }

  void allocate() {
    d_root_index_ = cuda_allocator<size_type>().allocate(1);
    cuda_try(cudaMemset(d_root_index_, 0x00, sizeof(size_type)));
    root_index_ = std::shared_ptr<size_type>(d_root_index_, cuda_deleter<size_type>());
    initialize();
  }

  void initialize() {
    const uint32_t num_blocks = 1;
    const uint32_t block_size = cg_tile_size;
    kernels::initialize_kernel<<<num_blocks, block_size>>>(*this);
    cuda_try(cudaDeviceSynchronize());
  }

  std::shared_ptr<size_type> root_index_;
  size_type* d_root_index_;
  allocator_type allocator_;

  template <typename btree>
  friend __global__ void kernels::initialize_kernel(btree);

  template <typename key_slice_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::masstree_insert_kernel(const key_slice_type* keys,
                                                         const size_type max_key_length,
                                                         const size_type* key_lengths,
                                                         const value_type* values,
                                                         const size_type keys_count,
                                                         btree tree);

  template <typename key_slice_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::masstree_find_kernel(const key_slice_type* keys,
                                                       const size_type max_key_length,
                                                       const size_type* key_lengths,
                                                       value_type* values,
                                                       const size_type keys_count,
                                                       btree tree,
                                                       bool concurrent);

  template <bool do_merge, bool do_remove_empty_root, typename key_slice_type, typename size_type, typename btree>
  friend __global__ void kernels::masstree_erase_kernel(const key_slice_type* keys,
                                                        const size_type max_key_length,
                                                        const size_type* key_lengths,
                                                        const size_type keys_count,
                                                        btree tree,
                                                        bool concurrent);

}; // struct gpu_masstree

} // namespace GPUBTree
