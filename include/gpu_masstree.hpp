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
    kernels::masstree_erase_kernel<false><<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, num_keys, *this, concurrent);
  }

  void erase_merge(const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   const size_type num_keys,
                   cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::masstree_erase_kernel<true><<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, num_keys, *this, false);
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
      const bool last_slice = (slice == key_length - 1);
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
          const bool found = current_node.get_key_value_from_node(key_slice, current_node_index, last_slice);
          if (!found) return invalid_value; // not exists
          else break; // value in current_node_index. continue to next layer.
        }
        else {
          current_node_index = current_node.find_next(key_slice);
        }
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
    size_type current_root_index = *d_root_index_;
    bool link_traversed = false;
    for (size_type slice = 0; slice < key_length; slice++) {
      const key_slice_type key_slice = key[slice];
      const bool last_slice = (slice == key_length - 1);
      size_type current_node_index = current_root_index;
      size_type parent_index = current_root_index;
      while (true) {
        auto current_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
            current_node_index,
            tile);
        current_node.load(cuda_memory_order::memory_order_relaxed);

        // if we restarted from root, we reset the traversal
        link_traversed = current_node_index == current_root_index ? false : link_traversed;

        // Traversing side-links
        link_traversed |= traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

        bool is_border = current_node.is_border();
        if (is_border) {
          // If it's not last slice, first check if the entry exists without lock
          if (!last_slice) {
            size_type next_root_index;
            if (current_node.get_key_value_from_node(key_slice, next_root_index, last_slice)) {
              // we found the next layer index, move on
              current_root_index = next_root_index;
              break;
            }
            // entry doesn't exist, we should insert, so try to lock the node
          }
          if (current_node.try_lock()) {
            current_node.load(cuda_memory_order::memory_order_relaxed);
            bool parent_unknown =
                current_node_index == parent_index && current_node_index != current_root_index;
            bool traversal_required = current_node.traverse_required(key_slice);
            // if the parent is unknown we will not proceed
            if (parent_unknown && traversal_required) {
              current_node.unlock();
              current_node_index = current_root_index;
              parent_index       = current_root_index;
              continue;
            }
            is_border = current_node.is_border();
            // if the node is not a leaf anymore, we don't need the lock
            if (!is_border) { current_node.unlock(); }
            // traversal while holding the lock
            while (current_node.traverse_required(key_slice)) {
              if (is_border) { current_node.unlock(); }
              current_node_index = current_node.get_sibling_index();
              current_node       = node_type(
                  reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
                  current_node_index,
                  tile);
              if (is_border) { current_node.lock(); }
              current_node.load(cuda_memory_order::memory_order_relaxed);
              is_border = current_node.is_border();
              // if the node is not a leaf anymore, we don't need the lock
              if (!is_border) { current_node.unlock(); }
              link_traversed = true;
            }
          }
          else {
            current_node_index = parent_index;
            continue;
          }
        }

        // make sure that if the node is full, we know the parent
        // we only know the parent if we didn't do side-traversal
        bool is_full = current_node.is_full();
        if (is_full && link_traversed) {
          if (is_border) {
            current_node.unlock();
            current_node_index = current_root_index;
            parent_index       = current_root_index;
            continue;
          }
        }

        // if is full, and not leaf we need to acquire the lock
        if (is_full && !is_border) {
          if (current_node.try_lock()) {
            current_node.load(cuda_memory_order::memory_order_relaxed);
            is_full = current_node.is_full();
            if (is_full) {
              // if we traverse, parent will change so we will restart
              if (current_node.traverse_required(key_slice)) {
                current_node.unlock();
                current_node_index = current_root_index;
                parent_index       = current_root_index;
                continue;
              }
            } else {
              current_node.unlock();
              // Traversing side-links
              link_traversed |=
                  traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);
            }
          } else {
            current_node_index = parent_index;
            continue;
          }
        }

        is_full = current_node.is_full();
        // if the node full after we restarted we can't proceed
        if (is_full && (current_node_index != current_root_index) && (current_node_index == parent_index)) {
          current_node.unlock();
          current_node_index = current_root_index;
          parent_index       = current_root_index;
          continue;
        }

        // splitting a non-root node
        if (is_full && (current_node_index != current_root_index)) {
          auto parent_node = node_type(
              reinterpret_cast<elem_type*>(allocator.address(allocator_, parent_index)), parent_index, tile);
          parent_node.lock();
          parent_node.load(cuda_memory_order::memory_order_relaxed);
          bool parent_is_full = parent_node.is_full();

          // make sure parent is not full
          if (parent_is_full) {
            current_node.unlock();
            parent_node.unlock();
            current_node_index = current_root_index;
            parent_index       = current_root_index;
            continue;
          }

          // make sure parent is correct parent
          auto parent_is_correct = parent_node.ptr_is_in_node(current_node_index);
          if (!parent_is_correct) {
            current_node.unlock();
            parent_node.unlock();
            current_node_index = current_root_index;
            parent_index       = current_root_index;
            continue;
          }

          // now it is safe to split
          auto sibling_index = allocator.allocate(allocator_, 1, tile);

          auto split_result = current_node.split(
              sibling_index,
              parent_index,
              reinterpret_cast<elem_type*>(allocator.address(allocator_, sibling_index)),
              parent_node,
              true);

          split_result.sibling.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          current_node.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          parent_node.store(cuda_memory_order::memory_order_relaxed);
          parent_node.unlock();

          if (current_node.key_is_in_upperhalf(split_result.pivot_key, key_slice)) {
            current_node_index = sibling_index;
            current_node.unlock();
            current_node = split_result.sibling;
          } else {
            split_result.sibling.unlock();
          }

          is_border = current_node.is_border();
          if (!is_border) { current_node.unlock(); }
        } else if (is_full) {
          auto sibling_index0 = allocator.allocate(allocator_, 1, tile);
          auto sibling_index1 = allocator.allocate(allocator_, 1, tile);
          auto two_siblings =
              current_node.split_as_root(sibling_index0,  // left node
                                         sibling_index1,  // left right
                                         reinterpret_cast<elem_type*>(allocator.address(
                                             allocator_, sibling_index0)),  // left ptr
                                         reinterpret_cast<elem_type*>(allocator.address(
                                             allocator_, sibling_index1)),  // right ptr
                                         true);                             // children_are_locked

          // write order here should be:
          // right node -> left node ->  parent
          two_siblings.right.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          two_siblings.left.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          current_node.store(cuda_memory_order::memory_order_relaxed);  // root is still locked
          current_node.unlock();

          // go right or left?
          current_node_index = current_node.find_next(key_slice);
          if (current_node_index == sibling_index0) {  // go left
            two_siblings.right.unlock();
            current_node = two_siblings.left;
          } else {  // go right
            two_siblings.left.unlock();
            current_node = two_siblings.right;
          }
          parent_index = current_root_index;
          is_border      = current_node.is_border();
          if (!is_border) { current_node.unlock(); }
        }

        // traversal and insertion
        is_border = current_node.is_border();
        if (is_border) {
          value_type value_to_insert;
          if (!last_slice) {
            value_type next_root_index;
            // if not last slice, check if the entry exists
            if (current_node.get_key_value_from_node(key_slice, next_root_index, last_slice)) {
              // found, move on to next layer
              current_node.unlock();
              current_root_index = next_root_index; // next_root_index
              break;
            }
            else {
              // not found: allocate next root node and insert it
              next_root_index = allocator.allocate(allocator_, 1, tile);
              auto next_root_node = masstree_node<tile_type>(
                reinterpret_cast<elem_type*>(allocator.address(allocator_, next_root_index)),
                next_root_index,
                tile);
              next_root_node.initialize_root();
              next_root_node.store(cuda_memory_order::memory_order_relaxed);
              __threadfence();
              value_to_insert = next_root_index;
            }
          }
          else {
            // last slice, leaf node: insert the value
            value_to_insert = value;
          }
          current_node.insert(key_slice, value_to_insert, last_slice);
          current_node.store(cuda_memory_order::memory_order_relaxed);
          current_node.unlock();
          if (!last_slice) {
            // move on to the next layer
            current_root_index = value_to_insert; // == next_root_index
            break;
          }
          else {
            // we inserted value to the last layer
            return true;
          }
        } else {  // traverse
          parent_index       = link_traversed ? current_root_index : current_node_index;
          current_node_index = current_node.find_next(key_slice);
        }
      } // while (true)
    } // for each slice
    assert(false);
    return false;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_erase(const key_slice_type* key,
                                          const size_type key_length,
                                          const tile_type& tile,
                                          DeviceAllocator& allocator,
                                          bool concurrent = false) {
    using node_type = masstree_node<tile_type>;
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length; slice++) {
      const key_slice_type key_slice = key[slice];
      const bool last_slice = (slice == key_length - 1);
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
          if (last_slice) {
            current_node.lock();
            current_node.load(cuda_memory_order::memory_order_relaxed);
            if (concurrent) {
              traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
            }
            const bool success = current_node.erase(key_slice);
            if (success) {
              current_node.store(cuda_memory_order::memory_order_relaxed);
            }
            current_node.unlock();
            return success;
          }
          else {
            const bool found = current_node.get_key_value_from_node(key_slice, current_node_index, false);
            if (!found) return false; // not exists
            else break; // value in current_node_index. continue to next layer.
          }
        }
        else {
          current_node_index = current_node.find_next(key_slice);
        }
      }
    }
    assert(false);
    return false;
  }

  template <typename tile_type, typename DeviceAllocator>
  DEVICE_QUALIFIER bool cooperative_erase_merge(const key_slice_type* key,
                                                const size_type key_length,
                                                const tile_type& tile,
                                                DeviceAllocator& allocator) {
    using node_type = masstree_node<tile_type>;
    // read-only traverse before the last slice
    size_type current_node_index = *d_root_index_;
    for (size_type slice = 0; slice < key_length - 1; slice++) {
      const key_slice_type key_slice = key[slice];
      while (true) {
        node_type current_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
            current_node_index,
            tile);
        current_node.load(cuda_memory_order::memory_order_relaxed);
        traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);
        if (current_node.is_border()) {
          const bool found = current_node.get_key_value_from_node(key_slice, current_node_index, false);
          if (!found) return false; // not exists
          else break; // value in current_node_index. continue to next layer.
        }
        else {
          current_node_index = current_node.find_next(key_slice);
        }
      }
    }
    // now erase and merge the entry at the last_slice layer
    {
      size_type current_root_index = current_node_index;
      bool link_traversed = false;
      const key_slice_type key_slice = key[key_length - 1];
      size_type parent_index = current_root_index;
      size_type sibling_index = current_root_index;
      bool sibling_at_left = false;
      while (true) {
        node_type current_node = node_type(
            reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
            current_node_index,
            tile);
        current_node.load(cuda_memory_order::memory_order_relaxed);

        // if we restarted from root, we reset the traversal
        link_traversed = current_node_index == current_root_index ? false : link_traversed;

        // Traversing side-links
        link_traversed |= traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);

        bool is_border = current_node.is_border();
        if (is_border) {
          if (current_node.try_lock()) {
            current_node.load(cuda_memory_order::memory_order_relaxed);
            bool parent_unknown =
                current_node_index == parent_index && current_node_index != current_root_index;
            bool traverse_required = current_node.traverse_required(key_slice);
            // if the parent is unknown we will not proceed
            if (parent_unknown && traverse_required) {
              current_node.unlock();
              current_node_index = current_root_index;
              parent_index = current_root_index;
              sibling_index = current_root_index;
              continue;
            }
            is_border = current_node.is_border();
            // if the node is not a leaf anymore, we don't need the lock
            if (!is_border) { current_node.unlock(); }
            // traversal while holding the lock
            while (current_node.traverse_required(key_slice)) {
              if (is_border) { current_node.unlock(); }
              current_node_index = current_node.get_sibling_index();
              current_node       = node_type(
                  reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
                  current_node_index,
                  tile);
              if (is_border) { current_node.lock(); }
              current_node.load(cuda_memory_order::memory_order_relaxed);
              is_border = current_node.is_border();
              // if the node is not a leaf anymore, we don't need the lock
              if (!is_border) { current_node.unlock(); }
              link_traversed = true;
            }
          }
          else {
            current_node_index = parent_index;
            sibling_index = current_root_index;
            continue;
          }
        }

        // make sure that if the node is underflow, we know the parent
        // we only know the parent if we didn't do side-traversal
        bool is_underflow = (current_node_index != current_root_index) && current_node.is_underflow();
        if (is_underflow && link_traversed) {
          if (is_border) {
            current_node.unlock();
            current_node_index = current_root_index;
            parent_index = current_root_index;
            sibling_index = current_root_index;
            continue;
          }
        }

        // if is underflow and not leaf, we need to acquire the lock
        if (is_underflow && !is_border) {
          if (current_node.try_lock()) {
            current_node.load(cuda_memory_order::memory_order_relaxed);
            is_underflow = current_node.is_underflow();
            if (is_underflow) {
              // if we traverse, parent will change so we will restart
              if (current_node.traverse_required(key_slice)) {
                current_node.unlock();
                current_node_index = current_root_index;
                parent_index = current_root_index;
                sibling_index = current_root_index;
                continue;
              }
            }
            else {
              current_node.unlock();
              // Traversing side links
              link_traversed |=
                  traverse_side_links(current_node, current_node_index, key_slice, tile, allocator);
            }
          }
          else {
            current_node_index = parent_index;
            sibling_index = current_root_index;
            continue;
          }
        }

        is_underflow = (current_node_index != current_root_index) && current_node.is_underflow();
        // if the node is underflow after we restarted we can't proceed
        if (is_underflow && (current_node_index == parent_index)) {
          current_node.unlock();
          current_node_index = current_root_index;
          parent_index = current_root_index;
          sibling_index = current_root_index;
          continue;
        }

        if (is_underflow) {
          assert(sibling_index != current_root_index);
          // first lock the sibling
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
              current_node.borrow_right(sibling_node,
                                        parent_node,
                                        plan.left_location,
                                        new_sibling_index,
                                        reinterpret_cast<elem_type*>(allocator.address(allocator_, new_sibling_index)));
              sibling_node.store(cuda_memory_order::memory_order_relaxed);
              __threadfence();
              current_node.store(cuda_memory_order::memory_order_relaxed);
              __threadfence();
              parent_node.store(cuda_memory_order::memory_order_relaxed);
            }
            parent_node.unlock();
            sibling_node.unlock();
          }

          is_border = current_node.is_border();
          if (!is_border) { current_node.unlock(); }
        }

        // traversal and erase
        is_border = current_node.is_border();
        if (is_border) {
          // current_node is already locked
          traverse_side_links_with_locks(current_node, current_node_index, key_slice, tile, allocator);
          const bool success = current_node.erase(key_slice);
          if (success) {
            current_node.store(cuda_memory_order::memory_order_relaxed);
          }
          current_node.unlock();
          return success;
        }
        else { // traverse
          parent_index = link_traversed ? current_root_index : current_node_index;
          current_node_index = current_node.find_next_and_sibling(
              key_slice, sibling_index, sibling_at_left);
        }
      }
    }
    assert(false);
    return false;
  }

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

  template <bool do_merge, typename key_slice_type, typename size_type, typename btree>
  friend __global__ void kernels::masstree_erase_kernel(const key_slice_type* keys,
                                                        const size_type max_key_length,
                                                        const size_type* key_lengths,
                                                        const size_type keys_count,
                                                        btree tree,
                                                        bool concurrent);

}; // struct gpu_masstree

} // namespace GPUBTree
