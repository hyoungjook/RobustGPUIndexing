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
  void insert_fixlen(const key_slice_type* keys,
                     const size_type key_length,
                     const value_type* values,
                     const size_type num_keys,
                     cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::insert_fixlen_kernel<<<num_blocks, block_size, 0, stream>>>(keys, key_length, values, num_keys, *this);
  }

  void find_fixlen(const key_slice_type* keys,
                   const size_type key_length,
                   value_type* values,
                   const size_type num_keys,
                   cudaStream_t stream = 0,
                   bool concurrent = false) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::find_fixlen_kernel<<<num_blocks, block_size, 0, stream>>>(keys, key_length, values, num_keys, *this, concurrent);
  }

  void insert_varlen(const key_slice_type* keys,
                     const size_type max_key_length,
                     const size_type* key_lengths,
                     const value_type* values,
                     const size_type num_keys,
                     cudaStream_t stream = 0) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::insert_varlen_kernel<<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, values, num_keys, *this);
  }

  void find_varlen(const key_slice_type* keys,
                   const size_type max_key_length,
                   const size_type* key_lengths,
                   value_type* values,
                   const size_type num_keys,
                   cudaStream_t stream = 0,
                   bool concurrent = false) {
    const uint32_t block_size = 512;
    const uint32_t num_blocks = (num_keys + block_size - 1) / block_size;
    kernels::find_varlen_kernel<<<num_blocks, block_size, 0, stream>>>(keys, max_key_length, key_lengths, values, num_keys, *this, concurrent);
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
        if (current_node.is_leaf()) {
          const bool found = current_node.get_key_value_from_node(key_slice, current_node_index, last_slice);
          if (!found) {
            // not exists
            return invalid_value;
          }
          else {
            // value retrieved in current_node_index. continue to next layer.
            break;
          }
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

        bool is_leaf = current_node.is_leaf();
        if (is_leaf) {
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
            is_leaf = current_node.is_leaf();
            // if the node is not a leaf anymore, we don't need the lock
            if (!is_leaf) { current_node.unlock(); }
            // traversal while holding the lock
            while (current_node.traverse_required(key_slice)) {
              if (is_leaf) { current_node.unlock(); }
              current_node_index = current_node.get_sibling_index();
              current_node       = node_type(
                  reinterpret_cast<elem_type*>(allocator.address(allocator_, current_node_index)),
                  current_node_index,
                  tile);
              if (is_leaf) { current_node.lock(); }
              current_node.load(cuda_memory_order::memory_order_relaxed);
              is_leaf = current_node.is_leaf();
              // if the node is not a leaf anymore, we don't need the lock
              if (!is_leaf) { current_node.unlock(); }
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
          if (is_leaf) {
            current_node.unlock();
            current_node_index = current_root_index;
            parent_index       = current_root_index;
            continue;
          }
        }

        // if is full, and not leaf we need to acquire the lock
        if (is_full && !is_leaf) {
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

        // splitting an intermediate node
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
              reinterpret_cast<elem_type*>(allocator.address(allocator_, parent_index)),
              true);

          split_result.sibling.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          current_node.store(cuda_memory_order::memory_order_relaxed);
          __threadfence();
          split_result.parent.store(cuda_memory_order::memory_order_relaxed);
          split_result.parent.unlock();

          if (current_node.key_is_in_upperhalf(split_result.pivot_key, key_slice)) {
            current_node_index = sibling_index;
            current_node.unlock();
            current_node = split_result.sibling;
          } else {
            split_result.sibling.unlock();
          }

          is_leaf = current_node.is_leaf();
          if (!is_leaf) { current_node.unlock(); }
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
          is_leaf      = current_node.is_leaf();
          if (!is_leaf) { current_node.unlock(); }
        }

        // traversal and insertion
        is_leaf = current_node.is_leaf();
        if (is_leaf) {
          value_type layer_value = value;
          current_node.insert_leaf(key_slice, layer_value, last_slice, allocator, allocator_);
          current_node.store(cuda_memory_order::memory_order_relaxed);
          current_node.unlock();
          if (last_slice) {
            // we inserted value to the last layer
            return true;
          }
          else {
            // layer_value points to the next root node
            current_root_index = layer_value;
            break; // move on to the next layer
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
        if ((!current_node.is_leaf()) ||
            (!current_node.get_key_meta_bit_from_location(num_traversed_children))) {
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
      if (node.is_leaf()) {
        uint16_t num_keys = node.num_keys();
        key_slice_type before_key = 0;
        bool before_key_meta = false;
        for (uint16_t i = 0; i < num_keys; i++) {
          auto key = node.get_key_from_location(i);
          auto key_meta = node.get_key_meta_bit_from_location(i);
          if (i > 0) {
            assert(before_key <= key);
            if (before_key == key) {
              assert(
                (before_key_meta == false && key_meta == true) ||
                (before_key_meta == true && key_meta == true)
              );
            }
          }
          before_key = key;
          before_key_meta = key_meta;
          if (node.get_key_meta_bit_from_location(i)) {
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
  friend __global__ void kernels::insert_fixlen_kernel(const key_slice_type* keys,
                                                       const size_type key_length,
                                                       const value_type* values,
                                                       const size_type keys_count,
                                                       btree tree);

  template <typename key_slice_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::find_fixlen_kernel(const key_slice_type* keys,
                                                     const size_type key_length,
                                                     value_type* values,
                                                     const size_type keys_count,
                                                     btree tree,
                                                     bool concurrent);

  template <typename key_slice_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::insert_varlen_kernel(const key_slice_type* keys,
                                              const size_type max_key_length,
                                              const size_type* key_lengths,
                                              const value_type* values,
                                              const size_type keys_count,
                                              btree tree);

  template <typename key_slice_type, typename value_type, typename size_type, typename btree>
  friend __global__ void kernels::find_varlen_kernel(const key_slice_type* keys,
                                            const size_type max_key_length,
                                            const size_type* key_lengths,
                                            value_type* values,
                                            const size_type keys_count,
                                            btree tree,
                                            bool concurrent);
}; // struct gpu_masstree

} // namespace GPUBTree
