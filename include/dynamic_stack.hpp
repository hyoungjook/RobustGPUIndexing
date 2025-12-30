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
#include <cstdint>
#include <macros.hpp>

namespace utils {

/*
 *    Device-side dynamic stack using the device allocator
 *  The first 31 elements are stored in the register.
 *  After then, the values are spilled to memory using the
 *  allocator.
 */
template <typename tile_type, typename DeviceAllocator>
struct dynamic_stack {
  using element_type = uint32_t;
  static constexpr uint32_t elems_per_node_ = 32;
  static constexpr uint32_t node_capacity_ = elems_per_node_ - 1;

  DEVICE_QUALIFIER dynamic_stack(DeviceAllocator& allocator, const tile_type& tile)
      : allocator_(allocator), tile_(tile) {
    assert(tile.size() == elems_per_node_);
  }

  DEVICE_QUALIFIER void push(const element_type& value) {
    // if register buffer is full, spill to memory
    if (top_ == (node_capacity_ - 1)) {
      auto new_head = allocator_.allocate(tile_);
      auto* node_ptr = reinterpret_cast<stack_node*>(allocator_.address(new_head));
      write_register_to_node(node_ptr);
      set_head(new_head);
      top_ = -1;
    }
    // push element to the register buffer
    top_++;
    if (tile_.thread_rank() == top_) {
      lane_elem_ = value;
    }
  }

  DEVICE_QUALIFIER element_type pop() {
    // if register buffer is empty, load from memory
    if (top_ < 0) {
      auto old_head = get_head();
      assert(old_head != invalid_index);
      auto* node_ptr = reinterpret_cast<stack_node*>(allocator_.address(old_head));
      read_register_from_node(node_ptr);
      allocator_.deallocate(old_head);
      top_ = node_capacity_ - 1;
    }
    // pop element from the register buffer
    auto value = tile_.shfl(lane_elem_, top_);
    top_--;
    return value;
  }

  DEVICE_QUALIFIER void destroy() {
    auto head = get_head();
    while (head != invalid_index) {
      auto node_ptr = reinterpret_cast<stack_node*>(allocator_.address(head));
      auto next = node_ptr->get_next();
      allocator_.deallocate(head);
      head = next;
    }
  }

private:
  static constexpr element_type invalid_index = std::numeric_limits<element_type>::max();
  static constexpr uint32_t loc_of_next_in_node_ = elems_per_node_ - 1;
  struct stack_node {
    // first 31 elements are values, the last 1 element is the next node pointer.
    element_type elems_[elems_per_node_];
    DEVICE_QUALIFIER element_type get_next() const { return elems_[loc_of_next_in_node_]; }
  };
  DEVICE_QUALIFIER element_type get_head() const {
    return tile_.shfl(lane_elem_, loc_of_next_in_node_);
  }
  DEVICE_QUALIFIER void set_head(const element_type& head) {
    if (tile_.thread_rank() == loc_of_next_in_node_) { lane_elem_ = head; }
  }
  DEVICE_QUALIFIER void write_register_to_node(stack_node* ptr) {
    ptr->elems_[tile_.thread_rank()] = lane_elem_;
  }
  DEVICE_QUALIFIER void read_register_from_node(stack_node* ptr) {
    lane_elem_ = ptr->elems_[tile_.thread_rank()];
  }

  int top_ = -1;
  // stack_node stored in register, i'th elem in i'th lane
  element_type lane_elem_ = invalid_index;
  DeviceAllocator& allocator_;
  const tile_type& tile_;
};

} // namespace utils
