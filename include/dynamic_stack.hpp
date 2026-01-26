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
template <int N, typename tile_type, typename DeviceAllocator>
struct dynamic_stack_u32 {
  using element_type = uint32_t;
  static constexpr uint32_t elems_per_node_ = 32;
  static constexpr uint32_t node_capacity_ = elems_per_node_ - 1;

  DEVICE_QUALIFIER dynamic_stack_u32(DeviceAllocator& allocator, const tile_type& tile)
      : allocator_(allocator), tile_(tile) {
    assert(tile.size() == elems_per_node_);
    for (int i = 0; i < N; i++) { lane_elem_[i] = invalid_index; }
  }

  template <typename... Ts>
  DEVICE_QUALIFIER void push(const Ts&... values) {
    // if register buffer is full, spill to memory
    if (top_ == (node_capacity_ - 1)) {
      for (int i = 0; i < N; i++) {
        auto new_head = allocator_.allocate(tile_);
        auto* node_ptr = reinterpret_cast<stack_node*>(allocator_.address(new_head));
        write_register_to_node(i, node_ptr);
        set_head(i, new_head);
      }
      top_ = -1;
    }
    // push element to the register buffer
    top_++;
    if (tile_.thread_rank() == top_) {
      int i = 0;
      ((lane_elem_[i++] = values), ...);
    }
  }

  template <typename... Ts>
  DEVICE_QUALIFIER void pop(Ts&... values) {
    // if register buffer is empty, load from memory
    if (top_ < 0) {
      for (int i = 0; i < N; i++) {
        auto old_head = get_head(i);
        assert(old_head != invalid_index);
        auto* node_ptr = reinterpret_cast<stack_node*>(allocator_.address(old_head));
        read_register_from_node(i, node_ptr);
        allocator_.deallocate(old_head);
      }
      top_ = node_capacity_ - 1;
    }
    // pop element from the register buffer
    int i = 0;
    ((values = tile_.shfl(lane_elem_[i++], top_)), ...);
    top_--;
  }

  DEVICE_QUALIFIER void destroy() {
    for (int i = 0; i < N; i++) {
      auto head = get_head(i);
      while (head != invalid_index) {
        auto node_ptr = reinterpret_cast<stack_node*>(allocator_.address(head));
        auto next = node_ptr->get_next();
        allocator_.deallocate(head);
        head = next;
      }
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
  DEVICE_QUALIFIER element_type get_head(int i) const {
    return tile_.shfl(lane_elem_[i], loc_of_next_in_node_);
  }
  DEVICE_QUALIFIER void set_head(int i,const element_type& head) {
    if (tile_.thread_rank() == loc_of_next_in_node_) { lane_elem_[i] = head; }
  }
  DEVICE_QUALIFIER void write_register_to_node(int i, stack_node* ptr) {
    ptr->elems_[tile_.thread_rank()] = lane_elem_[i];
  }
  DEVICE_QUALIFIER void read_register_from_node(int i, stack_node* ptr) {
    lane_elem_[i] = ptr->elems_[tile_.thread_rank()];
  }

  int top_ = -1;
  // stack_node stored in register, i'th elem in i'th lane
  element_type lane_elem_[N];
  DeviceAllocator& allocator_;
  const tile_type& tile_;

public:
  template <int key_slice_idx, typename dynamic_stack_type>
  friend DEVICE_QUALIFIER void fill_output_keys_from_key_slice_stack(const dynamic_stack_type& s,
                                                                     typename dynamic_stack_type::element_type* out_keys,
                                                                     uint32_t out_key_max_length,
                                                                     uint32_t layer,
                                                                     uint32_t count);
};

template <int key_slice_idx, typename dynamic_stack_type>
DEVICE_QUALIFIER void fill_output_keys_from_key_slice_stack(const dynamic_stack_type& s,
                                                            typename dynamic_stack_type::element_type* out_keys,
                                                            uint32_t out_key_max_length,
                                                            uint32_t layer,
                                                            uint32_t count) {
  // used for gpu_masstree::cooperative_range()
  if (layer == 0 || count == 0) { return; }
  auto lane_elem = s.lane_elem_[key_slice_idx];
  int top = s.top_;
  while (true) {
    // store stack_register[0, top] -> out_keys[layer-top-1, layer-1]
    assert(layer >= top + 1);
    if (top >= 0) {
      layer -= (top + 1);
      for (uint32_t i = 0; i < count; i++) {
        if (s.tile_.thread_rank() <= top) {
          out_keys[i * out_key_max_length + layer + s.tile_.thread_rank()] = lane_elem;
        }
      }
    }
    // fetch node into register
    auto head = s.tile_.shfl(lane_elem, dynamic_stack_type::loc_of_next_in_node_);
    if (head == dynamic_stack_type::invalid_index) {
      assert(layer == 0);
      break;
    }
    auto node_ptr = reinterpret_cast<typename dynamic_stack_type::stack_node*>(s.allocator_.address(head));
    lane_elem = node_ptr->elems_[s.tile_.thread_rank()];
    top = dynamic_stack_type::node_capacity_ - 1;
  }
}

} // namespace utils
