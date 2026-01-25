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
#include <cstdint>
#include <macros.hpp>
#include <memory_utils.hpp>
#include <utils.hpp>

#ifndef NDEBUG
//#define NODE_DEBUG
#endif
template <typename tile_type, typename allocator_type>
struct masstree_suffix_node {
  using elem_type = uint32_t;
  using size_type = uint32_t;
  static constexpr int node_width = 32;

  DEVICE_QUALIFIER masstree_suffix_node(elem_type* ptr, const size_type index, const tile_type& tile, allocator_type& allocator)
      : node_ptr_(ptr)
      #ifdef NODE_DEBUG
      , node_index_(index)
      #endif
      , tile_(tile)
      , allocator_(allocator) {}

  DEVICE_QUALIFIER void load(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    lane_elem_ = cuda_memory<elem_type>::load(node_ptr_ + tile_.thread_rank(), order);
  }
  DEVICE_QUALIFIER void store_head(cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    cuda_memory<elem_type>::store(node_ptr_ + tile_.thread_rank(), lane_elem_, order);
  }  

  DEVICE_QUALIFIER size_type get_next() const {
    return tile_.shfl(lane_elem_, next_lane_);
  }
  DEVICE_QUALIFIER size_type get_key_length() const {
    return tile_.shfl(lane_elem_, head_node_length_lane_);
  }
  DEVICE_QUALIFIER void set_length(const size_type& key_length) {
    if (tile_.thread_rank() == head_node_length_lane_) {
      lane_elem_ = key_length;
    }
  }
  DEVICE_QUALIFIER elem_type get_value() const {
    return tile_.shfl(lane_elem_, head_node_value_lane_);
  }
  DEVICE_QUALIFIER void update_value(const elem_type& value) {
    if (tile_.thread_rank() == head_node_value_lane_) {
      lane_elem_ = value;
    }
  }

  DEVICE_QUALIFIER bool streq(const elem_type* key, uint32_t key_length, cuda_memory_order order = cuda_memory_order::memory_order_weak) const {
    // ignore first slice in the border node
    key++;
    key_length--;
    if (get_key_length() != key_length) { return false; }
    // now key_length == this_key_length, compare head node
    auto elem = lane_elem_;
    while (true) {
      bool mismatch = (tile_.thread_rank() < node_max_len_) &&
                      (tile_.thread_rank() < key_length) && 
                      (elem != key[tile_.thread_rank()]);
      uint32_t mismatch_ballot = tile_.ballot(mismatch);
      if (mismatch_ballot != 0) { return false; }
      if (key_length <= node_max_len_) { return true; }
      key_length -= node_max_len_;
      key += node_max_len_;
      auto next_index = get_next();
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = cuda_memory<elem_type>::load(next_ptr + tile_.thread_rank(), order);
    }
    assert(false);
  }

  DEVICE_QUALIFIER int strcmp(const elem_type* key, uint32_t key_length, cuda_memory_order order = cuda_memory_order::memory_order_weak, elem_type* mismatch_value = nullptr) const {
    // ignore first slice in the border node
    key++;
    key_length--;
    // strcmp(this, key) -> 0 (match), +(this<key), -(this>key)
    // the absolute of return value: (1 + num_matches)
    // NOTE if one is prefix of the other, num_matches is (len(smaller) - 1)
    auto this_length = get_key_length();
    auto cmp_length = min(this_length, key_length);
    auto elem = lane_elem_;
    int total_num_matches = 0;
    while (true) {
      // compare elements
      bool this_more = (tile_.thread_rank() < this_length - 1);
      bool key_more = (tile_.thread_rank() < key_length - 1);
      bool valid_cmp = (tile_.thread_rank() < node_max_len_) &&
                       (tile_.thread_rank() < cmp_length);
      elem_type other = valid_cmp ? (key[tile_.thread_rank()]) : 0;
      bool match = valid_cmp &&
                   (elem == other) &&
                   (this_more == key_more);
      uint32_t match_ballot = tile_.ballot(match);
      int num_matches = __ffs(~match_ballot) - 1;
      // if all elements match
      if (num_matches == cmp_length) { return 0; }
      // if found mismatch in this node
      total_num_matches += num_matches;
      if (num_matches < node_max_len_ && num_matches < cmp_length) {
        // num_matches'th lane has the first mismatch elems
        bool ge = tile_.shfl((elem < other) || (elem == other && this_more < key_more), num_matches);
        if (mismatch_value) { *mismatch_value = tile_.shfl(elem, num_matches); }
        return (ge ? 1 : -1) * static_cast<int>(1 + total_num_matches);
      }
      // proceed to next node
      cmp_length -= node_max_len_;
      this_length -= node_max_len_;
      key_length -= node_max_len_;
      key += node_max_len_;
      auto next_index = tile_.shfl(elem, next_lane_);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      elem = cuda_memory<elem_type>::load(next_ptr + tile_.thread_rank(), order);
    }
    assert(false);
  }

  DEVICE_QUALIFIER void create_from(const elem_type* key, size_type key_length, elem_type value, cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    // ignore first slice, already stored in border node
    key++;
    key_length--;
    // head node metadata
    update_value(value);
    set_length(key_length);
    elem_type elem;
    elem_type* curr_ptr = nullptr;  // NULL if head, else appendix
    while (true) {
      // set elem
      if (tile_.thread_rank() < min(key_length, node_max_len_)) {
        elem = key[tile_.thread_rank()];
      }
      elem_type* next_ptr;
      if (key_length > node_max_len_) {
        auto next_index = allocator_.allocate(tile_);
        if (tile_.thread_rank() == next_lane_) { elem = next_index; }
        next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
      }
      // store
      if (curr_ptr) { // !is_head
        cuda_memory<elem_type>::store(curr_ptr + tile_.thread_rank(), elem, order);
      }
      else {  // is_head
        if (tile_.thread_rank() < node_max_len_ || tile_.thread_rank() == next_lane_) {
          lane_elem_ = elem;
        }
      }
      // proceed
      if (key_length <= node_max_len_) { break; }
      curr_ptr = next_ptr;
      key_length -= node_max_len_;
      key += node_max_len_;
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER size_type trim(uint32_t offset, size_type old_suffix_index, reclaimer_type& reclaimer, cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    // trim first offset elements and return new_suffix_index
    auto new_length = get_key_length() - offset;
    set_length(new_length);
    // retire nodes until the node with valid element
    size_type new_suffix_index = old_suffix_index;
    elem_type* dst_ptr = nullptr;
    elem_type src_lane_elem = lane_elem_;
    while (offset >= node_max_len_) {
      auto next_index = tile_.shfl(src_lane_elem, next_lane_);
      reclaimer.retire(new_suffix_index, tile_);
      new_suffix_index = next_index;
      dst_ptr = reinterpret_cast<elem_type*>(allocator_.address(new_suffix_index));
      src_lane_elem = cuda_memory<elem_type>::load(dst_ptr + tile_.thread_rank(), order);
      offset -= node_max_len_;
    }
    if (offset == 0) { return new_suffix_index; }
    // new_suffix_index is fixed from now
    // dst_ptr has the pointer of new_suffix_index; nullptr if new_suffix_index is old head
    // src_lane_elem has the contents of new_suffix_index (new head)
    if (dst_ptr) { node_ptr_ = dst_ptr; }
    elem_type dst_lane_elem;
    while (true) {
      // phase 1. copy src[offset:node_max_len) -> dst[0:node_max_len-offset) in same node
      uint32_t copy_count = min(new_length, node_max_len_ - offset);
      dst_lane_elem = tile_.shfl_down(src_lane_elem, offset);
      new_length -= copy_count;
      if (new_length == 0) { break; }
      // phase 2. copy src.next[0:offset) -> dst[node_max_len-offset:node_max_len)
      auto src_next_index = tile_.shfl(src_lane_elem, next_lane_);
      auto* src_next_ptr = reinterpret_cast<elem_type*>(allocator_.address(src_next_index));
      src_lane_elem = cuda_memory<elem_type>::load(src_next_ptr + tile_.thread_rank(), order);
      copy_count = min(new_length, offset);
      auto up_src_elem = tile_.shfl_up(src_lane_elem, node_max_len_ - offset);
      if (node_max_len_ - offset <= tile_.thread_rank()) {
        dst_lane_elem = up_src_elem;
      }
      new_length -= copy_count;
      if (new_length == 0) {
        // if new_length ends here, it means the last node is not used any more
        reclaimer.retire(src_next_index, tile_);
        break;
      }
      // phase 3. store dst
      if (tile_.thread_rank() < node_max_len_) {
        if (dst_ptr) {
          cuda_memory<elem_type>::store(dst_ptr + tile_.thread_rank(), dst_lane_elem, order);
        }
        else {
          lane_elem_ = dst_lane_elem;
        }
      }
      dst_ptr = src_next_ptr;
    }
    // flush dst_lane_elem
    if (tile_.thread_rank() < node_max_len_) {
      if (dst_ptr) {
        cuda_memory<elem_type>::store(dst_ptr + tile_.thread_rank(), dst_lane_elem, order);
      }
      else {
        lane_elem_ = dst_lane_elem;
      }
    }
    return new_suffix_index;
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void retire(reclaimer_type& reclaimer, size_type suffix_head_index, cuda_memory_order order = cuda_memory_order::memory_order_weak) {
    reclaimer.retire(suffix_head_index, tile_);
    auto key_length = get_key_length();
    auto suffix_index = get_next();
    while (true) {
      if (key_length <= node_max_len_) { break; }
      auto* suffix_ptr = reinterpret_cast<elem_type*>(allocator_.address(suffix_index));
      auto next_index = cuda_memory<elem_type>::load(suffix_ptr + next_lane_, order);
      reclaimer.retire(suffix_index, tile_);
      suffix_index = next_index;
      key_length -= node_max_len_;
    }
  }

  static DEVICE_QUALIFIER elem_type fetch_value_only(size_type suffix_index, allocator_type& allocator, bool concurrent) {
    auto* ptr = reinterpret_cast<elem_type*>(allocator.address(suffix_index));
    if (concurrent) {
      return cuda_memory<elem_type>::load(ptr + head_node_value_lane_, cuda_memory_order::memory_order_relaxed);
    }
    else {
      return cuda_memory<elem_type>::load(ptr + head_node_value_lane_, cuda_memory_order::memory_order_weak);
    }
  }

  DEVICE_QUALIFIER masstree_suffix_node<tile_type, allocator_type>& operator=(
      const masstree_suffix_node<tile_type, allocator_type>& other) {
    node_ptr_ = other.node_ptr_;
    #ifdef NODE_DEBUG
    node_index_ = other.node_index_;
    #endif
    lane_elem_ = other.lane_elem_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    #ifdef NODE_DEBUG
    bool lead_lane = (tile_.thread_rank() == 0);
    auto length = get_key_length();
    auto value = get_value();
    if (lead_lane) printf("node[%u]: s{l=%u v=%u; ", node_index_, length, value);
    for (uint32_t i = 0; i < min(length, node_max_len_); i++) {
      elem_type key_slice = tile_.shfl(lane_elem_, i);
      if (lead_lane) printf("%u ", key_slice);
    }
    if (length <= node_max_len_) {
      if (lead_lane) printf("}\n");
    }
    else {
      auto next = get_next();
      if (lead_lane) printf("(n=%u)}\n", next);
      length -= node_max_len_;
      while (true) {
        if (lead_lane) printf("node[%u]: s.{", next);
        auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(next));
        auto elem = next_ptr[tile_.thread_rank()];
        for (uint32_t i = 0; i < min(length, node_max_len_); i++) {
          elem_type key_slice = tile_.shfl(elem, i);
          if (lead_lane) printf("%u ", key_slice);
        }
        if (length <= node_max_len_) {
          if (lead_lane) printf("}\n");
          break;
        }
        else {
          next = tile_.shfl(elem, next_lane_);
          if (lead_lane) printf("(n=%u)}\n", next);
          length -= node_max_len_;
        }
      }
    }
    #endif
  }

 private:
  elem_type* node_ptr_;
  #ifdef NODE_DEBUG
  size_type node_index_;
  #endif
  elem_type lane_elem_;
  const tile_type& tile_;
  allocator_type& allocator_;

  // each node consists of 32 elements and stores up to 29 key slices
  // [key0] [key1] ... [key28] [key_length] [value] [next]
  // key_length and value is only stored at the head node, and left empty at later nodes

  static_assert(sizeof(elem_type) == sizeof(uint32_t));
  static constexpr uint32_t head_node_value_lane_ = node_width - 2;
  static constexpr uint32_t head_node_length_lane_ = node_width - 3;
  static constexpr uint32_t next_lane_ = node_width - 1;
  static constexpr uint32_t node_max_len_ = node_width - 3;
};
