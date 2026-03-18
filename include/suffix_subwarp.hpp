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
#include <limits>

#include <macros.hpp>
#include <utils.hpp>

template <typename tile_type, typename allocator_type>
struct suffix_node_subwarp {
  using elem_type = uint64_t;
  using slice_type = uint32_t;
  using size_type = uint32_t;
  static constexpr int node_width = 16;

  DEVICE_QUALIFIER suffix_node_subwarp(const tile_type& tile, allocator_type& allocator)
      : tile_(tile)
      , allocator_(allocator) {
    assert(tile_.size() == node_width);
  }
  DEVICE_QUALIFIER suffix_node_subwarp(size_type index, const tile_type& tile, allocator_type& allocator)
      : node_index_(index)
      , tile_(tile)
      , allocator_(allocator) {
    assert(tile_.size() == node_width);
  }

  DEVICE_QUALIFIER void load_head() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    lane_elem_ = utils::memory::load<elem_type, false>(node_ptr + tile_.thread_rank());
  }
  DEVICE_QUALIFIER void store_head() {
    auto node_ptr = reinterpret_cast<elem_type*>(allocator_.address(node_index_));
    utils::memory::store<elem_type, false>(node_ptr + tile_.thread_rank(), lane_elem_);
  }

  DEVICE_QUALIFIER size_type get_node_index() const { return node_index_; }
  DEVICE_QUALIFIER size_type get_next() const { return get_word(next_word_index_); }
  DEVICE_QUALIFIER size_type get_key_length() const { return get_word(head_node_length_index_); }
  DEVICE_QUALIFIER void set_length(const size_type& key_length) { set_word(head_node_length_index_, key_length); }
  DEVICE_QUALIFIER slice_type get_value() const { return get_word(head_node_value_index_); }
  DEVICE_QUALIFIER void update_value(const slice_type& value) { set_word(head_node_value_index_, value); }

  DEVICE_QUALIFIER uint32_t get_num_nodes() const {
    auto length = get_key_length();
    return ((length + 2) + node_max_words_ - 1) / node_max_words_;
  }

  DEVICE_QUALIFIER bool streq(const slice_type* key, uint32_t key_length) const {
    if (get_key_length() != key_length) { return false; }
    if (key_length == 0) { return true; }

    size_type current_index = node_index_;
    elem_type current_lane_elem = lane_elem_;
    bool is_head = true;
    uint32_t key_offset = 0;
    uint32_t remaining = key_length;
    while (true) {
      const uint32_t start_word = payload_start_word(is_head);
      const uint32_t payload_count = min(remaining, payload_capacity(is_head));
      const uint32_t local_word0 = 2 * tile_.thread_rank();
      const uint32_t local_word1 = local_word0 + 1;
      bool mismatch = false;
      if (start_word <= local_word0 && local_word0 < start_word + payload_count) {
        mismatch |= (word_from_lane(current_lane_elem, 0) != key[key_offset + (local_word0 - start_word)]);
      }
      if (start_word <= local_word1 && local_word1 < start_word + payload_count) {
        mismatch |= (word_from_lane(current_lane_elem, 1) != key[key_offset + (local_word1 - start_word)]);
      }
      if (tile_.any(mismatch)) { return false; }
      if (remaining <= payload_count) { return true; }

      remaining -= payload_count;
      key_offset += payload_count;
      is_head = false;
      current_index = word_from_lane(tile_.shfl(current_lane_elem, next_lane_), 1);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(current_index));
      current_lane_elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
    }
  }

  DEVICE_QUALIFIER int strcmp(const slice_type* key, uint32_t key_length, slice_type* mismatch_value = nullptr) const {
    // strcmp(this, key) -> 0 (match), +(this<key), -(this>key)
    const uint32_t this_length = get_key_length();
    const uint32_t cmp_length = min(this_length, key_length);

    size_type current_index = node_index_;
    elem_type current_lane_elem = lane_elem_;
    bool is_head = true;
    uint32_t key_offset = 0;
    uint32_t total_matches = 0;
    uint32_t remaining_cmp = cmp_length;
    while (remaining_cmp > 0) {
      const uint32_t start_word = payload_start_word(is_head);
      const uint32_t payload_count = min(remaining_cmp, payload_capacity(is_head));
      uint32_t local_mismatch = std::numeric_limits<uint32_t>::max();
      slice_type local_value = 0;

      const uint32_t local_word0 = 2 * tile_.thread_rank();
      const uint32_t local_word1 = local_word0 + 1;
      if (start_word <= local_word0 && local_word0 < start_word + payload_count) {
        const auto local_offset = key_offset + (local_word0 - start_word);
        const auto value = word_from_lane(current_lane_elem, 0);
        if (value != key[local_offset]) {
          local_mismatch = min(local_mismatch, local_offset);
          local_value = value;
        }
      }
      if (start_word <= local_word1 && local_word1 < start_word + payload_count) {
        const auto local_offset = key_offset + (local_word1 - start_word);
        const auto value = word_from_lane(current_lane_elem, 1);
        if (value != key[local_offset] && local_offset < local_mismatch) {
          local_mismatch = local_offset;
          local_value = value;
        }
      }

      uint32_t first_mismatch = reduce_min(local_mismatch);
      if (first_mismatch != std::numeric_limits<uint32_t>::max()) {
        const uint32_t mismatch_lane = (payload_start_word(is_head) + (first_mismatch - key_offset)) / 2;
        const auto this_value = tile_.shfl(local_value, mismatch_lane);
        if (mismatch_value) { *mismatch_value = this_value; }
        return ((this_value < key[first_mismatch]) ? 1 : -1) * static_cast<int>(1 + first_mismatch);
      }

      total_matches += payload_count;
      remaining_cmp -= payload_count;
      key_offset += payload_count;
      if (remaining_cmp == 0) { break; }

      is_head = false;
      current_index = word_from_lane(tile_.shfl(current_lane_elem, next_lane_), 1);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(current_index));
      current_lane_elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
    }

    if (this_length == key_length) { return 0; }
    if (mismatch_value && cmp_length > 0) {
      *mismatch_value = read_word_at(cmp_length - 1);
    }
    return ((this_length < key_length) ? 1 : -1) * static_cast<int>(cmp_length);
  }

  DEVICE_QUALIFIER void create_from(const slice_type* key, size_type key_length, slice_type value) {
    write_from_slices(key, key_length, value);
  }

  DEVICE_QUALIFIER void flush(slice_type* key_buffer) {
    auto current_index = node_index_;
    auto current_lane_elem = lane_elem_;
    bool is_head = true;
    uint32_t key_offset = 0;
    uint32_t remaining = get_key_length();
    while (remaining > 0) {
      const uint32_t start_word = payload_start_word(is_head);
      const uint32_t payload_count = min(remaining, payload_capacity(is_head));
      const uint32_t local_word0 = 2 * tile_.thread_rank();
      const uint32_t local_word1 = local_word0 + 1;
      if (start_word <= local_word0 && local_word0 < start_word + payload_count) {
        key_buffer[key_offset + (local_word0 - start_word)] = word_from_lane(current_lane_elem, 0);
      }
      if (start_word <= local_word1 && local_word1 < start_word + payload_count) {
        key_buffer[key_offset + (local_word1 - start_word)] = word_from_lane(current_lane_elem, 1);
      }
      remaining -= payload_count;
      key_offset += payload_count;
      if (remaining == 0) { break; }

      is_head = false;
      current_index = word_from_lane(tile_.shfl(current_lane_elem, next_lane_), 1);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(current_index));
      current_lane_elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void move_from(suffix_node_subwarp& src,
                                  uint32_t offset,
                                  reclaimer_type& reclaimer) {
    auto new_length = src.get_key_length() - offset;
    auto value = src.get_value();
    if (new_length == 0) {
      write_from_slices(nullptr, 0, value);
      src.retire(reclaimer);
      return;
    }

    size_type src_index = src.node_index_;
    elem_type src_lane_elem = src.lane_elem_;
    bool src_is_head = true;
    uint32_t src_offset = offset;
    while (src_offset >= payload_capacity(src_is_head)) {
      src_offset -= payload_capacity(src_is_head);
      auto next_index = word_from_lane(tile_.shfl(src_lane_elem, next_lane_), 1);
      reclaimer.retire(src_index, tile_);
      src_index = next_index;
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(src_index));
      src_lane_elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      src_is_head = false;
    }

    size_type dst_index = node_index_;
    elem_type* dst_ptr = nullptr;
    bool dst_is_head = true;
    uint32_t remaining = new_length;
    while (true) {
      const uint32_t dst_start_word = payload_start_word(dst_is_head);
      const uint32_t dst_payload = min(remaining, payload_capacity(dst_is_head));

      const uint32_t src_cur_capacity = payload_capacity(src_is_head);
      const uint32_t src_available = src_cur_capacity - src_offset;
      size_type next_src_index = 0;
      elem_type next_src_lane_elem = 0;
      bool have_next_src = dst_payload > src_available;
      if (have_next_src) {
        next_src_index = word_from_lane(tile_.shfl(src_lane_elem, next_lane_), 1);
        auto* next_src_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_src_index));
        next_src_lane_elem = utils::memory::load<elem_type, false>(next_src_ptr + tile_.thread_rank());
      }

      elem_type dst_lane_elem = 0;
      if (dst_is_head) {
        dst_lane_elem = pack_lane(new_length, value);
      }
      const uint32_t local_word0 = 2 * tile_.thread_rank();
      const uint32_t local_word1 = local_word0 + 1;
      if (dst_start_word <= local_word0 && local_word0 < dst_start_word + dst_payload) {
        const uint32_t rel = local_word0 - dst_start_word;
        set_word_in_lane(
          dst_lane_elem,
          0,
          (rel < src_available) ?
            word_at_payload_index(src_lane_elem, src_is_head, src_offset + rel) :
            word_at_payload_index(next_src_lane_elem, false, rel - src_available));
      }
      if (dst_start_word <= local_word1 && local_word1 < dst_start_word + dst_payload) {
        const uint32_t rel = local_word1 - dst_start_word;
        set_word_in_lane(
          dst_lane_elem,
          1,
          (rel < src_available) ?
            word_at_payload_index(src_lane_elem, src_is_head, src_offset + rel) :
            word_at_payload_index(next_src_lane_elem, false, rel - src_available));
      }

      remaining -= dst_payload;
      if (remaining > 0) {
        auto next_dst_index = allocator_.allocate(tile_);
        set_word_in_lane(dst_lane_elem, word_in_lane(next_word_index_), next_dst_index);
        if (dst_ptr) {
          utils::memory::store<elem_type, false>(dst_ptr + tile_.thread_rank(), dst_lane_elem);
        }
        else {
          lane_elem_ = dst_lane_elem;
        }
        dst_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_dst_index));
        dst_index = next_dst_index;
      }
      else {
        if (dst_ptr) {
          utils::memory::store<elem_type, false>(dst_ptr + tile_.thread_rank(), dst_lane_elem);
        }
        else {
          lane_elem_ = dst_lane_elem;
        }
      }

      uint32_t consumed = dst_payload;
      while (consumed >= src_available) {
        consumed -= src_available;
        auto retired_index = src_index;
        auto following_index = have_next_src ? next_src_index : word_from_lane(tile_.shfl(src_lane_elem, next_lane_), 1);
        reclaimer.retire(retired_index, tile_);
        if (!have_next_src) {
          src_index = following_index;
          auto* next_src_ptr = reinterpret_cast<elem_type*>(allocator_.address(src_index));
          src_lane_elem = utils::memory::load<elem_type, false>(next_src_ptr + tile_.thread_rank());
        }
        else {
          src_index = next_src_index;
          src_lane_elem = next_src_lane_elem;
          have_next_src = false;
        }
        src_is_head = false;
        src_offset = 0;
        if (remaining == 0 && consumed == 0) { break; }
      }
      src_offset += consumed;
      if (remaining == 0) { break; }
      dst_is_head = false;
    }

    // retire any source nodes not consumed during the copy.
    auto trailing_index = src_index;
    while (true) {
      auto* suffix_ptr = reinterpret_cast<elem_type*>(allocator_.address(trailing_index));
      auto current_lane = utils::memory::load<elem_type, false>(suffix_ptr + tile_.thread_rank());
      auto next_index = word_from_lane(tile_.shfl(current_lane, next_lane_), 1);
      reclaimer.retire(trailing_index, tile_);
      if (payload_capacity(false) > src_offset || next_index == 0) { break; }
      trailing_index = next_index;
      src_offset = 0;
    }
  }

  template <typename reclaimer_type>
  DEVICE_QUALIFIER void retire(reclaimer_type& reclaimer) {
    auto num_nodes = get_num_nodes();
    auto suffix_index = node_index_;
    for (uint32_t i = 0; i < num_nodes; i++) {
      auto* suffix_ptr = reinterpret_cast<elem_type*>(allocator_.address(suffix_index));
      auto lane_value = utils::memory::load<elem_type, false>(suffix_ptr + tile_.thread_rank());
      auto next_index = word_from_lane(tile_.shfl(lane_value, next_lane_), 1);
      reclaimer.retire(suffix_index, tile_);
      suffix_index = next_index;
    }
  }

  static DEVICE_QUALIFIER slice_type fetch_value_only(size_type suffix_index, allocator_type& allocator) {
    auto* ptr = reinterpret_cast<elem_type*>(allocator.address(suffix_index));
    auto lane_value = utils::memory::load<elem_type, false>(ptr + head_node_value_lane_);
    return word_from_lane(lane_value, 1);
  }

  static DEVICE_QUALIFIER slice_type fetch_length_only(size_type suffix_index, allocator_type& allocator) {
    auto* ptr = reinterpret_cast<elem_type*>(allocator.address(suffix_index));
    auto lane_value = utils::memory::load<elem_type, false>(ptr + head_node_length_lane_);
    return word_from_lane(lane_value, 0);
  }

  DEVICE_QUALIFIER suffix_node_subwarp& operator=(const suffix_node_subwarp& other) {
    node_index_ = other.node_index_;
    lane_elem_ = other.lane_elem_;
    return *this;
  }

  DEVICE_QUALIFIER void print() const {
    if (tile_.thread_rank() == 0) {
      printf("node[%u]: s{l=%u v=%u}\n", node_index_, get_key_length(), get_value());
    }
  }

 private:
  static constexpr uint32_t node_words_ = 2 * node_width;
  static constexpr uint32_t head_node_length_index_ = 0;
  static constexpr uint32_t head_node_value_index_ = 1;
  static constexpr uint32_t next_word_index_ = node_words_ - 1;
  static constexpr uint32_t head_node_length_lane_ = 0;
  static constexpr uint32_t head_node_value_lane_ = 0;
  static constexpr uint32_t next_lane_ = node_width - 1;
  static constexpr uint32_t node_max_words_ = node_words_ - 1;

  size_type node_index_;
  elem_type lane_elem_ = 0;
  const tile_type& tile_;
  allocator_type& allocator_;

  static DEVICE_QUALIFIER uint32_t word_in_lane(uint32_t logical_word) { return logical_word & 1u; }
  static DEVICE_QUALIFIER elem_type pack_lane(slice_type lo, slice_type hi) {
    return static_cast<elem_type>(lo) | (static_cast<elem_type>(hi) << 32);
  }
  static DEVICE_QUALIFIER slice_type word_from_lane(elem_type lane_elem, int which) {
    return static_cast<slice_type>(which == 0 ? lane_elem : (lane_elem >> 32));
  }
  static DEVICE_QUALIFIER void set_word_in_lane(elem_type& lane_elem, int which, slice_type value) {
    if (which == 0) {
      lane_elem = (lane_elem & 0xffffffff00000000ULL) | static_cast<elem_type>(value);
    }
    else {
      lane_elem = (lane_elem & 0x00000000ffffffffULL) | (static_cast<elem_type>(value) << 32);
    }
  }
  static DEVICE_QUALIFIER uint32_t payload_start_word(bool is_head) { return is_head ? 2u : 0u; }
  static DEVICE_QUALIFIER uint32_t payload_capacity(bool is_head) { return node_max_words_ - payload_start_word(is_head); }

  DEVICE_QUALIFIER slice_type get_word(uint32_t logical_word) const {
    auto lane_value = tile_.shfl(lane_elem_, logical_word / 2);
    return word_from_lane(lane_value, word_in_lane(logical_word));
  }
  DEVICE_QUALIFIER void set_word(uint32_t logical_word, slice_type value) {
    if (tile_.thread_rank() == logical_word / 2) {
      set_word_in_lane(lane_elem_, word_in_lane(logical_word), value);
    }
  }

  DEVICE_QUALIFIER slice_type read_word_at(uint32_t key_index) const {
    size_type current_index = node_index_;
    elem_type current_lane_elem = lane_elem_;
    bool is_head = true;
    uint32_t remaining_offset = key_index;
    while (remaining_offset >= payload_capacity(is_head)) {
      remaining_offset -= payload_capacity(is_head);
      current_index = word_from_lane(tile_.shfl(current_lane_elem, next_lane_), 1);
      auto* next_ptr = reinterpret_cast<elem_type*>(allocator_.address(current_index));
      current_lane_elem = utils::memory::load<elem_type, false>(next_ptr + tile_.thread_rank());
      is_head = false;
    }
    return word_at_payload_index(current_lane_elem, is_head, remaining_offset);
  }

  static DEVICE_QUALIFIER slice_type word_at_payload_index(elem_type lane_elem, bool is_head, uint32_t payload_index) {
    const uint32_t logical_word = payload_start_word(is_head) + payload_index;
    return word_from_lane(lane_elem, word_in_lane(logical_word));
  }

  DEVICE_QUALIFIER uint32_t reduce_min(uint32_t value) const {
    for (uint32_t offset = node_width / 2; offset > 0; offset >>= 1) {
      value = min(value, tile_.shfl_down(value, offset));
    }
    return tile_.shfl(value, 0);
  }

  DEVICE_QUALIFIER void write_from_slices(const slice_type* key, size_type key_length, slice_type value) {
    size_type remaining = key_length;
    uint32_t key_offset = 0;
    elem_type* current_ptr = nullptr;
    bool is_head = true;
    while (true) {
      elem_type current_lane_elem = 0;
      if (is_head) {
        current_lane_elem = pack_lane(key_length, value);
      }
      const uint32_t start_word = payload_start_word(is_head);
      const uint32_t payload_count = min(static_cast<uint32_t>(remaining), payload_capacity(is_head));
      const uint32_t local_word0 = 2 * tile_.thread_rank();
      const uint32_t local_word1 = local_word0 + 1;
      if (start_word <= local_word0 && local_word0 < start_word + payload_count) {
        set_word_in_lane(current_lane_elem, 0, key[key_offset + (local_word0 - start_word)]);
      }
      if (start_word <= local_word1 && local_word1 < start_word + payload_count) {
        set_word_in_lane(current_lane_elem, 1, key[key_offset + (local_word1 - start_word)]);
      }

      remaining -= payload_count;
      key_offset += payload_count;
      if (remaining > 0) {
        auto next_index = allocator_.allocate(tile_);
        if (tile_.thread_rank() == next_lane_) {
          set_word_in_lane(current_lane_elem, 1, next_index);
        }
        if (current_ptr) {
          utils::memory::store<elem_type, false>(current_ptr + tile_.thread_rank(), current_lane_elem);
        }
        else {
          lane_elem_ = current_lane_elem;
        }
        current_ptr = reinterpret_cast<elem_type*>(allocator_.address(next_index));
        is_head = false;
      }
      else {
        if (current_ptr) {
          utils::memory::store<elem_type, false>(current_ptr + tile_.thread_rank(), current_lane_elem);
        }
        else {
          lane_elem_ = current_lane_elem;
        }
        break;
      }
    }
  }
};
