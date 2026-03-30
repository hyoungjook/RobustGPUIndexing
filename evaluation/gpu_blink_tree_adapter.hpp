/*
 *   Copyright 2026 Hyoungjoo Kim, Carnegie Mellon University
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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <device_context.hpp>
#include <macros.hpp>
#include <pair_type.hpp>

namespace GpuBTree {
template <typename Key, typename Value, int b = 16>
struct node_type {
  using T = pair_type<Key, Value>;
  T node[b];
};
}  // namespace GpuBTree
#include <gpu_blink_tree.hpp>

struct gpu_blink_tree_adapter {
  static constexpr bool is_ordered = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using index_type = GpuBTree::gpu_blink_tree<key_slice_type, value_type>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    index_ = new index_type();
  }
  void destroy() {
    delete index_;
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_type* values,
              std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    index_->insert(keys, values, num_keys);
  }

  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    index_->erase(keys, num_keys, 0, configs_.erase_concurrent);
  }

  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_type* results,
            std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    index_->find(keys, results, num_keys, 0, configs_.lookup_concurrent);
  }

  void scan(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            uint32_t count,
            value_type* results,
            std::size_t num_keys,
            const key_slice_type* upper_keys) {
    (void)keylen_max;
    (void)key_lengths;
    index_->range_query(keys, upper_keys,
      reinterpret_cast<pair_type<key_slice_type, value_type>*>(results), nullptr,
      count, num_keys, 0, configs_.lookup_concurrent);
  }

 private:
  struct configs {
    bool lookup_concurrent;
    bool erase_concurrent;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      lookup_concurrent = get_arg_value<bool>(arguments, "lookup_concurrent").value_or(true);
      erase_concurrent = get_arg_value<bool>(arguments, "erase_concurrent").value_or(true);
      uint32_t keylen_max = get_arg_value<uint32_t>(arguments, "keylen_max").value_or(1);
      check_argument(keylen_max == 1);
    }
    void print() const {
      std::cout << "    lookup_concurrent=" << lookup_concurrent << std::endl
                << "    erase_concurrent=" << erase_concurrent << std::endl
                ;
    }
  };

  configs configs_;
  index_type* index_;
};
