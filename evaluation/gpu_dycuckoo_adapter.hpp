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
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <gpu_dycuckoo_backend.hpp>
#include <macros.hpp>

struct gpu_dycuckoo_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    if (configs_.use_lock) {
      index_ = gpu_dycuckoo_dynamic_lock_create(configs_.initial_capacity,
                                                configs_.small_batch_size,
                                                configs_.fill_factor_lower_bound,
                                                configs_.fill_factor_upper_bound);
    }
    else {
      index_ = gpu_dycuckoo_dynamic_create(configs_.initial_capacity,
                                           configs_.small_batch_size,
                                           configs_.fill_factor_lower_bound,
                                           configs_.fill_factor_upper_bound);
    }
  }
  void destroy() {
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_destroy(index_);
    }
    else {
      gpu_dycuckoo_dynamic_destroy(index_);
    }
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_type* values,
              std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_insert(index_, keys, values, num_keys);
    }
    else {
      gpu_dycuckoo_dynamic_insert(index_, keys, values, num_keys);
    }
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_erase(index_, keys, num_keys);
    }
    else {
      gpu_dycuckoo_dynamic_erase(index_, keys, num_keys);
    }
  }
  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_type* results,
            std::size_t num_keys) {
    (void)keylen_max;
    (void)key_lengths;
    if (configs_.use_lock) {
      gpu_dycuckoo_dynamic_lock_find(index_, keys, results, num_keys);
    }
    else {
      gpu_dycuckoo_dynamic_find(index_, keys, results, num_keys);
    }
  }

 private:
  struct configs {
    bool use_lock;
    uint32_t initial_capacity;
    float fill_factor_lower_bound;
    float fill_factor_upper_bound;
    int small_batch_size;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      use_lock = get_arg_value<bool>(arguments, "use_lock").value_or(false);
      initial_capacity = get_arg_value<uint32_t>(arguments, "initial_capacity").value_or(100000);
      fill_factor_lower_bound = get_arg_value<float>(arguments, "fill_factor_lower_bound").value_or(0.5f);
      fill_factor_upper_bound = get_arg_value<float>(arguments, "fill_factor_upper_bound").value_or(0.8f);
      small_batch_size = get_arg_value<int>(arguments, "small-batch-size").value_or(20000);
      check_argument(0 < initial_capacity);
      check_argument(0 < fill_factor_lower_bound &&
                     fill_factor_lower_bound < fill_factor_upper_bound &&
                     fill_factor_upper_bound <= 1.0f);
      uint32_t keylen_max = get_arg_value<uint32_t>(arguments, "keylen-max").value_or(1);
      check_argument(keylen_max == 1);
    }

    void print() const {
      std::cout << "    use_lock=" << use_lock << std::endl
                << "    initial_capacity=" << initial_capacity << std::endl
                << "    fill_factor_lower_bound=" << fill_factor_lower_bound << std::endl
                << "    fill_factor_upper_bound=" << fill_factor_upper_bound << std::endl
                << "    small_batch_size=" << small_batch_size << std::endl
                ;
    }
  };

  configs configs_;
  void* index_;
};
