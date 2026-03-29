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
    auto initial_capacity = static_cast<uint32_t>(
      static_cast<float>(configs_.num_keys) / configs_.initial_array_fill_factor);
    if (configs_.use_lock) {
      index_ = gpu_dycuckoo_dynamic_lock_create(initial_capacity,
                                                static_cast<int>(configs_.small_batch_size),
                                                configs_.lower_bound,
                                                configs_.upper_bound);
    }
    else {
      index_ = gpu_dycuckoo_dynamic_create(initial_capacity,
                                           static_cast<int>(configs_.small_batch_size),
                                           configs_.lower_bound,
                                           configs_.upper_bound);
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
    std::size_t num_keys;
    float initial_array_fill_factor;

    double lower_bound;
    double upper_bound;
    uint32_t small_batch_size;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      use_lock = get_arg_value<bool>(arguments, "use-lock").value_or(false);
      num_keys = get_arg_value<std::size_t>(arguments, "num-keys").value_or(1000000);
      initial_array_fill_factor = get_arg_value<float>(arguments, "initial-array-fill-factor").value_or(0.8f);
      uint32_t keylen_max = get_arg_value<uint32_t>(arguments, "keylen-max").value_or(1);
      check_argument(keylen_max == 1);
      check_argument(0 < initial_array_fill_factor && initial_array_fill_factor <= 1.0);

      lower_bound = get_arg_value<double>(arguments, "dycuckoo-lower-bound").value_or(0.5);
      upper_bound = get_arg_value<double>(arguments, "dycuckoo-upper-bound").value_or(0.85);
      small_batch_size = get_arg_value<uint32_t>(arguments, "dycuckoo-small-batch-size")
                           .value_or(static_cast<uint32_t>(std::min<std::size_t>(
                             num_keys, std::numeric_limits<uint32_t>::max()
                           )));
      check_argument(upper_bound <= 1.0);
      check_argument(small_batch_size > 0);
    }

    void print() const {
      std::cout << "    use-lock: " << use_lock << std::endl
                << "    initial-array-fill-factor: " << initial_array_fill_factor << std::endl
                ;
    }
  };

  configs configs_;
  void* index_;
};
