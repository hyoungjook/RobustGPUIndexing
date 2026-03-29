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
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <cmd.hpp>
#include <libcuckoo/cuckoohash_map.hh>

struct cpu_libcuckoo_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  struct key_type {
    const key_slice_type* data;
    size_type length;
  };
  struct key_hash {
    static uint64_t mix64(uint64_t x) {
      x ^= x >> 30;
      x *= 0xbf58476d1ce4e5b9ULL;
      x ^= x >> 27;
      x *= 0x94d049bb133111ebULL;
      x ^= x >> 31;
      return x;
    }
    std::size_t operator()(const key_type& key) const {
      uint64_t hash = 1469598103934665603ULL;
      for (size_t i = 0; i < key.length; i++) {
        hash ^= key.data[i];
        hash *= 1099511628211ULL;
      }
      return static_cast<std::size_t>(mix64(hash ^ (static_cast<uint64_t>(key.length) << 32)));
    }
  };
  struct key_equal {
    bool operator()(const key_type& lhs, const key_type& rhs) const {
      if (lhs.length != rhs.length) {
        return false;
      }
      if (lhs.length == 0) {
        return true;
      }
      return std::memcmp(lhs.data, rhs.data, sizeof(key_slice_type) * lhs.length) == 0;
    }
  };
  using index_type = libcuckoo::cuckoohash_map<key_type, value_type, key_hash, key_equal>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    index_ = std::make_unique<index_type>(configs_.initial_capacity);
  }
  void destroy() {
    index_.reset();
  }
  void thread_enter() noexcept {}
  void thread_exit() noexcept {}
  void insert(const key_slice_type* key, size_type key_length, value_type value) {
    index_->insert_or_assign(key_type{key, key_length}, value);
  }
  void erase(const key_slice_type* key, size_type key_length) {
    index_->erase(key_type{key, key_length});
  }
  value_type find(const key_slice_type* key, size_type key_length) {
    value_type value = std::numeric_limits<value_type>::max();
    index_->find(key_type{key, key_length}, value);
    return value;
  }

 private:
  struct configs {
    std::size_t initial_capacity;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      initial_capacity = get_arg_value<float>(arguments, "initial-capacity").value_or(100000);
      check_argument(0 < initial_capacity);
    }
    void print() const {
      std::cout << "  initial-capacity: " << initial_capacity << std::endl;
    }
  };

  configs configs_;
  std::unique_ptr<index_type> index_;
};
