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
#include <vector>
#include <string>
#include <macros.hpp>
#include <cmd.hpp>
#include <adapter_util.hpp>
#include <gpu_cuckoohashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

struct gpu_cuckoohashtable_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using allocator_type = simple_slab_allocator<128>;
  using reclaimer_type = simple_debra_reclaimer<>;
  using index32_type = GpuHashtable::gpu_cuckoohashtable<allocator_type, reclaimer_type, 32>;
  using index16_type = GpuHashtable::gpu_cuckoohashtable<allocator_type, reclaimer_type, 16>;

  void parse(std::vector<std::string>& arguments) {
    configs_ = configs(arguments);
  }
  void print_args() const {
    configs_.print();
  }
  void initialize() {
    allocator_ = new allocator_type(configs_.allocator_pool_ratio);
    reclaimer_ = new reclaimer_type();
    if (configs_.tile_size == 32) {
      index_ = reinterpret_cast<void*>(new index32_type(*allocator_, *reclaimer_, configs_.num_keys, configs_.initial_array_fill_factor));
    }
    else {
      index_ = reinterpret_cast<void*>(new index16_type(*allocator_, *reclaimer_, configs_.num_keys, configs_.initial_array_fill_factor));
    }
  }
  void destroy() {
    if (configs_.tile_size == 32) {
      delete reinterpret_cast<index32_type*>(index_);
    }
    else {
      delete reinterpret_cast<index16_type*>(index_);
    }
    delete allocator_;
    delete reclaimer_;
  }
  void insert(const key_slice_type* keys,
              uint32_t keylen_max,
              const size_type* key_lengths,
              const value_type* values,
              std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.use_hash_tag, [&](auto t2, auto h2) {
        do_insert<t2.value, h2.value>(keys, keylen_max, key_lengths, values, num_keys);
      }, t1);
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.use_hash_tag, [&](auto t2, auto h2) {
        do_erase<t2.value, h2.value>(keys, keylen_max, key_lengths, num_keys);
      }, t1);
    });
  }
  void find(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            value_type* results,
            std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.lookup_concurrent, [&](auto t2, auto c2) {
        adapter_util::dispatch_bool(configs_.use_hash_tag, [&](auto t3, auto c3, auto h3) {
          do_find<t3.value, c3.value, h3.value>(keys, keylen_max, key_lengths, results, num_keys);
        }, t2, c2);
      }, t1);
    });
  }

 private:
  struct configs {
    float allocator_pool_ratio;
    uint32_t tile_size;
    bool lookup_concurrent;
    std::size_t num_keys; // parse again here; do not print
    float initial_array_fill_factor;
    bool use_hash_tag;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      allocator_pool_ratio = get_arg_value<float>(arguments, "allocator_pool_ratio").value_or(0.1f);
      tile_size = get_arg_value<uint32_t>(arguments, "tile_size").value_or(32);
      lookup_concurrent = get_arg_value<bool>(arguments, "lookup_concurrent").value_or(true);
      num_keys = get_arg_value<std::size_t>(arguments, "num_keys").value_or(1000000);
      initial_array_fill_factor = get_arg_value<float>(arguments, "initial_array_fill_factor").value_or(0.8f);
      use_hash_tag = get_arg_value<bool>(arguments, "use_hash_tag").value_or(true);
      check_argument(tile_size == 32 || tile_size == 16);
      check_argument(0 < initial_array_fill_factor && initial_array_fill_factor <= 0.9f);
    }
    void print() const {
      std::cout << "    allocator_pool_ratio=" << allocator_pool_ratio << std::endl
                << "    tile_size=" << tile_size << std::endl
                << "    lookup_concurrent=" << lookup_concurrent << std::endl
                << "    initial_array_fill_factor=" << initial_array_fill_factor << std::endl
                << "    use_hash_tag=" << use_hash_tag << std::endl
                ;
    }
  };

  template <uint32_t tile_size, bool use_hash_tag, typename... arg_types>
  void do_insert(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template insert<use_hash_tag>(args...);
  }

  template <uint32_t tile_size, bool use_hash_tag, typename... arg_types>
  void do_erase(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template erase<use_hash_tag>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, bool use_hash_tag, typename... arg_types>
  void do_find(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template find<lookup_concurrent, use_hash_tag>(args...);
  }

  configs configs_;
  allocator_type* allocator_;
  reclaimer_type* reclaimer_;
  void* index_;
};
