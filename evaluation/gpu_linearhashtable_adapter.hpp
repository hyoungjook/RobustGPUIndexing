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
#include <gpu_linearhashtable.hpp>
#include <simple_slab_linear_alloc.hpp>
#include <simple_debra_reclaim.hpp>

struct gpu_linearhashtable_adapter {
  static constexpr bool is_ordered = false;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using allocator_type = simple_slab_linear_allocator<128>;
  using reclaimer_type = simple_debra_reclaimer<>;
  using index32_type = GpuLinearHashtable::gpu_linearhashtable<allocator_type, reclaimer_type, 32>;
  using index16_type = GpuLinearHashtable::gpu_linearhashtable<allocator_type, reclaimer_type, 16>;

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
      index_ = reinterpret_cast<void*>(new index32_type(*allocator_, *reclaimer_,
        configs_.initial_directory_size, configs_.resize_policy, configs_.load_factor_threshold));
    }
    else {
      index_ = reinterpret_cast<void*>(new index16_type(*allocator_, *reclaimer_,
        configs_.initial_directory_size, configs_.resize_policy, configs_.load_factor_threshold));
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
      adapter_util::dispatch_uint32<0, 1, 2>(configs_.hash_tag_level, [&](auto t2, auto h2) {
        adapter_util::dispatch_uint32<0, 1, 2>(configs_.merge_level, [&](auto t3, auto h3, auto m3) {
          adapter_util::dispatch_bool(configs_.reuse_dirsize, [&](auto t4, auto h4, auto m4, auto r4) {
            do_insert<t4.value, h4.value, m4.value, r4.value>(keys, keylen_max, key_lengths, values, num_keys);
          }, t3, h3, m3);
        }, t2, h2);
      }, t1);
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_uint32<0, 1, 2>(configs_.hash_tag_level, [&](auto t2, auto h2) {
        adapter_util::dispatch_uint32<0, 1, 2>(configs_.merge_level, [&](auto t3, auto h3, auto m3) {
          adapter_util::dispatch_bool(configs_.reuse_dirsize, [&](auto t4, auto h4, auto m4, auto r4) {
            do_erase<t4.value, h4.value, m4.value, r4.value>(keys, keylen_max, key_lengths, num_keys);
          }, t3, h3, m3);
        }, t2, h2);
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
        adapter_util::dispatch_uint32<0, 1, 2>(configs_.hash_tag_level, [&](auto t3, auto c3, auto h3) {
          adapter_util::dispatch_bool(configs_.reuse_dirsize, [&](auto t4, auto c4, auto h4, auto r4) {
            do_find<t4.value, c4.value, h4.value, r4.value>(keys, keylen_max, key_lengths, results, num_keys);
          }, t3, c3, h3);
        }, t2, c2);
      }, t1);
    });
  }

 private:
  struct configs {
    float allocator_pool_ratio;
    uint32_t tile_size;
    bool lookup_concurrent;
    uint32_t initial_directory_size;
    float resize_policy;
    float load_factor_threshold;
    inline static const char* hash_tag_level_strings[3] = {
      "slice0_tag", "hash_tag", "samehash_tag"
    };
    uint32_t hash_tag_level;  // 0: 1st slice as tag, 1: hash tag, 2: same hash tag
    inline static const char* merge_level_strings[3] = {
      "naive", "merge_chains", "merge_buckets"
    };
    uint32_t merge_level;   // 0: naive, 1: merge chains, 2: merge buckets
    bool reuse_dirsize;
    configs() {}
    configs(std::vector<std::string>& arguments) {
      allocator_pool_ratio = get_arg_value<float>(arguments, "allocator_pool_ratio").value_or(0.9f);
      tile_size = get_arg_value<uint32_t>(arguments, "tile_size").value_or(32);
      lookup_concurrent = get_arg_value<bool>(arguments, "lookup_concurrent").value_or(true);
      initial_directory_size = get_arg_value<uint32_t>(arguments, "initial_directory_size").value_or(1024);
      resize_policy = get_arg_value<float>(arguments, "resize_policy").value_or(2.0f);
      load_factor_threshold = get_arg_value<float>(arguments, "load_factor_threshold").value_or(2.5f);
      hash_tag_level = get_arg_value<uint32_t>(arguments, "hash_tag_level").value_or(2);
      merge_level = get_arg_value<uint32_t>(arguments, "merge_level").value_or(2);
      reuse_dirsize = get_arg_value<bool>(arguments, "reuse_dirsize").value_or(true);
      check_argument(tile_size == 32 || tile_size == 16);
      check_argument(0 < load_factor_threshold);
      check_argument(hash_tag_level <= 2);
    }
    void print() const {
      std::cout << "    allocator_pool_ratio=" << allocator_pool_ratio << std::endl
                << "    tile_size=" << tile_size << std::endl
                << "    lookup_concurrent=" << lookup_concurrent << std::endl
                << "    initial_directory_size=" << initial_directory_size << std::endl
                << "    resize_policy=" << resize_policy << std::endl
                << "    load_factor_threshold=" << load_factor_threshold << std::endl
                << "    hash_tag_level=" << hash_tag_level << "(" << hash_tag_level_strings[hash_tag_level] << ")" << std::endl
                << "    merge_level=" << merge_level << "(" << merge_level_strings[merge_level] << ")" << std::endl
                << "    reuse_dirsize=" << reuse_dirsize<< std::endl
                ;
    }
  };

  template <uint32_t tile_size, uint32_t hash_tag_level, uint32_t merge_level, bool reuse_dirsize, typename... arg_types>
  void do_insert(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template insert<hash_tag_level >= 1, hash_tag_level >= 2, merge_level >= 1, reuse_dirsize>(args...);
  }

  template <uint32_t tile_size, uint32_t hash_tag_level, uint32_t merge_level, bool reuse_dirsize, typename... arg_types>
  void do_erase(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template erase<hash_tag_level >= 1, hash_tag_level >= 2, merge_level >= 2, merge_level >= 1, reuse_dirsize>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, uint32_t hash_tag_level, bool reuse_dirsize, typename... arg_types>
  void do_find(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template find<lookup_concurrent, hash_tag_level >= 1, hash_tag_level >= 2, reuse_dirsize>(args...);
  }

  configs configs_;
  allocator_type* allocator_;
  reclaimer_type* reclaimer_;
  void* index_;
};
