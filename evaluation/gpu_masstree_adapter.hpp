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

#include <vector>
#include <string>
#include <macros.hpp>
#include <cmd.hpp>
#include <adapter_util.hpp>
#include <gpu_masstree.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

struct gpu_masstree_adapter {
  static constexpr bool is_ordered = true;
  using key_slice_type = uint32_t;
  using value_type = uint32_t;
  using size_type = uint32_t;
  using allocator_type = simple_slab_allocator<128>;
  using reclaimer_type = simple_debra_reclaimer<>;
  using index32_type = GpuMasstree::gpu_masstree<allocator_type, reclaimer_type, 32>;
  using index16_type = GpuMasstree::gpu_masstree<allocator_type, reclaimer_type, 32>;

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
      index_ = reinterpret_cast<void*>(new index32_type(*allocator_, *reclaimer_));
    }
    else {
      index_ = reinterpret_cast<void*>(new index16_type(*allocator_, *reclaimer_));
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
      adapter_util::dispatch_bool(configs_.enable_suffix, [&](auto t2, auto s2) {
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto s3, auto r3) {
          do_insert<t3.value, s3.value, r3.value>(keys, keylen_max, key_lengths, values, num_keys);
        }, t2, s2);
      }, t1);
    });
  }
  void erase(const key_slice_type* keys,
             uint32_t keylen_max,
             const size_type* key_lengths,
             std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_uint32<0, 1, 2, 3>(configs_.merge_level, [&](auto t2, auto m2) {
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto m3, auto r3) {
          do_erase<t3.value, m3.value, r3.value>(keys, keylen_max, key_lengths, num_keys);
        }, t2, m2);
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
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto c3, auto r3) {
          do_find<t3.value, c3.value, r3.value>(keys, keylen_max, key_lengths, results, num_keys);
        }, t2, c2);
      }, t1);
    });
  }
  void scan(const key_slice_type* keys,
            uint32_t keylen_max,
            const size_type* key_lengths,
            uint32_t count,
            value_type* results,
            std::size_t num_keys) {
    adapter_util::dispatch_uint32<32, 16>(configs_.tile_size, [&](auto t1) {
      adapter_util::dispatch_bool(configs_.lookup_concurrent, [&](auto t2, auto c2) {
        adapter_util::dispatch_bool(configs_.reuse_root, [&](auto t3, auto c3, auto r3) {
          do_scan<t3.value, c3.value, r3.value>(keys, key_lengths, keylen_max, count, num_keys,
                                                nullptr, nullptr, nullptr, results, nullptr, nullptr);
        }, t2, c2);
      }, t1);
    });
  }

 private:
  struct configs {
    float allocator_pool_ratio;
    uint32_t tile_size;
    bool lookup_concurrent;
    bool enable_suffix;
    uint32_t merge_level;  // 0: naive, 1: concurrent, 2: merge, 3: remove_root
    bool reuse_root;

    configs() {}
    configs(std::vector<std::string>& arguments) {
      allocator_pool_ratio = get_arg_value<float>(arguments, "allocator-pool-ratio").value_or(0.9f);
      tile_size = get_arg_value<uint32_t>(arguments, "tile-size").value_or(32);
      lookup_concurrent = get_arg_value<bool>(arguments, "lookup-concurrent").value_or(true);
      enable_suffix = get_arg_value<bool>(arguments, "enable-suffix").value_or(true);
      merge_level = get_arg_value<uint32_t>(arguments, "merge-level").value_or(3);
      reuse_root = get_arg_value<bool>(arguments, "reuse-root").value_or(true);
      check_argument(tile_size == 32 || tile_size == 16);
      check_argument(merge_level <= 3);
    }
    void print() const {
      std::cout << "  allocator-pool-ratio: " << allocator_pool_ratio << std::endl
                << "  tile-size: " << tile_size << std::endl
                << "  lookup-concurrent: " << lookup_concurrent << std::endl
                << "  enable-suffix: " << enable_suffix << std::endl
                << "  merge-level: " << merge_level << std::endl
                << "  reuse-root: " << reuse_root << std::endl
                ;
    }
  };

  template <uint32_t tile_size, bool enable_suffix, bool reuse_root, typename... arg_types>
  void do_insert(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template insert<enable_suffix, reuse_root>(args...);
  }

  template <uint32_t tile_size, uint32_t merge_level, bool reuse_root, typename... arg_types>
  void do_erase(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template erase<merge_level >= 3, merge_level >= 2, merge_level >= 1, reuse_root>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, bool reuse_root, typename... arg_types>
  void do_find(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template find<lookup_concurrent, reuse_root>(args...);
  }

  template <uint32_t tile_size, bool lookup_concurrent, bool reuse_root, typename... arg_types>
  void do_scan(arg_types... args) {
    reinterpret_cast<std::conditional_t<tile_size == 32, index32_type, index16_type>*>(index_)
      ->template scan<false, lookup_concurrent, reuse_root>(args...);
  }

  configs configs_;
  allocator_type* allocator_;
  reclaimer_type* reclaimer_;
  void* index_;
};
