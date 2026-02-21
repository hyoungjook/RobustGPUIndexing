/*
 *   Copyright 2022 The Regents of the University of California, Davis
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
#include <thrust/device_vector.h>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <macros.hpp>
#include <memory>
#include <type_traits>
#include <host_allocators.hpp>
#include <cooperative_groups/reduce.h>

template <uint32_t slab_size = 128, std::size_t max_bytes = 6UL * 1024 * 1024 * 1024>
struct simple_slab_linear_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;
  using bitmap_type = uint32_t;
  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr uint32_t tile_size_ = 32;
  static constexpr uint32_t num_slabs_in_block_ = tile_size_ * sizeof(bitmap_type) * 8;
  static constexpr uint32_t block_size_ = slab_size_ * num_slabs_in_block_;
  static constexpr uint32_t blocks_delta_ = 8 * 1024;
  static constexpr std::size_t total_blocks_ = max_bytes / block_size_;
  static_assert(blocks_delta_ <= total_blocks_);
  static constexpr std::size_t initial_blocks_ = (total_blocks_ / 4 + blocks_delta_ - 1) / blocks_delta_ * blocks_delta_;
  static constexpr std::size_t total_bytes_ = total_blocks_ * block_size_;
  static constexpr std::size_t total_bitmap_bytes_ = (total_blocks_ * num_slabs_in_block_) / 8;
  static_assert(total_bitmap_bytes_ % (sizeof(bitmap_type) * tile_size_) == 0);
  static constexpr float load_factor_threshold_ = 0.8f;
  static constexpr uint32_t check_load_factor_every_ = 128;
  struct global_counters {
    size_type slab_block_count_;  // grows upward from the bottom
    size_type linear_count_;      // grows downward from the top
    uint8_t _align_buf0[128 - 2 * sizeof(size_type)];
    size_type num_slabs_;
    uint8_t _align_buf1[128 - sizeof(size_type)];
    global_counters(size_type initial_slab_blocks)
        : slab_block_count_(initial_slab_blocks)
        , linear_count_(0)
        , num_slabs_(0) {}
  };
  struct device_instance_type {
    void* pool_;
    global_counters* counts_;
  };

  simple_slab_linear_allocator() {
    cuda_try(cudaMalloc(&pool_, total_bitmap_bytes_ + total_bytes_));
    cuda_try(cudaMemset(pool_, 0x00, total_bitmap_bytes_));
    pool_ = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(pool_) + total_bitmap_bytes_);
    cuda_try(cudaMalloc(&counts_, sizeof(global_counters)));
    global_counters h_counts(initial_blocks_);
    cuda_try(cudaMemcpy(counts_, &h_counts, sizeof(global_counters), cudaMemcpyHostToDevice));
  }
  ~simple_slab_linear_allocator() {
    pool_ = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(pool_) - total_bitmap_bytes_);
    cuda_try(cudaFree(pool_));
    cuda_try(cudaFree(counts_));
  }
  simple_slab_linear_allocator(const simple_slab_linear_allocator& other) = delete;
  simple_slab_linear_allocator& operator=(const simple_slab_linear_allocator& other) = delete;

  device_instance_type get_device_instance() const { return device_instance_type{pool_, counts_}; }

  void print_stats() const {
    bitmap_type *h_bitmap = new bitmap_type[total_bitmap_bytes_ / sizeof(bitmap_type)];
    cuda_try(cudaMemcpy(h_bitmap,
                        reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(pool_) - total_bitmap_bytes_),
                        total_bitmap_bytes_,
                        cudaMemcpyDeviceToHost));
    std::size_t slab_count = 0;
    for (std::size_t i = 0; i < total_bitmap_bytes_ / sizeof(bitmap_type); i++) {
      slab_count += __builtin_popcount(h_bitmap[i]);
    }
    delete[] h_bitmap;
    global_counters h_counts(0);
    cuda_try(cudaMemcpy(&h_counts, counts_, sizeof(global_counters), cudaMemcpyDeviceToHost));
    const uint64_t total_slabs = total_blocks_ * num_slabs_in_block_;
    std::cout << "simple_slab_linear_allocator(" << slab_size_ << "B slabs, " << h_counts.slab_block_count_ << "/" << total_blocks_ << " blocks): "
        << slab_count << "/" << total_slabs << " slabs + " << h_counts.linear_count_ << " linear allocated "
        << "(" << (float)(((std::size_t)slab_count) * slab_size_ +
                          ((std::size_t)h_counts.linear_count_) * sizeof(size_type)) /
                         total_bytes_ * 100.0f << "%)" << std::endl;
  }

private:
  void* pool_;
  global_counters* counts_;
};

template <uint32_t slab_size, std::size_t max_bytes>
struct device_allocator_context<simple_slab_linear_allocator<slab_size, max_bytes>> {
  using host_alloc_type = simple_slab_linear_allocator<slab_size, max_bytes>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using size_type = typename host_alloc_type::size_type;
  using pointer_type = typename host_alloc_type::pointer_type;
  using bitmap_type = typename host_alloc_type::bitmap_type;
  using sizex2_type = uint64_t;
  static_assert(sizeof(sizex2_type) == 2 * sizeof(size_type));

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc, const tile_type& tile)
      : alloc_(alloc) {
    assert(tile.size() == tile_size_);
    block_index_ = initialize_index(get_tile_id());
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    pointer_type slab_index = try_allocate_in_block(block_index_, tile);
    if (slab_index == invalid_pointer) {
      uint32_t trials = 0;
      while (slab_index == invalid_pointer) {
        trials++;
        block_index_ = next_index(block_index_, trials);
        slab_index = try_allocate_in_block(block_index_, tile);
      }
    }
    // manage load factor
    //  NOTE since try_allocate_in_block() fills slab_index from 0 to 1023 (small to large),
    //  better to use the last modulo value here (represents all the previous bits are filled)
    //  yet becomes just an approximate if deallocate() pops in.
    if (slab_index % check_load_factor_every_ == (check_load_factor_every_ - 1)) {
      size_type num_slabs;
      if (tile.thread_rank() == 0) {
        cuda::atomic_ref<size_type, cuda::thread_scope_device> num_slabs_ref(alloc_.counts_->num_slabs_);
        num_slabs = num_slabs_ref.fetch_add(1, cuda::memory_order_relaxed);
      }
      num_slabs = tile.shfl(num_slabs, 0);
      if ((static_cast<float>(num_slabs) / alloc_.counts_->slab_block_count_ * (static_cast<float>(check_load_factor_every_) / num_slabs_in_block_)) > load_factor_threshold_) {
        // extend blocks
        if (tile.thread_rank() == 0) {
          cuda::atomic_ref<sizex2_type, cuda::thread_scope_device> two_counts_ref(
              *reinterpret_cast<sizex2_type*>(&alloc_.counts_->slab_block_count_));
          auto counters = two_counts_ref.load(cuda::memory_order_relaxed);
          // re-compute load factor
          if ((static_cast<float>(num_slabs) / static_cast<size_type>(counters) * (static_cast<float>(check_load_factor_every_) / num_slabs_in_block_)) > load_factor_threshold_) {
            // check if adding delta is okay
            auto new_num_block = min(
                static_cast<size_type>(counters) + blocks_delta_,
                total_blocks_ - ((static_cast<size_type>(counters >> 32) + linear_elems_per_block_ - 1) / linear_elems_per_block_));
            if (new_num_block > static_cast<size_type>(counters)) {
              auto new_counters = (counters & ~((static_cast<sizex2_type>(1) << 32) - 1)) | new_num_block;
              two_counts_ref.compare_exchange_strong(counters, new_counters,
                                                     cuda::memory_order_acquire,
                                                     cuda::memory_order_relaxed);
              // if failed, other warp's allocate() will do the extension, so no repeat
            }
          }
        }
      }
    }
    return (block_index_ * num_slabs_in_block_) + slab_index;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_coop(pointer_type p, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      deallocate_in_block(p);
    }
  }
  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) {
    deallocate_in_block(p);
    return (p % check_load_factor_every_ == (check_load_factor_every_ - 1)) ? 1 : 0;
  }
  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish_sync(uint32_t sum, const tile_type& tile) {
    sum = cooperative_groups::reduce(tile, sum, cooperative_groups::plus<uint32_t>());
    if (tile.thread_rank() == 0) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device> num_slabs_ref(alloc_.counts_->num_slabs_);
      num_slabs_ref.fetch_sub(sum, cuda::memory_order_relaxed);
    }
  }

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    return reinterpret_cast<void*>(reinterpret_cast<slab_type*>(alloc_.pool_) + p);
  }

  DEVICE_QUALIFIER void* get_linear() const {
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(alloc_.pool_) + max_bytes);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER size_type reallocate_linear(size_type size, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      cuda::atomic_ref<sizex2_type, cuda::thread_scope_device> two_counts_ref(
          *reinterpret_cast<sizex2_type*>(&alloc_.counts_->slab_block_count_));
      auto counters = two_counts_ref.load(cuda::memory_order_relaxed);
      while (true) {
        if (static_cast<size_type>(counters >> 32) >= size) {
          break;  // already enough
        }
        auto new_size = min(size,
            (total_blocks_ - static_cast<size_type>(counters)) * linear_elems_per_block_);
        auto new_counters = (counters & ((static_cast<sizex2_type>(1) << 32) - 1)) |
                            (static_cast<sizex2_type>(new_size) << 32);
        // try update
        if (two_counts_ref.compare_exchange_strong(counters, new_counters,
                                                   cuda::memory_order_acquire,
                                                   cuda::memory_order_relaxed)) {
          size = new_size;
          break;
        }
      }
    }
    return tile.shfl(size, 0);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint32_t tile_size_ = host_alloc_type::tile_size_;
  static_assert(host_alloc_type::total_blocks_ * host_alloc_type::num_slabs_in_block_ < std::numeric_limits<pointer_type>::max());
  static constexpr uint32_t num_slabs_in_block_ = host_alloc_type::num_slabs_in_block_;
  static constexpr uint32_t total_blocks_ = static_cast<uint32_t>(host_alloc_type::total_blocks_);
  static constexpr std::size_t total_bytes_ = host_alloc_type::total_bytes_;
  static constexpr std::size_t total_bitmap_bytes_ = host_alloc_type::total_bitmap_bytes_;
  static constexpr pointer_type invalid_pointer = std::numeric_limits<pointer_type>::max();
  static constexpr uint32_t blocks_delta_ = host_alloc_type::blocks_delta_;
  static constexpr uint32_t linear_elems_per_block_ = num_slabs_in_block_ * slab_size_ / sizeof(size_type);
  static constexpr float load_factor_threshold_ = host_alloc_type::load_factor_threshold_;
  static constexpr uint32_t check_load_factor_every_ = host_alloc_type::check_load_factor_every_;
  struct slab_type { uint8_t _[slab_size_]; };
  struct block_bitmap_type { uint8_t _[num_slabs_in_block_ / 8]; };

  const device_instance_type& alloc_;
  pointer_type block_index_;

  static DEVICE_QUALIFIER bool check_counters(size_type slab_block_count, size_type linear_count) {
    return (slab_block_count + ((linear_count + linear_elems_per_block_ - 1) / linear_elems_per_block_) <= total_blocks_);
  }

  static constexpr uint32_t mix_prime = 0x9e3779b1;
  DEVICE_QUALIFIER uint32_t get_tile_id() const {
    return (threadIdx.x + blockIdx.x * blockDim.x) / tile_size_;
  }
  DEVICE_QUALIFIER pointer_type initialize_index(uint32_t tile_id) const {
    size_type num_blocks = alloc_.counts_->slab_block_count_;
    return static_cast<pointer_type>(tile_id * mix_prime) % num_blocks;
  }
  DEVICE_QUALIFIER pointer_type next_index(pointer_type index, uint32_t trials) const {
    size_type num_blocks = alloc_.counts_->slab_block_count_;
    // linear probing
    //return (index + trials) % num_blocks;
    // quadratic probing
    return (index + trials * trials) % num_blocks;
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type try_allocate_in_block(pointer_type block_index, const tile_type& tile) {
    auto* bitmap_base = reinterpret_cast<uint8_t*>(alloc_.pool_) - total_bitmap_bytes_;
    bitmap_type* bitmap_addr =
        reinterpret_cast<bitmap_type*>(
          reinterpret_cast<block_bitmap_type*>(bitmap_base) + block_index) + tile.thread_rank();
    bitmap_type bitmap = *bitmap_addr;
    pointer_type result = invalid_pointer;
    while (result == invalid_pointer) {
      auto empty_lane = __ffs(~bitmap) - 1;
      auto free_lane = tile.ballot(empty_lane >= 0);
      if (free_lane != 0) {
        uint32_t src_lane = __ffs(free_lane) - 1;
        if (src_lane == tile.thread_rank()) {
          bitmap_type mask = static_cast<bitmap_type>(1) << empty_lane;
          cuda::atomic_ref<bitmap_type, cuda::thread_scope_device> bitmap_ref(*bitmap_addr);
          bitmap = bitmap_ref.fetch_or(mask, cuda::memory_order_relaxed);
          if ((bitmap & mask) == 0) {
            // atomically acquired the bit
            result = empty_lane + src_lane * sizeof(bitmap_type) * 8;
            bitmap |= mask;
          }
        }
        result = tile.shfl(result, src_lane);
      }
      else {
        // block full
        break;
      }
    }
    return result;
  }
  DEVICE_QUALIFIER void deallocate_in_block(pointer_type ptr) {
    auto* bitmap_base = reinterpret_cast<uint8_t*>(alloc_.pool_) - total_bitmap_bytes_;
    auto bitmap_index = ptr / (sizeof(bitmap_type) * 8);
    auto bit_index = ptr % (sizeof(bitmap_type) * 8);
    bitmap_type* bitmap_addr =
        reinterpret_cast<bitmap_type*>(bitmap_base) + bitmap_index;
    bitmap_type mask = static_cast<bitmap_type>(1) << bit_index;
    cuda::atomic_ref<bitmap_type, cuda::thread_scope_device> bitmap_ref(*bitmap_addr);
    bitmap_ref.fetch_and(~mask, cuda::memory_order_relaxed);
  }
};
