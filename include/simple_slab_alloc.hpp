/*
 *   Copyright 2022 The Regents of the University of California, Davis
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
#include <bitset>
#include <cstdint>
#include <iostream>
#include <macros.hpp>
#include <memory>
#include <type_traits>
#include <host_allocators.hpp>
#include <device_context.hpp>

template <uint32_t slab_size = 128>
struct simple_slab_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;
  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr uint32_t num_slabs_in_block_ = 32 * sizeof(uint32_t) * 8;
  static_assert(num_slabs_in_block_ == 16 * sizeof(uint64_t) * 8);
  static constexpr uint32_t block_size_ = slab_size_ * num_slabs_in_block_;
  struct device_instance_type {
    void* pool_;
    pointer_type total_blocks_;
  };

  simple_slab_allocator(float pool_ratio = 0.9f) {
    auto meminfo = utils::compute_device_memory_usage();
    auto max_bytes = static_cast<std::size_t>(static_cast<double>(meminfo.total_bytes) * pool_ratio);
    total_blocks_ = static_cast<pointer_type>(max_bytes / block_size_);
    if (static_cast<std::size_t>(total_blocks_) * num_slabs_in_block_ >= std::numeric_limits<pointer_type>::max()) {
      std::cerr << "simple_slab_allocator: pointer exceeds uint32 limit" << std::endl;
      abort();
    }
    auto total_bytes = static_cast<std::size_t>(total_blocks_) * block_size_;
    auto total_bitmap_bytes = (static_cast<std::size_t>(total_blocks_) * num_slabs_in_block_) / 8;
    cuda_try(cudaMalloc(&pool_, total_bitmap_bytes + total_bytes));
    cuda_try(cudaMemset(pool_, 0x00, total_bitmap_bytes));
    pool_ = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(pool_) + total_bitmap_bytes);
  }
  ~simple_slab_allocator() {
    auto total_bitmap_bytes = (static_cast<std::size_t>(total_blocks_) * num_slabs_in_block_) / 8;
    pool_ = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(pool_) - total_bitmap_bytes);
    cuda_try(cudaFree(pool_));
  }
  simple_slab_allocator(const simple_slab_allocator& other) = delete;
  simple_slab_allocator& operator=(const simple_slab_allocator& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{pool_, total_blocks_};
  }

  void print_stats() const {
    using bitmap_type = uint32_t;
    auto total_bitmap_bytes = (static_cast<std::size_t>(total_blocks_) * num_slabs_in_block_) / 8;
    bitmap_type *h_bitmap = new bitmap_type[total_bitmap_bytes / sizeof(bitmap_type)];
    cuda_try(cudaMemcpy(h_bitmap,
                        reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(pool_) - total_bitmap_bytes),
                        total_bitmap_bytes,
                        cudaMemcpyDeviceToHost));
    std::size_t slab_count = 0;
    for (std::size_t i = 0; i < total_bitmap_bytes / sizeof(bitmap_type); i++) {
      slab_count += __builtin_popcount(h_bitmap[i]);
    }
    delete[] h_bitmap;
    const uint64_t total_slabs = total_blocks_ * num_slabs_in_block_;
    std::cout << "simple_slab_allocator(" << slab_size_ << "B slabs): "
        << slab_count << "/" << total_slabs << " slabs allocated " 
        << "(" << (float)slab_count / total_slabs * 100.0f << "%)" << std::endl;
  }

private:
  void* pool_;
  pointer_type total_blocks_;
};

template <uint32_t slab_size>
struct device_allocator_context<simple_slab_allocator<slab_size>> {
  using host_alloc_type = simple_slab_allocator<slab_size>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using size_type = typename host_alloc_type::size_type;
  using pointer_type = typename host_alloc_type::pointer_type;

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc, const tile_type& tile)
      : alloc_(alloc) {
    static_assert(tile_type::size() == 32 || tile_type::size() == 16);
    block_index_ = initialize_index(get_tile_id<tile_type>());
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
    return 0;
  }
  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(uint32_t sum, const tile_type& tile) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    return reinterpret_cast<void*>(reinterpret_cast<slab_type*>(alloc_.pool_) + p);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint32_t num_slabs_in_block_ = host_alloc_type::num_slabs_in_block_;
  static constexpr pointer_type invalid_pointer = std::numeric_limits<pointer_type>::max();
  struct slab_type { uint8_t _[slab_size_]; };
  struct block_bitmap_type { uint8_t _[num_slabs_in_block_ / 8]; };

  const device_instance_type& alloc_;
  pointer_type block_index_;

  static constexpr uint32_t mix_prime = 0x9e3779b1;
  template <typename tile_type>
  DEVICE_QUALIFIER uint32_t get_tile_id() const {
    return (threadIdx.x + blockIdx.x * blockDim.x) / tile_type::size();
  }
  DEVICE_QUALIFIER pointer_type initialize_index(uint32_t tile_id) const {
    return static_cast<pointer_type>(tile_id * mix_prime) % alloc_.total_blocks_;
  }
  DEVICE_QUALIFIER pointer_type next_index(pointer_type index, uint32_t trials) const {
    // linear probing
    //return (index + trials) % total_blocks_;
    // quadratic probing
    return (index + trials * trials) % alloc_.total_blocks_;
  }
  DEVICE_QUALIFIER pointer_type total_bitmap_bytes() const {
    return alloc_.total_blocks_ * (num_slabs_in_block_ / 8);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type try_allocate_in_block(pointer_type block_index, const tile_type& tile) {
    using bitmap_type = std::conditional_t<tile_type::size() == 32, uint32_t, uint64_t>;
    auto* bitmap_base = reinterpret_cast<uint8_t*>(alloc_.pool_) - total_bitmap_bytes();
    bitmap_type* bitmap_addr =
        reinterpret_cast<bitmap_type*>(
          reinterpret_cast<block_bitmap_type*>(bitmap_base) + block_index) + tile.thread_rank();
    bitmap_type bitmap = *bitmap_addr;
    pointer_type result = invalid_pointer;
    while (result == invalid_pointer) {
      int empty_lane;
      if constexpr (tile_type::size() == 32) {
        empty_lane = __ffs(~bitmap) - 1;
      }
      else {
        empty_lane = __ffsll(~bitmap) - 1;
      }
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
    using bitmap_type = uint32_t;
    auto* bitmap_base = reinterpret_cast<uint8_t*>(alloc_.pool_) - total_bitmap_bytes();
    auto bitmap_index = ptr / (sizeof(bitmap_type) * 8);
    auto bit_index = ptr % (sizeof(bitmap_type) * 8);
    bitmap_type* bitmap_addr =
        reinterpret_cast<bitmap_type*>(bitmap_base) + bitmap_index;
    bitmap_type mask = static_cast<bitmap_type>(1) << bit_index;
    cuda::atomic_ref<bitmap_type, cuda::thread_scope_device> bitmap_ref(*bitmap_addr);
    [[maybe_unused]] auto old = bitmap_ref.fetch_and(~mask, cuda::memory_order_relaxed);
    assert((old & mask) != 0); // double free
  }
};
