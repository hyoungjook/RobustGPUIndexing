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

namespace simple_slab_alloc_helper {
DEVICE_QUALIFIER int cuda_ffs(const uint32_t& x) { return __ffs(x); }

DEVICE_QUALIFIER int cuda_popc(const uint32_t& x) { return __popc(x); }

constexpr uint32_t constexpr_log2(uint32_t n) {
  return (n > 1) ? (1 + constexpr_log2(n >> 1)) : 0;
}

template <typename block_layout, typename func, uint32_t tile_size>
__global__ void forall_blocks_kernel(block_layout* pool, uint32_t num_blocks, void* args) {
  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<tile_size>(block);
  uint32_t block_id = blockIdx.x;
  if (block_id >= num_blocks) return;
  block_layout* target_block = &pool[block_id];
  func{}(target_block, tile, args);
}

template <typename block_type, typename func>
void forall_blocks(void* pool, std::size_t num_blocks, void* args = nullptr) {
  using block_layout = typename block_type::block_layout;
  constexpr uint32_t tile_size = block_type::tile_size_;
  forall_blocks_kernel<block_layout, func, tile_size>
    <<<num_blocks, block_type::tile_size_>>>(
      reinterpret_cast<block_layout*>(pool), static_cast<uint32_t>(num_blocks), args);
  cuda_try(cudaDeviceSynchronize());
}
}

template <uint32_t slab_size>
struct simple_slab_block {
  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr uint32_t tile_size_ = 32;
  using bitmap_type = uint32_t;
  static constexpr uint32_t slab_count_ = sizeof(bitmap_type) * tile_size_ * 8;
  struct __align__(slab_size) slab_layout { uint8_t _[slab_size_]; };
  struct block_layout {
    slab_layout slabs[slab_count_];
    bitmap_type bitmap[tile_size_];
  };
  static constexpr uint32_t block_size_ = sizeof(block_layout);
};


template <uint32_t slab_size = 128, std::size_t max_bytes = 6UL * 1024 * 1024 * 1024>
struct simple_slab_allocator {
  using pointer_type = uint32_t;
  static constexpr uint32_t slab_size_ = slab_size;
  using block_type = simple_slab_block<slab_size_>;
  static constexpr uint32_t block_size_ = block_type::block_size_;
  static constexpr uint32_t num_blocks_in_superblock_ = 8 * 1024;
  static constexpr uint32_t superblock_size_ = num_blocks_in_superblock_ * block_size_;
  static constexpr std::size_t num_superblocks_ = max_bytes / superblock_size_;
  static constexpr std::size_t total_bytes_ = num_superblocks_ * superblock_size_;
  static constexpr std::size_t num_blocks_ = num_superblocks_ * num_blocks_in_superblock_;
  struct device_instance_type {
    void* pool_;
  };

  struct initialize_block {
    template <typename tile_type>
    DEVICE_QUALIFIER void operator()(typename block_type::block_layout* block, const tile_type& tile, void* _) const {
      block->bitmap[tile.thread_rank()] = 0;
    }
  };
  simple_slab_allocator() {
    cuda_try(cudaMalloc(&pool_, total_bytes_));
    simple_slab_alloc_helper::forall_blocks<block_type, initialize_block>(pool_, num_blocks_);
  }
  ~simple_slab_allocator() {
    cuda_try(cudaFree(pool_));
  }
  simple_slab_allocator(const simple_slab_allocator& other) = delete;
  simple_slab_allocator& operator=(const simple_slab_allocator& other) = delete;

  device_instance_type get_device_instance() const { return device_instance_type{pool_}; }

  struct count_slabs_block {
    template <typename tile_type>
    DEVICE_QUALIFIER void operator()(typename block_type::block_layout* block, const tile_type& tile, void* args) const {
      auto bitmap = block->bitmap[tile.thread_rank()];
      int count = simple_slab_alloc_helper::cuda_popc(bitmap);
      for (int offset = 16; offset > 0; offset >>= 1) {
        count += tile.shfl_down(count, offset);
      }
      if (tile.thread_rank() == 0) {
        auto* global_count = reinterpret_cast<unsigned long long int*>(args);
        atomicAdd(global_count, static_cast<unsigned long long int>(count));
      }
    }
  };
  void print_stats() const {
    uint64_t* d_slab_count;
    uint64_t h_slab_count = 0;
    cuda_try(cudaMalloc(&d_slab_count, sizeof(uint64_t)));
    cuda_try(cudaMemset(d_slab_count, 0x00, sizeof(uint64_t)));
    simple_slab_alloc_helper::forall_blocks<block_type, count_slabs_block>(pool_, num_blocks_, d_slab_count);
    cuda_try(cudaMemcpy(&h_slab_count, d_slab_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cuda_try(cudaFree(d_slab_count));
    const uint64_t total_slabs = num_blocks_ * block_type::slab_count_;
    std::cout << "simple_slab_allocator(" << slab_size_ << "B slabs, " << num_superblocks_ << " SBs): "
        << h_slab_count << "/" << total_slabs << " slabs allocated " 
        << "(" << (float)h_slab_count / total_slabs * 100.0f << "%)" << std::endl;
  }

private:
  void* pool_;
};

template <uint32_t slab_size, std::size_t max_bytes>
struct device_allocator_context<simple_slab_allocator<slab_size, max_bytes>> {
  using host_alloc_type = simple_slab_allocator<slab_size, max_bytes>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using pointer_type = typename host_alloc_type::pointer_type;

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc, const tile_type& tile)
      : alloc_(alloc) {
    assert(tile.size() == tile_size_);
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tile_id = thread_id / tile_size_;
    index_ = initialize_index(tile_id);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    auto* block = get_block_addr(index_);
    uint32_t block_index = try_allocate_in_block(block, tile);
    while (block_index == invalid_block_index_) {
      // linearly probe blocks
      index_++;
      if (index_ == num_blocks_) index_ = 0;
      block = get_block_addr(index_);
      block_index = try_allocate_in_block(block, tile);
    }
    assert(block_index < slabs_in_block_);
    return (index_ << block_idx_bits_) | static_cast<pointer_type>(block_index);
  }

  DEVICE_QUALIFIER void deallocate(pointer_type p) {
    auto* block = get_block_addr(p >> block_idx_bits_);
    deallocate_in_block(block, static_cast<uint32_t>(p & block_idx_mask_));
  }

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    auto* block = get_block_addr(p >> block_idx_bits_);
    return reinterpret_cast<void*>(&block->slabs[p & block_idx_mask_]);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  using block_type = simple_slab_block<slab_size_>;
  using block_layout = typename block_type::block_layout;
  static_assert(host_alloc_type::num_blocks_ * block_type::slab_count_ <= std::numeric_limits<pointer_type>::max());
  static constexpr uint32_t num_blocks_ = static_cast<uint32_t>(host_alloc_type::num_blocks_);
  static constexpr uint32_t num_superblocks_ = host_alloc_type::num_superblocks_;
  static constexpr uint32_t num_blocks_in_superblock_ = host_alloc_type::num_blocks_in_superblock_;
  static constexpr uint32_t subindex_bits_ = simple_slab_alloc_helper::constexpr_log2(num_blocks_in_superblock_);
  static constexpr pointer_type subindex_mask_ = static_cast<pointer_type>(num_blocks_in_superblock_) - 1;
  static constexpr uint64_t hash_coef_ = 0x5904ULL;
  static constexpr uint32_t tile_size_ = block_type::tile_size_;
  using bitmap_type = typename block_type::bitmap_type;
  static constexpr uint32_t invalid_block_index_ = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t slabs_in_block_ = block_type::slab_count_;
  static constexpr uint32_t block_idx_bits_ = simple_slab_alloc_helper::constexpr_log2(slabs_in_block_);
  static constexpr pointer_type block_idx_mask_ = static_cast<pointer_type>(slabs_in_block_) - 1;

  const device_instance_type& alloc_;
  // (index_ & subindex_mask_) is the index of block within a superblock
  // (index_ >> subindex_bits_) is the index of superblock
  pointer_type index_;

  DEVICE_QUALIFIER pointer_type compute_first_sub_index(uint32_t tile_id) const {
    return static_cast<pointer_type>(hash_coef_ * tile_id) & subindex_mask_;
  }
  DEVICE_QUALIFIER pointer_type initialize_index(uint32_t tile_id) const {
    pointer_type superblock_index = tile_id % num_superblocks_;
    pointer_type sub_index = compute_first_sub_index(tile_id);
    return (superblock_index << subindex_bits_) | sub_index;
  }
  DEVICE_QUALIFIER pointer_type rehash_index(pointer_type index, uint32_t tile_id) const {
    // circular-increment sub-index
    index = ((index & subindex_mask_) == subindex_mask_) ? (index & ~subindex_mask_) : (index + 1);
    if ((index & subindex_mask_) == compute_first_sub_index(tile_id)) {
      // circular-incremented one round, now circular-increment superblock index
      index += (subindex_mask_ + 1);
      index = (index < num_blocks_) ? index : (index & subindex_mask_);
    }
  }

  DEVICE_QUALIFIER block_layout* get_block_addr(uint32_t index) const {
    return reinterpret_cast<block_layout*>(alloc_.pool_) + index;
  }
  template <typename tile_type>
  DEVICE_QUALIFIER uint32_t try_allocate_in_block(block_layout* block, const tile_type& tile) {
    bitmap_type* bitmap_addr = &block->bitmap[tile.thread_rank()];
    bitmap_type bitmap = *bitmap_addr;
    uint32_t result = invalid_block_index_;
    while (result == invalid_block_index_) {
      auto empty_lane = simple_slab_alloc_helper::cuda_ffs(~bitmap) - 1;
      auto free_lane = tile.ballot(empty_lane >= 0);
      if (free_lane != 0) {
        uint32_t src_lane = simple_slab_alloc_helper::cuda_ffs(free_lane) - 1;
        if (src_lane == tile.thread_rank()) {
          bitmap_type mask = static_cast<bitmap_type>(1) << empty_lane;
          bitmap = atomicOr(bitmap_addr, mask);
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
        return invalid_block_index_;
      }
    }
    return result;
  }
  DEVICE_QUALIFIER void deallocate_in_block(block_layout* block, uint32_t block_index) {
    uint32_t bitmap_idx = block_index / (sizeof(bitmap_type) * 8);
    bitmap_type* bitmap_addr = &block->bitmap[bitmap_idx];
    uint32_t bit_idx = block_index % (sizeof(bitmap_type) * 8);
    bitmap_type mask = static_cast<bitmap_type>(1) << bit_idx;
    atomicAnd(bitmap_addr, ~mask);
  }
};


