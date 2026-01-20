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
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <host_allocators.hpp>
#include <macros.hpp>
#include <cstdint>
#include <iostream>
#include <memory>

// refactoring of memory_reclaimer.hpp

template <uint32_t buffer_size_per_block = 4096>
struct simple_debr_reclaimer {
  using size_type = uint32_t;
  using pointer_type = size_type;
  struct device_instance_type {
    size_type* announce_;
    size_type* current_epoch_;
    size_type max_num_blocks_;
    size_type buffer_size_;
  };
  simple_debr_reclaimer() {
    max_num_blocks_ = compute_max_num_blocks();
    buffer_size_ = max_num_blocks_ * (buffer_size_per_block / sizeof(size_type));
    cuda_try(cudaMalloc(&announce_, sizeof(size_type) * (max_num_blocks_ + buffer_size_)));
    cuda_try(cudaMalloc(&current_epoch_, sizeof(size_type)));
    cuda_try(cudaMemset(current_epoch_, 0x00, sizeof(size_type)));
    thrust::fill(thrust::device, announce_, announce_ + max_num_blocks_, 0x1);
  }
  ~simple_debr_reclaimer() {
    cuda_try(cudaFree(announce_));
    cuda_try(cudaFree(current_epoch_));
  }
  simple_debr_reclaimer(const simple_debr_reclaimer& other) = delete;
  simple_debr_reclaimer& operator=(const simple_debr_reclaimer& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{announce_, current_epoch_, max_num_blocks_, buffer_size_};
  }

  static constexpr uint32_t block_size_ = 128;

private:
  uint32_t compute_max_num_blocks() const {
    // upper bound of gridDim
    // which is computed using cudaOccupancyMaxActiveBlocksPerMultiprocessor()
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
  }

  size_type* announce_;
  size_type* current_epoch_;
  size_type max_num_blocks_;
  size_type buffer_size_;
};

template <uint32_t buffer_size_per_block>
struct device_reclaimer_context<simple_debr_reclaimer<buffer_size_per_block>> {
  using host_reclaim_type = simple_debr_reclaimer<buffer_size_per_block>;
  using device_instance_type = typename host_reclaim_type::device_instance_type;
  using size_type = typename host_reclaim_type::size_type;
  using pointer_type = typename host_reclaim_type::pointer_type;
  static constexpr uint32_t block_size_ = host_reclaim_type::block_size_;
  __host__ __device__ static constexpr uint32_t required_shmem_size() {
    return (shmem_size_per_bag_ * num_bags_)    // limbo_bags
           + num_bags_                          // count_per_bag
           + 1                                  // cur_bag
           + 1;                                 // epoch_advanced
  }

  template <typename tile_type>
  DEVICE_QUALIFIER device_reclaimer_context(const device_instance_type& reclaimer,
                                            size_type* shmem_buffer,
                                            size_type num_active_blocks,
                                            const tile_type& block_wide_tile)
      : reclaimer_(reclaimer)
      , shmem_buffer_(shmem_buffer)
      , num_active_blocks_(num_active_blocks) {
    assert(num_active_blocks_ < reclaimer_.max_num_blocks_);
    buffer_size_per_active_block_ = reclaimer_.buffer_size_ / num_active_blocks_;
    // initialize shmem buffer to all zero
    block_wide_tile.sync();
    for (uint32_t i = block_wide_tile.thread_rank();
         i < required_shmem_size();
         i += block_wide_tile.size()) {
      shmem_buffer[i] = 0;
    }
    block_wide_tile.sync();
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void retire(const pointer_type& address, const tile_type& tile) {
    // add to current limbo bag
    if (tile.thread_rank() == 0) {
      auto cur_bag = current_bag();
      auto num_in_bag = atomicAdd(count_per_bag() + cur_bag, 1);
      // if shared memory buffer overflow, store in global memory
      if (num_in_bag >= shmem_size_per_bag_) {
        assert((num_in_bag - shmem_size_per_bag_) < (buffer_size_per_active_block_ / num_bags_ - 1));
        auto offset = reclaimer_.max_num_blocks_ +
            buffer_size_per_active_block_ * blockIdx.x +
            (buffer_size_per_active_block_ / num_bags_) * cur_bag +
            (num_in_bag - shmem_size_per_bag_);
        cuda_memory<pointer_type>::store(reclaimer_.announce_ + offset, address);
      }
      else {
        auto offset = cur_bag * shmem_size_per_bag_ + num_in_bag;
        limbo_bags()[offset] = address;
      }
    }
  }

  template <typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void leave_qstate(const tile_type& block_wide_tile, allocator_type& allocator) {
    block_wide_tile.sync();
    __threadfence();

    // read current epoch
    auto cur_epoch = cuda_memory<size_type>::load(reclaimer_.current_epoch_, cuda_memory_order::memory_order_relaxed);
    assert((cur_epoch & quiescent_bit_mask_) == 0);
    size_type old_epoch;
    // atomically unset quiescent bit and set epoch num
    if (block_wide_tile.thread_rank() == 0) {
      old_epoch = atomicExch(reclaimer_.announce_ + blockIdx.x, cur_epoch);
      assert((old_epoch & quiescent_bit_mask_) != 0);
      old_epoch = old_epoch & ~quiescent_bit_mask_;
      if (old_epoch != cur_epoch) {
        epoch_advanced() = true;
      }
    }
    block_wide_tile.sync();

    // if advancing epoch, reclaim current limbo bag and change cur_bag
    if (epoch_advanced()) {
      uint32_t cur_bag = current_bag();
      cur_bag = (cur_bag + 1) % num_bags_;
      auto total_count = count_per_bag()[cur_bag];
      // free pointers in shared memory
      auto count_in_shmem = min(total_count, shmem_size_per_bag_);
      for (uint32_t i = block_wide_tile.thread_rank();
           i < count_in_shmem;
           i += block_wide_tile.size()) {
        auto address = limbo_bags()[cur_bag * shmem_size_per_bag_ + i];
        allocator.deallocate(address);
      }
      // free pointers in global memory
      if (total_count >= shmem_size_per_bag_) {
        auto count_in_gmem = total_count - shmem_size_per_bag_;
        auto announce_offset = reclaimer_.max_num_blocks_ +
              buffer_size_per_active_block_ * blockIdx.x +
              (buffer_size_per_active_block_ / num_bags_) * cur_bag;
        for (uint32_t i = block_wide_tile.thread_rank();
             i < count_in_gmem;
             i += block_wide_tile.size()) {
          auto address = reclaimer_.announce_[announce_offset + i];
          allocator.deallocate(address);
        }
      }
      block_wide_tile.sync();
      // reset the counter
      if (block_wide_tile.thread_rank() == 0) {
        count_per_bag()[cur_bag] = 0;
        current_bag() = cur_bag;
        epoch_advanced() = false;
      }
      block_wide_tile.sync();
    }

    // check announce array and advance epoch
    auto rounded_active_blocks = (num_active_blocks_ + block_wide_tile.size() - 1) / block_wide_tile.size() * block_wide_tile.size();
    for (uint32_t i = block_wide_tile.thread_rank();
         i < rounded_active_blocks;
         i += block_wide_tile.size()) {
      bool is_quiescent = true;
      size_type epoch = cur_epoch;
      if (i < num_active_blocks_) {
        epoch = cuda_memory<size_type>::load(&reclaimer_.announce_[i], cuda_memory_order::memory_order_relaxed);
        is_quiescent = (epoch & quiescent_bit_mask_) != 0;
        epoch = (epoch & ~quiescent_bit_mask_);
      }
      bool advanced = is_quiescent || (epoch == cur_epoch);
      if (!block_wide_tile.all(advanced)) { return; }
    }
    if (block_wide_tile.thread_rank() == 0) {
      atomicCAS(reclaimer_.current_epoch_, cur_epoch, cur_epoch + 2);
    }
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void enter_qstate(const tile_type& block_wide_tile) {
    block_wide_tile.sync();
    if (block_wide_tile.thread_rank() == 0) {
      atomicOr(reclaimer_.announce_ + blockIdx.x, quiescent_bit_mask_);
    }
    block_wide_tile.sync();
    __threadfence();
  }

private:
  static constexpr uint32_t num_bags_ = 3;
  static constexpr uint32_t shmem_size_per_bag_ = 128;
  static constexpr size_type quiescent_bit_mask_ = 0x1;

  DEVICE_QUALIFIER pointer_type* limbo_bags() const { return shmem_buffer_; }
  DEVICE_QUALIFIER size_type* count_per_bag() const { return shmem_buffer_ + (shmem_size_per_bag_ * num_bags_); }
  DEVICE_QUALIFIER size_type& current_bag() const { return shmem_buffer_[(shmem_size_per_bag_ * num_bags_) + num_bags_]; }
  DEVICE_QUALIFIER size_type& epoch_advanced() const { return shmem_buffer_[(shmem_size_per_bag_ * num_bags_) + num_bags_ + 1]; }

  const device_instance_type& reclaimer_;
  size_type* shmem_buffer_;
  size_type num_active_blocks_;
  size_type buffer_size_per_active_block_;
};
