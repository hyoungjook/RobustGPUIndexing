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
#include <host_allocators.hpp>
#include <macros.hpp>
#include <utils.hpp>
#include <cstdint>
#include <iostream>
#include <memory>

// refactoring of memory_reclaimer.hpp
#ifndef NDEBUG
//#define RECLAIMER_DEBUG
#endif

template <uint32_t buffer_size_per_block = 16384>
struct simple_debra_reclaimer {
  using size_type = uint32_t;
  using pointer_type = size_type;
#ifdef RECLAIMER_DEBUG
  struct debug_stats {
    unsigned long long num_retires;
    unsigned long long num_deallocates;
    unsigned max_bag_size;
    void print() {
      printf("simple_debra_reclaimer: retires(%llu) deallocs(%llu) maxbag(%u)",
        num_retires, num_deallocates, max_bag_size);
    }
  };
#endif
  struct device_instance_type {
    size_type* announce_;
    size_type* current_epoch_;
    size_type max_num_blocks_;
    size_type buffer_size_;
    #ifdef RECLAIMER_DEBUG
    debug_stats* stats_;
    #endif
  };
  simple_debra_reclaimer() {
    max_num_blocks_ = compute_max_num_blocks();
    buffer_size_ = max_num_blocks_ * (buffer_size_per_block / sizeof(size_type));
    cuda_try(cudaMalloc(&announce_, sizeof(size_type) * (max_num_blocks_ + buffer_size_)));
    cuda_try(cudaMalloc(&current_epoch_, sizeof(size_type)));
    cuda_try(cudaMemset(current_epoch_, 0, sizeof(size_type)));
    cuda_try(cudaMemset(announce_, 0, sizeof(size_type) * max_num_blocks_));
    #ifdef RECLAIMER_DEBUG
    cuda_try(cudaMalloc(&stats_, sizeof(debug_stats)));
    cuda_try(cudaMemset(stats_, 0, sizeof(debug_stats)));
    #endif
  }
  ~simple_debra_reclaimer() {
    #ifdef RECLAIMER_DEBUG
    debug_stats h_stats;
    cuda_try(cudaMemcpy(&h_stats, stats_, sizeof(debug_stats), cudaMemcpyDeviceToHost));
    h_stats.print();
    cuda_try(cudaFree(stats_));
    size_type h_final_epoch;
    cuda_try(cudaMemcpy(&h_final_epoch, current_epoch_, sizeof(size_type), cudaMemcpyDeviceToHost));
    printf(" final_epoch(%u)\n", h_final_epoch);
    #endif
    cuda_try(cudaFree(announce_));
    cuda_try(cudaFree(current_epoch_));
  }
  simple_debra_reclaimer(const simple_debra_reclaimer& other) = delete;
  simple_debra_reclaimer& operator=(const simple_debra_reclaimer& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{announce_, current_epoch_, max_num_blocks_, buffer_size_
      #ifdef RECLAIMER_DEBUG
      , stats_
      #endif
    };
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
  #ifdef RECLAIMER_DEBUG
  debug_stats* stats_;
  #endif
};

template <uint32_t buffer_size_per_block>
struct device_reclaimer_context<simple_debra_reclaimer<buffer_size_per_block>> {
  using host_reclaim_type = simple_debra_reclaimer<buffer_size_per_block>;
  using device_instance_type = typename host_reclaim_type::device_instance_type;
  using size_type = typename host_reclaim_type::size_type;
  using pointer_type = typename host_reclaim_type::pointer_type;
  static constexpr uint32_t block_size_ = host_reclaim_type::block_size_;
  __host__ __device__ static constexpr uint32_t required_shmem_size() {
    return (shmem_size_per_bag_ * num_bags_)    // limbo_bags
           + num_bags_                          // count_per_bag
           + 1                                  // cur_bag
           ;
  }

  template <typename block_type>
  DEVICE_QUALIFIER device_reclaimer_context(const device_instance_type& reclaimer,
                                            size_type* shmem_buffer,
                                            size_type num_active_blocks,
                                            const block_type& block)
      : reclaimer_(reclaimer)
      , shmem_buffer_(shmem_buffer)
      , num_active_blocks_(num_active_blocks) {
    assert(num_active_blocks_ < reclaimer_.max_num_blocks_);
    assert(block.size() == block_size_);
    buffer_size_per_active_block_ = reclaimer_.buffer_size_ / num_active_blocks_;
    // initialize shmem buffer to all zero
    for (uint32_t i = block.thread_rank(); i < required_shmem_size(); i += block_size_) {
      shmem_buffer[i] = 0;
    }
    block.sync();
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void retire(const pointer_type& address, const tile_type& tile) {
    // add to current limbo bag
    if (tile.thread_rank() == 0) {
      auto cur_bag = current_bag();
      auto num_in_bag = atomicAdd(count_per_bag() + cur_bag, 1);
      #ifdef RECLAIMER_DEBUG
      atomicAdd(&reclaimer_.stats_->num_retires, 1ull);
      atomicMax(&reclaimer_.stats_->max_bag_size, num_in_bag + 1);
      #endif
      // if shared memory buffer overflow, store in global memory
      if (num_in_bag >= shmem_size_per_bag_) {
        auto num_in_gmem = num_in_bag - shmem_size_per_bag_;
        assert(num_in_gmem < (buffer_size_per_active_block_ / num_bags_)); // out of memory in bag
        auto offset = reclaimer_.max_num_blocks_ +
            buffer_size_per_active_block_ * blockIdx.x +
            (buffer_size_per_active_block_ / num_bags_) * cur_bag +
            num_in_gmem;
        // atomic store is not required here b/c it's used by warps in the same block after block.sync()
        reclaimer_.announce_[offset] = address;
      }
      else {
        auto offset = cur_bag * shmem_size_per_bag_ + num_in_bag;
        limbo_bags()[offset] = address;
      }
    }
  }

  template <typename block_type, typename allocator_type>
  DEVICE_QUALIFIER void begin_critical_section(const block_type& block, allocator_type& allocator) {
    block.sync();
    __threadfence();

    auto cur_epoch = utils::memory::load<size_type, true>(reclaimer_.current_epoch_);
    if (try_update_global_announce(block, cur_epoch, critical_bit_mask_, 0)) {
      reclaim_and_switch_limbo_bag(block, allocator);
    }
    scan_announce_and_advance_epoch(block, cur_epoch);
  }

  template <typename block_type>
  DEVICE_QUALIFIER void end_critical_section(const block_type& block) {
    // unset critical bit in global memory announce array
    block.sync();
    if (block.thread_rank() == 0) {
      atomicAnd(reclaimer_.announce_ + blockIdx.x, ~critical_bit_mask_);
    }
    block.sync();
    __threadfence();
  }

  template <typename block_type, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void drain_all(const block_type& block, const tile_type& tile, allocator_type& allocator) {
    // only one tile is required, exit the rest to free hardware resource
    if (block.thread_rank() >= tile_size_) { return; }
    while (true) {
      // check all bags are empty
      bool bag_empty = true;
      static_assert(num_bags_ <= tile_size_);
      if (tile.thread_rank() < num_bags_) {
        bag_empty = (count_per_bag()[tile.thread_rank()] == 0);
      }
      if (tile.all(bag_empty)) { return; }

      // try announcing
      auto cur_epoch = utils::memory::load<size_type, true>(reclaimer_.current_epoch_);
      if (try_update_global_announce(tile, cur_epoch, 0, 0)) {
        reclaim_and_switch_limbo_bag(tile, allocator);
      }
      scan_announce_and_advance_epoch(tile, cur_epoch);
    }
  }

private:
  template <typename tile_type>
  DEVICE_QUALIFIER bool try_update_global_announce(const tile_type& tile, size_type cur_epoch, size_type new_mask, size_type old_mask) {
    bool epoch_advanced = false;
    if (tile.thread_rank() == 0) {
      auto old_epoch = atomicExch(reclaimer_.announce_ + blockIdx.x, cur_epoch | new_mask);
      assert((old_epoch & critical_bit_mask_) == old_mask);
      if ((old_epoch & ~old_mask) != cur_epoch) {
        epoch_advanced = true;
      }
    }
    return tile.shfl(epoch_advanced, 0);
  }

  template <typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void reclaim_and_switch_limbo_bag(const tile_type& tile, allocator_type& allocator) {
    // reclaim current limbo bag and change current_bag
    uint32_t cur_bag = current_bag();
    cur_bag = (cur_bag + 1) % num_bags_;
    auto total_count = count_per_bag()[cur_bag];
    #ifdef RECLAIMER_DEBUG
    if (tile.thread_rank() == 0) {
      atomicAdd(&reclaimer_.stats_->num_deallocates, total_count);
    }
    #endif
    // free pointers in shared memory
    auto count_in_shmem = min(total_count, shmem_size_per_bag_);
    for (uint32_t i = tile.thread_rank(); i < count_in_shmem; i += tile.size()) {
      auto address = limbo_bags()[cur_bag * shmem_size_per_bag_ + i];
      allocator.deallocate(address);
    }
    // free pointers in global memory
    if (total_count > shmem_size_per_bag_) {
      auto count_in_gmem = total_count - shmem_size_per_bag_;
      auto announce_offset = reclaimer_.max_num_blocks_ +
            buffer_size_per_active_block_ * blockIdx.x +
            (buffer_size_per_active_block_ / num_bags_) * cur_bag;
      for (uint32_t i = tile.thread_rank(); i < count_in_gmem; i += tile.size()) {
        auto address = reclaimer_.announce_[announce_offset + i];
        allocator.deallocate(address);
      }
    }
    tile.sync();
    // reset the counter
    if (tile.thread_rank() == 0) {
      count_per_bag()[cur_bag] = 0;
      current_bag() = cur_bag;
    }
    tile.sync();
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void scan_announce_and_advance_epoch(const tile_type& tile, size_type cur_epoch) {
    auto rounded_active_blocks = (num_active_blocks_ + tile.size() - 1) / tile.size() * tile.size();
    for (uint32_t i = tile.thread_rank(); i < rounded_active_blocks; i += tile.size()) {
      bool is_quiescent = true;
      bool is_epoch_up_to_date = true;
      if (i < num_active_blocks_) {
        auto epoch = utils::memory::load<size_type, true>(reclaimer_.announce_ + i);
        is_quiescent = (epoch & critical_bit_mask_) == 0;
        is_epoch_up_to_date = ((epoch & ~critical_bit_mask_) == cur_epoch);
      }
      bool advanced = is_quiescent || is_epoch_up_to_date;
      if (!tile.all(advanced)) { return; }
    }
    if (tile.thread_rank() == 0) {
      atomicCAS(reclaimer_.current_epoch_, cur_epoch, cur_epoch + 2);
    }
  }

  static constexpr uint32_t num_bags_ = 3;
  static constexpr uint32_t shmem_size_per_bag_ = 128;
  static constexpr size_type critical_bit_mask_ = 0x1;
  static constexpr uint32_t tile_size_ = 32;

  DEVICE_QUALIFIER pointer_type* limbo_bags() const { return shmem_buffer_; }
  DEVICE_QUALIFIER size_type* count_per_bag() const { return shmem_buffer_ + (shmem_size_per_bag_ * num_bags_); }
  DEVICE_QUALIFIER size_type& current_bag() const { return shmem_buffer_[(shmem_size_per_bag_ * num_bags_) + num_bags_]; }

  const device_instance_type& reclaimer_;
  size_type* shmem_buffer_;
  size_type num_active_blocks_;
  size_type buffer_size_per_active_block_;
};
