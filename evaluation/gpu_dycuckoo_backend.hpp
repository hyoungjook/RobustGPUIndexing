#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

void* gpu_dycuckoo_dynamic_create(std::uint32_t init_kv_num,
                                  int small_batch_size,
                                  double lower_bound,
                                  double upper_bound);
void gpu_dycuckoo_dynamic_destroy(void* index);
void gpu_dycuckoo_dynamic_insert(void* index, const uint32_t* keys, const uint32_t* values, uint32_t num_keys);
void gpu_dycuckoo_dynamic_erase(void* index, const uint32_t* keys, uint32_t num_keys);
void gpu_dycuckoo_dynamic_find(void* index, const uint32_t* keys, uint32_t* results, uint32_t num_keys);

void* gpu_dycuckoo_dynamic_lock_create(std::uint32_t init_kv_num,
                                       int small_batch_size,
                                       double lower_bound,
                                       double upper_bound);
void gpu_dycuckoo_dynamic_lock_destroy(void* index);
void gpu_dycuckoo_dynamic_lock_insert(void* index, const uint32_t* keys, const uint32_t* values, uint32_t num_keys);
void gpu_dycuckoo_dynamic_lock_erase(void* index, const uint32_t* keys, uint32_t num_keys);
void gpu_dycuckoo_dynamic_lock_find(void* index, const uint32_t* keys, uint32_t* results, uint32_t num_keys);

}
