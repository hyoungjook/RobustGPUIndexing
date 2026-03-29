#include <cstdint>
#include <limits>
#include <gpu_dycuckoo_backend.hpp>
#include <macros.hpp>

#define DataLayout DycuckooLockDataLayout
#define DynamicHash DycuckooLockDynamicHash
#define DynamicCuckoo DycuckooLockDynamicCuckoo
#define cuckoo_helpers dycuckoo_lock_cuckoo_helpers
#define hashers dycuckoo_lock_hashers
#include "../baselines/DyCuckoo/dynamicHash_lock/core/dynamic_cuckoo.cuh"
#undef hashers
#undef cuckoo_helpers
#undef DynamicCuckoo
#undef DynamicHash
#undef DataLayout

namespace {
using index_type = DycuckooLockDynamicCuckoo<512, 512>;
using value_type = DycuckooLockDataLayout<>::value_t;
}  // namespace

extern "C" void* gpu_dycuckoo_dynamic_lock_create(uint32_t init_kv_num,
                                                  int small_batch_size,
                                                  double lower_bound,
                                                  double upper_bound) {
  return new index_type(init_kv_num, small_batch_size, lower_bound, upper_bound);
}

extern "C" void gpu_dycuckoo_dynamic_lock_destroy(void* index) {
  delete reinterpret_cast<index_type*>(index);
}

extern "C" void gpu_dycuckoo_dynamic_lock_insert(void* index,
                                                 const uint32_t* keys,
                                                 const uint32_t* values,
                                                 uint32_t num_keys) {
  reinterpret_cast<index_type*>(index)->batch_insert(
    const_cast<uint32_t*>(keys), reinterpret_cast<value_type*>(const_cast<uint32_t*>(values)), num_keys);
}

extern "C" void gpu_dycuckoo_dynamic_lock_erase(void* index,
                                                const uint32_t* keys,
                                                uint32_t num_keys) {
  reinterpret_cast<index_type*>(index)->batch_delete(
    const_cast<uint32_t*>(keys), nullptr, num_keys);
}

extern "C" void gpu_dycuckoo_dynamic_lock_find(void* index,
                                               const uint32_t* keys,
                                               uint32_t* results,
                                               uint32_t num_keys) {
  reinterpret_cast<index_type*>(index)->batch_search(
    const_cast<uint32_t*>(keys), reinterpret_cast<value_type*>(results), num_keys);
}
