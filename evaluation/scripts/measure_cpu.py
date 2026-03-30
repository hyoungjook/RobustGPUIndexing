from evaluate import *

NUM_REPEATS = 100

MILLION = 1000000
DEFAULT_NUM_KEYS = 10 * MILLION
DEFAULT_KEY_LENGHT = 4
DEFAULT_DELETE_RATIO = 0.1

def generate_configs():
    configs = []
    # different key lengths
    for index_type in [IndexType.cpu_art,
                       IndexType.cpu_masstree,
                       IndexType.cpu_libcuckoo]:
        for key_length in [2, 4, 8, 16, 32]:
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: DEFAULT_NUM_KEYS,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: key_length,
                ConfigType.keylen_max: key_length,
                ConfigType.delete_ratio: DEFAULT_DELETE_RATIO,
                ConfigType.num_lookups: DEFAULT_NUM_KEYS,
                ConfigType.repeats_insert: NUM_REPEATS,
                ConfigType.repeats_delete: NUM_REPEATS,
                ConfigType.repeats_lookup: NUM_REPEATS,
                ConfigType.repeats_scan: 0,
            })
    # different table sizes
    for index_type in [IndexType.cpu_art,
                       IndexType.cpu_masstree,
                       IndexType.cpu_libcuckoo]:
        for table_size in [100000, MILLION, 10 * MILLION, 100 * MILLION]:
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: table_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.delete_ratio: DEFAULT_DELETE_RATIO,
                ConfigType.num_lookups: table_size,
                ConfigType.repeats_insert: NUM_REPEATS,
                ConfigType.repeats_delete: NUM_REPEATS,
                ConfigType.repeats_lookup: NUM_REPEATS,
                ConfigType.repeats_scan: 0,
            })
    # different prefix lengths for trees
    for index_type in [IndexType.cpu_art,
                       IndexType.cpu_masstree]:
        for prefix in [0, 1, 2, 4, 7]:
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: DEFAULT_NUM_KEYS,
                ConfigType.keylen_prefix: prefix,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.delete_ratio: DEFAULT_DELETE_RATIO,
                ConfigType.num_lookups: DEFAULT_NUM_KEYS,
                ConfigType.num_scans: DEFAULT_NUM_KEYS,
                ConfigType.scan_count: 1,
                ConfigType.repeats_insert: NUM_REPEATS,
                ConfigType.repeats_delete: NUM_REPEATS,
                ConfigType.repeats_lookup: NUM_REPEATS,
                ConfigType.repeats_scan: NUM_REPEATS,
            })
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: DEFAULT_NUM_KEYS,
                ConfigType.keylen_prefix: prefix,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.num_scans: DEFAULT_NUM_KEYS,
                ConfigType.scan_count: 2,
                ConfigType.repeats_insert: 0,
                ConfigType.repeats_delete: 0,
                ConfigType.repeats_lookup: 0,
                ConfigType.repeats_scan: NUM_REPEATS,
            })
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: DEFAULT_NUM_KEYS,
                ConfigType.keylen_prefix: prefix,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.num_scans: DEFAULT_NUM_KEYS,
                ConfigType.scan_count: 4,
                ConfigType.repeats_insert: 0,
                ConfigType.repeats_delete: 0,
                ConfigType.repeats_lookup: 0,
                ConfigType.repeats_scan: NUM_REPEATS,
            })
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_cpu")
