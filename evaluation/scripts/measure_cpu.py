from evaluate import *
from constants import *

def generate_configs():
    configs = []
    # different table sizes
    for index_type in INDEX_TYPES_CPU_BASELINE:
        for table_size in EXP_TABLE_SIZES:
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: table_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.delete_ratio: DEFAULT_DELETE_RATIO,
                ConfigType.num_lookups: DEFAULT_NUM_KEYS,
                ConfigType.repeats_insert: NUM_REPEATS,
                ConfigType.repeats_delete: NUM_REPEATS,
                ConfigType.repeats_lookup: NUM_REPEATS,
                ConfigType.repeats_scan: 0,
            })
    for index_type in INDEX_TYPES_ORDERED_IN(INDEX_TYPES_CPU_BASELINE):
        for table_size in EXP_TABLE_SIZES:
            configs.append({
                ConfigType.index_type: index_type,
                ConfigType.num_keys: table_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.num_scans: DEFAULT_NUM_KEYS,
                ConfigType.scan_count: DEFAULT_SCAN_COUNT,
                ConfigType.repeats_insert: 0,
                ConfigType.repeats_delete: 0,
                ConfigType.repeats_lookup: 0,
                ConfigType.repeats_scan: NUM_REPEATS,
            })
    # different key lengths
    for index_type in INDEX_TYPES_CPU_BASELINE:
        for key_length in EXP_KEY_LENGTHS:
            for prefix_length in [0, key_length - 1]:
                configs.append({
                    ConfigType.index_type: index_type,
                    ConfigType.num_keys: DEFAULT_NUM_KEYS,
                    ConfigType.keylen_prefix: prefix_length,
                    ConfigType.keylen_min: key_length,
                    ConfigType.keylen_max: key_length,
                    ConfigType.num_lookups: DEFAULT_NUM_KEYS,
                    ConfigType.repeats_insert: 0,
                    ConfigType.repeats_delete: 0,
                    ConfigType.repeats_lookup: NUM_REPEATS,
                    ConfigType.repeats_scan: 0,
                })
    return configs

if __name__ == "__main__":
    args = parse_args_for_measure()
    configs = generate_configs()
    run_all_and_add_to_json(args, configs, "result_cpu", 0)
