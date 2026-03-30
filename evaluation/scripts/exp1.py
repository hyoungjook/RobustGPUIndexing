from evaluate import *

CONFIGS = [
    {
        ConfigType.index_type: IndexType.gpu_masstree,
        ConfigType.repeats_scan: 10
    }
]

if __name__ == "__main__":
    args = parse_args()
    run_all_and_add_to_json(args, CONFIGS)
