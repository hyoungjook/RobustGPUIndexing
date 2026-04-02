from evaluate import *
from constants import *
import matplotlib.legend_handler as mlegh
import matplotlib.lines as mline
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation

IndexType_gpu_masstree_no_suffix = 'gpu_masstree_no_suffix'
IndexType_gpu_extendht_no_hashtag = 'gpu_extendht_no_hashtag'
IndexType_gpu_dycuckoo_with_lock = 'gpu_dycuckoo_with_lock'
INDEX_LABELS = {
    IndexType.gpu_masstree: "GPUMasstree",
    IndexType.gpu_chainhashtable: "GPUChainHT",
    IndexType.gpu_cuckoohashtable: "GPUCuckooHT",
    IndexType.gpu_extendhashtable: "GPUExtendHT",
    IndexType.gpu_blink_tree: "GPUBtree",
    IndexType.gpu_dycuckoo: "GPUDyCuckoo",
    IndexType.cpu_art: "(CPU)ART",
    IndexType.cpu_masstree: "(CPU)Masstree",
    IndexType.cpu_libcuckoo: "(CPU)Libcuckoo",
    IndexType_gpu_masstree_no_suffix: "GPUMasstree (no suffix)",
    IndexType_gpu_extendht_no_hashtag: "GPUExtendHT (no hashtag)",
    IndexType_gpu_dycuckoo_with_lock: "GPUDyCuckoo (with lock)"
}
INDEX_STYLES = {
    IndexType.gpu_masstree: {"color": "#0B6E4F", "marker": "o"},
    IndexType.gpu_chainhashtable: {"color": "#D1495B", "marker": "s"},
    IndexType.gpu_cuckoohashtable: {"color": "#00798C", "marker": "^"},
    IndexType.gpu_extendhashtable: {"color": "#EDAE49", "marker": "D"},
    IndexType.gpu_blink_tree: {"color": "#8F2D56", "marker": "<"},
    IndexType.gpu_dycuckoo: {"color": "#3B60E4", "marker": "h"},
    IndexType.cpu_art: {"color": "#5C4D7D", "marker": "P"},
    IndexType.cpu_masstree: {"color": "#6C9A8B", "marker": "X"},
    IndexType.cpu_libcuckoo: {"color": "#9C6644", "marker": "v"},
    IndexType_gpu_masstree_no_suffix: {"color": "#549A84", "marker": "*"},
    IndexType_gpu_extendht_no_hashtag: {"color": "#CE973E", "marker": "d"},
    IndexType_gpu_dycuckoo_with_lock: {"color": "#7993EF", "marker": "H"},
}

def _table_size_label(value, _):
    if value >= MILLION:
        return f"{int(value / MILLION)}M"
    if value >= 1000:
        return f"{int(value / 1000)}K"
    return str(int(value))

def _convert_mops_to_gops(values):
        return [v / 1000 for v in values]

def table_size_plots(configs_and_results, plot_file):
    insert_tputs = {}
    delete_tputs = {}
    lookup_tputs = {}
    scan_tputs = {}
    all_index_types = INDEX_TYPES_ROBUST + INDEX_TYPES_CPU_BASELINE
    scan_index_types = INDEX_TYPES_ORDERED_IN(all_index_types)
    for index_type in all_index_types:
        insert_tputs[index_type] = []
        delete_tputs[index_type] = []
        lookup_tputs[index_type] = []
        for table_size in EXP_TABLE_SIZES:
            desired_config = {
                ConfigType.index_type: index_type,
                ConfigType.num_keys: table_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.delete_ratio: DEFAULT_DELETE_RATIO,
                ConfigType.num_lookups: DEFAULT_NUM_KEYS
            }
            if index_type in INDEX_TYPES_ROBUST:
                desired_config[OptionalConfigType.tile_size] = 16
            result = filter(configs_and_results, desired_config, ConfigType.repeats_insert)
            insert_tputs[index_type].append(float(result['insert']))
            result = filter(configs_and_results, desired_config, ConfigType.repeats_delete)
            delete_tputs[index_type].append(float(result['delete']))
            if index_type in INDEX_TYPES_ROBUST:
                desired_config[OptionalConfigType.lookup_concurrent] = 0
            result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
            lookup_tputs[index_type].append(float(result['lookup']))
    for index_type in scan_index_types:
        scan_tputs[index_type] = []
        for table_size in EXP_TABLE_SIZES:
            desired_config = {
                ConfigType.index_type: index_type,
                ConfigType.num_keys: table_size,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: DEFAULT_KEY_LENGHT,
                ConfigType.keylen_max: DEFAULT_KEY_LENGHT,
                ConfigType.num_scans: DEFAULT_NUM_KEYS,
                ConfigType.scan_count: DEFAULT_SCAN_COUNT
            }
            if index_type in INDEX_TYPES_ROBUST:
                desired_config[OptionalConfigType.tile_size] = 16
                desired_config[OptionalConfigType.lookup_concurrent] = 0
            result = filter(configs_and_results, desired_config, ConfigType.repeats_scan)
            scan_tputs[index_type].append(float(result['scan']))
    # plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 2), constrained_layout=True)
    plot_specs = [
        (0, axes[0], "Insert", insert_tputs, all_index_types),
        (1, axes[1], "Delete", delete_tputs, all_index_types),
        (2, axes[2], "Lookup", lookup_tputs, all_index_types),
        (3, axes[3], "Scan (count=10)", scan_tputs, scan_index_types),
    ]
    legend_handles = []
    legend_labels = []
    for i, ax, title, series_map, index_types in plot_specs:
        for index_type in index_types:
            line, = ax.plot(
                EXP_TABLE_SIZES,
                series_map[index_type],
                label=INDEX_LABELS[index_type],
                linewidth=2,
                markersize=6,
                **INDEX_STYLES[index_type],
            )
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(INDEX_LABELS[index_type])
        ax.set_title(title)
        ax.set_xlabel("Table Size")
        ax.set_xscale("log")
        ax.set_xticks(EXP_TABLE_SIZES)
        ax.xaxis.set_major_formatter(FuncFormatter(_table_size_label))
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    axes[0].set_ylabel("Throughput (Mop/s)")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, 1),
    )
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

def key_length_plots(configs_and_results, plot_file):
    no_prefix_lookup_tputs = {}
    prefix_lookup_tputs = {}
    key_lengths_xdata = [int(4 * l) for l in EXP_KEY_LENGTHS]
    all_index_types = INDEX_TYPES_ROBUST + INDEX_TYPES_CPU_BASELINE
    no_prefix_lookup_tputs[IndexType_gpu_masstree_no_suffix] = []
    no_prefix_lookup_tputs[IndexType_gpu_extendht_no_hashtag] = []
    prefix_lookup_tputs[IndexType_gpu_masstree_no_suffix] = []
    prefix_lookup_tputs[IndexType_gpu_extendht_no_hashtag] = []
    for index_type in all_index_types:
        no_prefix_lookup_tputs[index_type] = []
        prefix_lookup_tputs[index_type] = []
        for key_length in EXP_KEY_LENGTHS:
            desired_config = {
                ConfigType.index_type: index_type,
                ConfigType.num_keys: DEFAULT_NUM_KEYS,
                ConfigType.keylen_prefix: 0,
                ConfigType.keylen_min: key_length,
                ConfigType.keylen_max: key_length,
                ConfigType.num_lookups: DEFAULT_NUM_KEYS
            }
            if index_type in INDEX_TYPES_ROBUST:
                desired_config[OptionalConfigType.tile_size] = 16
                desired_config[OptionalConfigType.lookup_concurrent] = 0
            if index_type == IndexType.gpu_masstree:
                desired_config[OptionalConfigType.enable_suffix] = 1
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                no_prefix_lookup_tputs[index_type].append(float(result['lookup']))
                if GPU_MASSTREE_NOSUFFIX_TEST(0, key_length):
                    desired_config[OptionalConfigType.enable_suffix] = 0
                    result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                    no_prefix_lookup_tputs[IndexType_gpu_masstree_no_suffix].append(float(result['lookup']))
            elif index_type == IndexType.gpu_extendhashtable:
                desired_config[OptionalConfigType.hash_tag_level] = 2
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                no_prefix_lookup_tputs[index_type].append(float(result['lookup']))
                desired_config[OptionalConfigType.hash_tag_level] = 0
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                no_prefix_lookup_tputs[IndexType_gpu_extendht_no_hashtag].append(float(result['lookup']))
            else:
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                no_prefix_lookup_tputs[index_type].append(float(result['lookup']))
            desired_config = {
                ConfigType.index_type: index_type,
                ConfigType.num_keys: DEFAULT_NUM_KEYS,
                ConfigType.keylen_prefix: key_length - 1,
                ConfigType.keylen_min: key_length,
                ConfigType.keylen_max: key_length,
                ConfigType.num_lookups: DEFAULT_NUM_KEYS
            }
            if index_type in INDEX_TYPES_ROBUST:
                desired_config[OptionalConfigType.tile_size] = 16
                desired_config[OptionalConfigType.lookup_concurrent] = 0
            if index_type == IndexType.gpu_masstree:
                desired_config[OptionalConfigType.enable_suffix] = 1
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                prefix_lookup_tputs[index_type].append(float(result['lookup']))
                if GPU_MASSTREE_NOSUFFIX_TEST(key_length - 1, key_length):
                    desired_config[OptionalConfigType.enable_suffix] = 0
                    result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                    prefix_lookup_tputs[IndexType_gpu_masstree_no_suffix].append(float(result['lookup']))
            elif index_type == IndexType.gpu_extendhashtable:
                desired_config[OptionalConfigType.hash_tag_level] = 2
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                prefix_lookup_tputs[index_type].append(float(result['lookup']))
                desired_config[OptionalConfigType.hash_tag_level] = 0
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                prefix_lookup_tputs[IndexType_gpu_extendht_no_hashtag].append(float(result['lookup']))
            else:
                result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
                prefix_lookup_tputs[index_type].append(float(result['lookup']))
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.3), constrained_layout=True)
    legend_handles = []
    legend_labels = []
    
    for index_type in all_index_types + [IndexType_gpu_masstree_no_suffix, IndexType_gpu_extendht_no_hashtag]:
        ydata = _convert_mops_to_gops(no_prefix_lookup_tputs[index_type])
        xdata = key_lengths_xdata[0:len(ydata)]
        line, = axes[0].plot(
            xdata,
            ydata,
            label=INDEX_LABELS[index_type],
            linewidth=2,
            markersize=6,
            **INDEX_STYLES[index_type]
        )
        axes[0].set_title("Keys Without Common Prefix")
        axes[0].set_xlabel("Key Length")
        legend_handles.append(line)
        legend_labels.append(INDEX_LABELS[index_type])
    for index_type in all_index_types + [IndexType_gpu_masstree_no_suffix, IndexType_gpu_extendht_no_hashtag]:
        ydata = _convert_mops_to_gops(prefix_lookup_tputs[index_type])
        xdata = key_lengths_xdata[0:len(ydata)]
        axes[1].plot(
            xdata,
            ydata,
            label=INDEX_LABELS[index_type],
            linewidth=2,
            markersize=6,
            **INDEX_STYLES[index_type]
        )
        axes[1].set_title("Keys With Common Prefix")
        axes[1].set_xlabel("Key Length")
    axes[0].set_ylabel("Throughput (Gop/s)")
    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 1)
    )
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

def single_slice_plots(configs_and_results, plot_file):
    insert_tputs = []
    delete_tputs = []
    lookup_tputs = []
    scan_tputs = []
    all_index_types = INDEX_TYPES_ROBUST + INDEX_TYPES_GPU_BASELINE
    scan_index_types = INDEX_TYPES_ORDERED_IN(all_index_types)
    for index_type in all_index_types:
        desired_config = {
            ConfigType.index_type: index_type,
            ConfigType.num_keys: DEFAULT_NUM_KEYS,
            ConfigType.keylen_prefix: 0,
            ConfigType.keylen_min: 1,
            ConfigType.keylen_max: 1,
            ConfigType.delete_ratio: DEFAULT_DELETE_RATIO,
            ConfigType.num_lookups: DEFAULT_NUM_KEYS
        }
        if index_type in INDEX_TYPES_ROBUST:
            desired_config[OptionalConfigType.tile_size] = 16
        if index_type == IndexType.gpu_dycuckoo:
            desired_config[OptionalConfigType.use_lock] = 0
        result = filter(configs_and_results, desired_config, ConfigType.repeats_insert)
        insert_tputs.append(float(result['insert']))
        result = filter(configs_and_results, desired_config, ConfigType.repeats_delete)
        delete_tputs.append(float(result['delete']))
        if index_type in INDEX_TYPES_ROBUST + [IndexType.gpu_blink_tree]:
                desired_config[OptionalConfigType.lookup_concurrent] = 0
        result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
        lookup_tputs.append(float(result['lookup']))
        if index_type == IndexType.gpu_dycuckoo:
            desired_config[OptionalConfigType.use_lock] = 1
            result = filter(configs_and_results, desired_config, ConfigType.repeats_insert)
            insert_tputs.append(float(result['insert']))
            result = filter(configs_and_results, desired_config, ConfigType.repeats_delete)
            delete_tputs.append(float(result['delete']))
            result = filter(configs_and_results, desired_config, ConfigType.repeats_lookup)
            lookup_tputs.append(float(result['lookup']))
    for index_type in scan_index_types:
        desired_config = {
            ConfigType.index_type: index_type,
            ConfigType.num_keys: DEFAULT_NUM_KEYS,
            ConfigType.keylen_prefix: 0,
            ConfigType.keylen_min: 1,
            ConfigType.keylen_max: 1,
            ConfigType.num_scans: DEFAULT_NUM_KEYS,
            ConfigType.scan_count: DEFAULT_SCAN_COUNT
        }
        if index_type in INDEX_TYPES_ROBUST:
            desired_config[OptionalConfigType.tile_size] = 16
        if index_type in INDEX_TYPES_ROBUST + [IndexType.gpu_blink_tree]:
            desired_config[OptionalConfigType.lookup_concurrent] = 0
        result = filter(configs_and_results, desired_config, ConfigType.repeats_scan)
        scan_tputs.append(float(result['scan']))
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), constrained_layout=True)
    plot_specs = [
        (axes[0, 0], "Insert", insert_tputs, all_index_types + [IndexType_gpu_dycuckoo_with_lock]),
        (axes[0, 1], "Delete", delete_tputs, all_index_types + [IndexType_gpu_dycuckoo_with_lock]),
        (axes[1, 0], "Lookup", lookup_tputs, all_index_types + [IndexType_gpu_dycuckoo_with_lock]),
        (axes[1, 1], "Scan", scan_tputs, scan_index_types),
    ]
    legend_index_types = all_index_types + [IndexType_gpu_dycuckoo_with_lock]
    legend_handles = [
        mpatch.Patch(
            color=INDEX_STYLES[index_type]["color"],
            label=INDEX_LABELS[index_type],
        )
        for index_type in legend_index_types
    ]
    for ax, title, tputs, index_types in plot_specs:
        x = range(len(index_types))
        ax.bar(
            x,
            _convert_mops_to_gops(tputs),
            color=[INDEX_STYLES[index_type]["color"] for index_type in index_types],
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    axes[0, 0].set_ylabel("Throughput (Gop/s)")
    axes[1, 0].set_ylabel("Throughput (Gop/s)")
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 1),
    )
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)

def generate_plots(args, configs_and_results):
    table_size_plots(configs_and_results, Path(args.result_dir) / 'plot_tablesizes.pdf')
    key_length_plots(configs_and_results, Path(args.result_dir) / 'plot_keylengths.pdf')
    single_slice_plots(configs_and_results, Path(args.result_dir) / 'plot_singleslice.pdf')

if __name__ == "__main__":
    args = parse_args_for_plot()
    configs_and_results = read_configs_and_results(args)
    generate_plots(args, configs_and_results)
