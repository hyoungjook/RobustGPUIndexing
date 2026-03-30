from evaluate import *

def generate_plots(configs_and_results):
    pass

if __name__ == "__main__":
    args = parse_args_for_plot()
    configs_and_results = read_configs_and_results(args)
    generate_plots(configs_and_results)
