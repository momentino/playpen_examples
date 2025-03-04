import argparse

from pathlib import Path
from playpen.clemgame import benchmark
from playpen.clemgame.benchmark import load_benchmark
from src.agents import build_agent_list

PROJECT_ROOT = script_path = Path(__file__).resolve().parent.parent


def main(args: argparse.Namespace):
    if args.command_name == "ls":
        benchmark.list_games()
    if args.command_name == "run":
        game = load_benchmark(args.game, instances_name=args.instances_name)
        agents = build_agent_list(game=game, agent_kwargs=args.agent_kwargs, gen_kwargs=args.gen_kwargs, eos_to_cull=args.eos_to_cull)
        benchmark.run_playpen(args.game,
                      agents=agents,
                      experiment_name=args.experiment_name,
                      instances_name=args.instances_name,
                      results_dir=args.results_dir)
        """benchmark.run(args.game,
                      model_specs=read_model_specs(args.models),
                      gen_args=read_gen_args(args),
                      experiment_name=args.experiment_name,
                      instances_name=args.instances_name,
                      results_dir=args.results_dir)"""
    if args.command_name == "score":
        benchmark.score(args.game, experiment_name=args.experiment_name, results_dir=args.results_dir)
    if args.command_name == "transcribe":
        benchmark.transcripts(args.game, experiment_name=args.experiment_name, results_dir=args.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="command_name")
    sub_parsers.add_parser("ls")

    run_parser = sub_parsers.add_parser("run", formatter_class=argparse.RawTextHelpFormatter)
    run_parser.add_argument("-e", "--experiment_name", type=str,
                            help="Optional argument to only run a specific experiment")
    run_parser.add_argument("-g", "--game", type=str,
                            required=True, help="A specific game name (see ls).")
    run_parser.add_argument("-a", "--agent_kwargs", type=str, required=True,
                            help="Argument to specify the arguments to initialize agents (comma-separated arguments).")
    run_parser.add_argument("-k", "--gen_kwargs", type=str, default="temperature=0.0,max_new_tokens=100",
                            help="Argument to specify the (comma-separated) generation arguments for agents.")
    run_parser.add_argument("-c", "--eos_to_cull", type=str,
                            help="Argument to specify the eos token. e.g. <|eot_id|>")
    run_parser.add_argument("-i", "--instances_name", type=str, default="instances",
                            help="The instances file name (.json suffix will be added automatically.")
    run_parser.add_argument("-r", "--results_dir", type=str, default=str(PROJECT_ROOT / "results"),
                            help="A relative or absolute path to the results root directory. "
                                 "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                 "When not specified, then the results will be located in './results'")

    score_parser = sub_parsers.add_parser("score")
    score_parser.add_argument("-e", "--experiment_name", type=str,
                              help="Optional argument to only run a specific experiment")
    score_parser.add_argument("-g", "--game", type=str,
                              help="A specific game name (see ls).", default="all")
    score_parser.add_argument("-r", "--results_dir", type=str, default=str(PROJECT_ROOT / "results"),
                              help="A relative or absolute path to the results root directory. "
                                   "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                   "When not specified, then the results will be located in './results'")

    transcribe_parser = sub_parsers.add_parser("transcribe")
    transcribe_parser.add_argument("-e", "--experiment_name", type=str,
                                   help="Optional argument to only run a specific experiment")
    transcribe_parser.add_argument("-g", "--game", type=str,
                                   help="A specific game name (see ls).", default="all")
    transcribe_parser.add_argument("-r", "--results_dir", type=str, default=str(PROJECT_ROOT / "results"),
                                   help="A relative or absolute path to the results root directory. "
                                        "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                        "When not specified, then the results will be located in './results'")

    main(parser.parse_args())
