import argparse

from playpen.clemgame import benchmark
from playpen.clemgame.benchmark import load_benchmark
from src.agents import build_agent_list

"""
    Use good old argparse to run the commands.

    To list available games: 
    $> python3 scripts/cli.py ls

    To run a specific game with a single player:
    $> python3 scripts/cli.py run -g privateshared -m mock

    To run a specific game with a two players:
    $> python3 scripts/cli.py run -g taboo -m mock mock

    If the game supports model expansion (using the single specified model for all players):
    $> python3 scripts/cli.py run -g taboo -m mock

    To score all games:
    $> python3 scripts/cli.py score

    To score a specific game:
    $> python3 scripts/cli.py score -g privateshared

    To score all games:
    $> python3 scripts/cli.py transcribe

    To score a specific game:
    $> python3 scripts/cli.py transcribe -g privateshared
"""


def main(args: argparse.Namespace):
    if args.command_name == "ls":
        benchmark.list_games()
    if args.command_name == "run":
        game = load_benchmark(args.game, instances_name=args.instances_name)
        agents = build_agent_list(game=game, agent_kwargs=args.agent_kwargs, gen_kwargs=args.gen_kwargs)
        print(agents)
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
    run_parser.add_argument("-i", "--instances_name", type=str, default="instances",
                            help="The instances file name (.json suffix will be added automatically.")
    run_parser.add_argument("-r", "--results_dir", type=str, default="results",
                            help="A relative or absolute path to the results root directory. "
                                 "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                 "When not specified, then the results will be located in './results'")

    score_parser = sub_parsers.add_parser("score")
    score_parser.add_argument("-e", "--experiment_name", type=str,
                              help="Optional argument to only run a specific experiment")
    score_parser.add_argument("-g", "--game", type=str,
                              help="A specific game name (see ls).", default="all")
    score_parser.add_argument("-r", "--results_dir", type=str, default="results",
                              help="A relative or absolute path to the results root directory. "
                                   "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                   "When not specified, then the results will be located in './results'")

    transcribe_parser = sub_parsers.add_parser("transcribe")
    transcribe_parser.add_argument("-e", "--experiment_name", type=str,
                                   help="Optional argument to only run a specific experiment")
    transcribe_parser.add_argument("-g", "--game", type=str,
                                   help="A specific game name (see ls).", default="all")
    transcribe_parser.add_argument("-r", "--results_dir", type=str, default="results",
                                   help="A relative or absolute path to the results root directory. "
                                        "For example '-r results/v1.5/de‘ or '-r /absolute/path/for/results'. "
                                        "When not specified, then the results will be located in './results'")

    main(parser.parse_args())
