from typing import List
from agents.hf_agent import HFAgent
from playpen.agents.base_agent import Agent
from playpen.clemgame.clemgame import GameBenchmark

def build_agent_list(game: GameBenchmark, agent_kwargs: str, gen_kwargs: str) -> List[Agent]:

    agent_args = dict(pair.split("=") for pair in agent_kwargs.split(","))
    gen_kwargs = dict(pair.split("=") for pair in gen_kwargs.split(","))

    agents = [HFAgent(gen_kwargs=gen_kwargs, **agent_args)]

    min_num_agents = 1
    max_num_agents = 1 if game.is_single_player() else 2
    if len(agents) > max_num_agents:
        message = f"Too many agents for this game. The maximum number of player agents for this game is {max_num_agents}"
        stdout_logger.error(message)
        raise ValueError(message)
    elif len(agents) < min_num_agents:
        message = f"The number of agents was insufficient for playing the game. Creating {min_num_agents - len(agents)} agents with the last model specified in the arguments."
        stdout_logger.warning(message)
        agent = agents[-1]
        agents.extend([agent for _ in range(min_num_agents - len(agents))])
    return agents





def create_agents(agent_args_list: List[str], agents_root: str, game: GameBenchmark) -> List[Agent]:
    agents = [init_agent_from_args(agent_args, agents_root) for agent_args in agent_args_list]
    # TODO: when we have the same model playing multiple agents, we could have one single instance in the memory rather than two, and have separate histories
    min_num_agents = 1
    max_num_agents = 1 if game.is_single_player() else 2
    if len(agents) > max_num_agents:
        message = f"Too many agents for this game. The maximum number of player agents for this game is {max_num_agents}"
        stdout_logger.error(message)
        raise ValueError(message)
    elif len(agents) < min_num_agents:
        message = f"The number of agents was insufficient for playing the game. Creating {min_num_agents - len(agents)} agents with the last model specified in the arguments."
        stdout_logger.warning(message)
        agent = agents[-1]
        agents.extend([agent for _ in range(min_num_agents - len(agents))])
    return agents