from typing import List
from src.agents.hf_agent import HFAgent
from playpen.agents.base_agent import Agent
from playpen.agents.clembench_agent import ClembenchAgent
from playpen.clemgame.clemgame import GameBenchmark
from src.utils.logger import out_logger

def build_agent_list(game: GameBenchmark, agent_kwargs: str, gen_kwargs: str, eos_to_cull: str) -> List[Agent]:

    agent_args = dict(pair.split("=") for pair in agent_kwargs.split(","))
    gen_kwargs = dict(pair.split("=") for pair in gen_kwargs.split(","))

    agents = [HFAgent(eos_to_cull=eos_to_cull, gen_kwargs=gen_kwargs, **agent_args)]
    num_agents = 1 if game.is_single_player() else 2
    if len(agents) > num_agents:
        message = f"Too many agents for this game. The maximum number of player agents for this game is {max_num_agents}"
        out_logger.error(message)
        raise ValueError(message)
    elif len(agents) < num_agents:
        message = f"The number of agents was insufficient for playing the game. Creating {num_agents - len(agents)} agents with the last model specified in the arguments."
        out_logger.warning(message)
        agent = agents[-1]
        agents.extend([agent for _ in range(num_agents - len(agents))])
    return agents
