import time
import typing
import os, subprocess
import asyncio
import random
from collections import Counter

from aiofile import AIOFile, Writer
import bittensor as bt
import networkx as nx

from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from CliqueAI.scoring.clique_scoring import CliqueScoreCalculator
from CliqueAI.graph.model import LambdaGraph
from common.base.miner import BaseMinerNeuron


async def export_to_metis_async(graph, filename):
    """
    Exports a NetworkX graph to a METIS format file.

    Parameters:
    - graph: A NetworkX graph (unweighted).
    - filename: Name of the output METIS file.
    """
    # Ensure the graph is undirected
    if not graph.is_directed():
        # Number of vertices and edges
        num_vertices = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # Open the file to write
        async with AIOFile(filename, "w") as afp:
            writer = Writer(afp)

            # Write the header (number of vertices and edges)
            await writer(f"{num_vertices} {num_edges}\n")

            # Write the adjacency list for each vertex
            for node in graph.nodes():
                neighbors = graph.neighbors(node)
                # Convert node IDs to 1-based indexing (METIS format)
                neighbors_1_based = [str(neighbor + 1) for neighbor in neighbors]
                await writer(" ".join(neighbors_1_based) + "\n")
        bt.logging.info(f"Graph exported to {filename} in METIS format.")
    else:
        raise ValueError("Graph must be undirected.")


async def run_kamis_async(
    graph_file,
    kamis_executable="./online_mis",
    time_out=3,
    output_path=None,
    number_of_nodes=None,
    seed=None,
):
    """
    Run KaMIS on the exported graph file and return the MIS results.
    """
    if not os.path.exists(kamis_executable):
        raise FileNotFoundError(
            f"KaMIS executable not found at {kamis_executable}. Please build KaMIS first."
        )

    kamis_executable = "./lts"

    time_limit = time_out

    if output_path == None:
        output_path = "output.graph"

    if seed == None:
        seed = random.randint(1, 2**31 - 1)

    # Run KaMIS asynchronously
    process = await asyncio.create_subprocess_exec(
        kamis_executable,
        graph_file,
        "--output",
        str(output_path),
        "--console_log",
        "--time_limit",
        f"{time_limit}",
        "--disable_checks",
        "--adaptive_greedy",
        "--seed",
        f"{seed}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), time_out + 1)

        # Check if KaMIS ran successfully
        if process.returncode != 0:
            raise RuntimeError(f"KaMIS failed with error: {stderr.decode()}")

        # Parse the output to extract the MIS size and nodes
        mis_nodes = []
        index = 0
        async with AIOFile(str(output_path), mode="r") as afp:
            content = await afp.read()  # Read the entire file content
            lines = content.splitlines()
            for line in lines:
                if (
                    int(line.strip()) == 1
                ):  # Check if the vertex is in the MIS (binary 1)
                    mis_nodes.append(
                        index
                    )  # Append the vertex index (0-based indexing)
                index += 1  # Increment index for each line, regardless of the condition

        if os.path.exists(str(output_path)):
            os.remove(str(output_path))
        return mis_nodes

    except Exception as e:
        process.terminate()
        await process.wait()
        if os.path.exists(str(output_path)):
            os.remove(str(output_path))
        raise RuntimeError(f"KaMIS failed with error: Time Exceed => ", e)

    return []


async def run_kamis_with_timeout(
    graph_file,
    timeout,
    output_path=None,
    clique_score_calculator=None,
    seed=None,
    number_of_nodes=None,
):
    try:
        # Run the async task with a time limit
        result = await asyncio.wait_for(
            run_kamis_async(
                graph_file=graph_file,
                output_path=output_path,
                seed=seed,
                number_of_nodes=number_of_nodes,
            ),
            timeout,
        )

        if clique_score_calculator and clique_score_calculator.is_valid_maximum_clique(
            list(result)
        ):
            return result
        elif clique_score_calculator:
            raise RuntimeError(f"Kamis did not found the correct Clique")
    except Exception as e:
        print(f"Kamis has occuerd with Error: {e}")
        return []  # Return None or handle timeout as needed


async def compute_maximum_clique(
    number_of_nodes, graph, clique_score_calculator=None, uuid=""
):
    try:
        cgraph = nx.complement(graph)

        graph_file = f"graph-{uuid}.graph"

        writing_time = time.time()
        await export_to_metis_async(cgraph, graph_file)
        bt.logging.info(
            f"Writing graph file completed in {time.time() - writing_time} seconds"
        )

        graph_time = time.time()
        timeout = 7
        num_tasks = 15  # Number of tasks to run
        mis_results = await asyncio.gather(
            *[
                run_kamis_with_timeout(
                    graph_file=graph_file,
                    timeout=timeout,
                    output_path=f"output-{uuid}-{i}.graph",
                    clique_score_calculator=clique_score_calculator,
                    number_of_nodes=number_of_nodes,
                )
                for i in range(1, num_tasks + 1)
            ]
        )

        bt.logging.info(f"Running Kamis finished in {time.time() - graph_time} seconds")

        max_length = max(len(arr) for arr in mis_results)
        unique_arrays = list(map(list, set(map(tuple, mis_results))))

        # Get all arrays with the maximum length
        max_arrays = [arr for arr in unique_arrays if len(arr) == max_length]

        lengths = [len(mis) for mis in unique_arrays]

        length_counts = Counter(lengths)
        most_frequent_length = length_counts.most_common(1)[0][0]

        filtered_arrays = [
            subarray
            for subarray in unique_arrays
            if len(subarray) == most_frequent_length
        ]

        mis = random.choice(filtered_arrays + max_arrays)

        if os.path.exists(graph_file):
            os.remove(graph_file)
        return mis

    except Exception as e:
        bt.logging.info(f"Error in validating maximum clique: {e}")
        approxy_time = time.time()
        approxy = nx.approximation.max_clique(graph)
        bt.logging.info(
            f"Approxy Solution has made in {time.time() - approxy_time} seconds"
        )
        return approxy


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        number_of_nodes = synapse.number_of_nodes
        uuid = synapse.uuid

        bt.logging.info(f"Number of nodes in the {uuid} graph: {number_of_nodes}")

        adjacency_list = synapse.adjacency_list
        # bt.logging.info(f"Adjacency list received: {adjacency_list}")

        making_time = time.time()
        dict_of_lists = {i: adjacency_list[i] for i in range(number_of_nodes)}

        graph = nx.from_dict_of_lists(dict_of_lists)
        bt.logging.info(
            f"NetworkX graph has made in {time.time() - making_time} seconds"
        )

        lamba_graph = LambdaGraph(
            uuid="",
            label="",
            number_of_nodes=number_of_nodes,
            adjacency_list=adjacency_list,
        )

        clique_score_calculator = CliqueScoreCalculator(
            graph=lamba_graph, difficulty=0.2, responses=[[]]
        )

        try:
            mis = await asyncio.wait_for(
                compute_maximum_clique(
                    number_of_nodes=number_of_nodes,
                    graph=graph,
                    clique_score_calculator=clique_score_calculator,
                    uuid=uuid,
                ),
                timeout=25,
            )
            maximum_clique = mis

            if len(mis) > 0:
                bt.logging.info(
                    f"=========================================================="
                )
                bt.logging.info(
                    f"Maximum clique found: {maximum_clique} with size {len(mis)}"
                )
                bt.logging.info(
                    f"=========================================================="
                )

            else:
                raise RuntimeError(f"Error in running KAMIS")

        except Exception as e:
            bt.logging.info(f"Error in validating maximum clique: {e}")
            maximum_clique = []

        synapse.adjacency_list = [
            []
        ]  # Clear up the adjacency list to reduce response size.
        synapse.maximum_clique = maximum_clique
        bt.logging.info(f"Sending response: {maximum_clique}")
        return synapse

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Miner has started running.")
        while True:
            if miner.should_exit:
                bt.logging.info("Miner is exiting.")
                break
            time.sleep(1)
