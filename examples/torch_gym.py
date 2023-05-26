from leap_torch.ops import mutate_guassian, uniform_crossover
from leap_torch.initializers import create_instance
from leap_torch.decoders import NumpyDecoder

from leap_ec import ops
from leap_ec.probe import AttributesCSVProbe, FitnessPlotProbe
from leap_ec.distrib.synchronous import eval_pool
from leap_ec.representation import Representation
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep.problems import EnvironmentProblem
from leap_ec.executable_rep.executable import ArgmaxExecutable, WrapperDecoder

import argparse
from distributed import Client
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn

class GymNetwork(nn.Module):
    
    # Constructs a stack of densely connected hidden layers with sigmoid activation
    # Initializes parameters with values uniformly drawn from -1 to 1 in order to
    # mirror the leap neural network cartpole example.
    
    def __init__(self, num_inputs, num_outputs, hidden_layers, activation=nn.Sigmoid):
        super().__init__()
        latent_sizes = [num_inputs, *hidden_layers, num_outputs]
        
        self.linear_layers = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(latent_sizes[:-1], latent_sizes[1:])
            ])
        
        for param in self.linear_layers.parameters():
            nn.init.uniform_(param, -1, 1)
        
        self.activation = activation()

    def forward(self, obs):
        latent = obs
        for layer in self.linear_layers:
            latent = self.activation(layer(latent))

        return latent
        

def main(
            runs_per_fitness_eval=5, simulation_steps=500,
            generations=250, pop_size=30,
            hidden_nodes=10, hidden_layers=1,
            mutate_std=0.05, expected_num_mutations=1,
            env_name="CartPole-v1",
            n_workers=None, data_output_fp="data.csv", model_output_fp="model.pt"
        ):
    # While you can override the device, its not very effective
    # GPUs are better for running many models at once, not necessarily one very fast
    # Further, they do batch multiplexing, not thread multiplexing
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    env = gym.make(env_name)
    decoder = WrapperDecoder(
            wrapped_decoder=NumpyDecoder(device=device), decorator=ArgmaxExecutable
        )

    def save_probe(pop):
        best_ind = max(pop)
        torch.save(best_ind.genome, model_output_fp)
        return pop
    
    with open(data_output_fp, "w") as genomes_file:
        att_probe = AttributesCSVProbe(
                stream=genomes_file,
                best_only=True,
                do_fitness=True,
                do_genome=False
            )
        
        plot_probe = FitnessPlotProbe(
                ylim=(0, 1), xlim=(0, 1),
                modulo=1, ax=plt.gca()
            )
        
        eval_pool_pipeline = [
                ops.evaluate,
                ops.pool(size=pop_size)
            ]
        
        if n_workers is not None:
            client = Client(n_workers=n_workers)
            eval_pool_pipeline = [eval_pool(client=client, size=pop_size)]
        
        generational_ea(
                max_generations=generations, pop_size=pop_size,
                
                problem=EnvironmentProblem(
                    runs_per_fitness_eval, simulation_steps,
                    environment=env, fitness_type="reward", gui=False
                ),
                
                representation=Representation(
                    initialize=create_instance(
                        GymNetwork,
                        env.observation_space.shape[0], env.action_space.n,
                        (hidden_nodes,) * hidden_layers
                    ), decoder=decoder
                ),
                
                pipeline=[
                    ops.tournament_selection,
                    ops.clone,
                    mutate_guassian(std=mutate_std, expected_num_mutations=expected_num_mutations),
                    uniform_crossover(),
                    *eval_pool_pipeline,
                    save_probe,
                    att_probe,
                    plot_probe,
                ]
            )
        
        if n_workers is not None:
            client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-r", "--runs_per_fitness_eval", default=5, type=int,
            help="The number of runs to perform per fitness evaluation."
        )
    parser.add_argument(
            "-s", "--simulation_steps", default=500, type=int,
            help="The number of steps the environment can take before stopping."
        )
    parser.add_argument(
            "-g", "--generations", default=250, type=int,
            help="The number of generations the evolutionary algorithm is ran for."
        )
    parser.add_argument(
            "-p", "--pop_size", default=30, type=int,
            help="The size of the population to be evolved."
        )
    parser.add_argument(
            "-e", "--env_name", default="CartPole-v1", type=str,
            help="The name of the gym environment to be evaluated."
        )
    parser.add_argument(
            "--hidden_nodes", default=10, type=int,
            help="The number of hidden nodes per hidden layer in the model."
        )
    parser.add_argument(
            "--hidden_layers", default=1, type=int,
            help="The number of hidden layers in the model."
        )
    parser.add_argument(
            "--mutate_std", default=0.05, type=float,
            help="The standard deviation of the mutation applied to the genomes."
        )
    parser.add_argument(
            "--expected_num_mutations", default=1, type=int,
            help="The expected number of mutations to occur within the genome."
        )
    parser.add_argument(
            "--n_workers", default=None, type=int,
            help="If set, uses distributed evaluation of the poplation with the given number of workers."
        )
    parser.add_argument(
            "-d", "--data_output_fp", default="data.csv", type=str,
            help="The destination path of the data csv file."
        )
    parser.add_argument(
            "-m", "--model_output_fp", default="model.pt", type=str,
            help="The destination path of the pytorch model weights."
        )
    
    args = parser.parse_args()
    main(**args.__dict__)