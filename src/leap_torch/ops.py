from .util import check_module

from leap_ec.ops import iteriter_op, UniformCrossover as _UniformCrossover
from leap_ec.util import wrap_curry

import torch
from torch import nn
import numpy as np
from toolz import curry
from typing import Iterator
from itertools import islice


def torch_parameters_mutate_gaussian(
            module: nn.Module, p_mutate: float, std: float
        ):
    """
    Mutates each parameter of the provided pytorch module at a probability
    p_mutate per element, with a gaussian shift of standard deviation std.
    This operator is applied in place, but returns the module for convenicence.

    :param module: the pytorch module to be mutated.
    :param p_mutate: the probability of mutation per element.
    :param std: the standard deviation of the gaussian shift applied to the
        mutated elements.
    :return: the mutated module.
    """
    with torch.no_grad():
        p_mutate_tensor = torch.tensor(p_mutate)
        std_tensor = torch.tensor(std)
        
        for param in module.parameters():
            # Create a mask of where the updates will occur
            mask = torch.rand(param.size()) < p_mutate_tensor
            # Create a gaussian shift with the provided std
            gauss = torch.normal(0, std_tensor, param.size())

            # Mask out the gaussian shift for applying to the array
            shift = torch.where(mask, gauss, 0)
            param += shift
    
    return module


def _compute_expected_probability(module: nn.Module, expected_num_mutations: float):
    """
    Computes the probability of mutation to have the expected number of mutations
    when mutating each parameter element inividually.

    :param module: the module to which the mutation will be applied.
    :param expected_num_mutations: the expected number of mutations when mutating
        the provided module.
    :return: the probability which has an expected value for number of mutations
        of expected_num_mutations when used to mutate module.
    """
    
    sum_size = 0.0
    for param in module.parameters():
        sum_size += np.prod(param.size(), dtype=np.float64)
    return expected_num_mutations / sum_size

@wrap_curry
@iteriter_op
def mutate_gaussian(
            next_individual: Iterator,
            std: float, *, expected_num_mutations: int=None, p_mutate: float=None,
        ):
    """
    Applies a gaussian mutation to the pytorch module genomes of individuals
    supplied to this operator. One of an expected number of mutations,
    expected_num_mutations, or the probability of mutation, p_mutate,
    must be supplied. The mutation is a gaussian shift with standard
    deviation equal to std.

    :param next_individual: the individuals to be mutated.
    :param std: the standard deviation of the mutation.
    :param expected_num_mutations: the expected number of mutations to be
        applied to each individual.
    :param p_mutate: the probability of each element of the module's parameters
        being mutated.
    :yield: an individual which has been mutated.
    """
    assert expected_num_mutations is not None or p_mutate is not None,\
        "One of expected_num_mutations or p_mutate must be specified"
    assert expected_num_mutations is None or p_mutate is None,\
        "Only one of expected_num_mutations or p_mutate should be specified"
    
    while True:
        individual = next(next_individual)
        check_module(individual.genome)
        
        if expected_num_mutations is not None:
            p_mutate = _compute_expected_probability(individual.genome, expected_num_mutations)
        
        individual.genome = torch_parameters_mutate_gaussian(
                individual.genome, p_mutate, std
            )
        # individual.fitness = None
        individual.evaluation = None
        
        yield individual


def torch_parameters_mutate_bounded_polynomial(module, p_mutate: float, eta: float, low: float, high: float):
    """
    Mutates each parameter of the provided pytorch module at a probability
    p_mutate per element, with a bounded polynomial update as specified in the NSGA-II paper.
    This operator is applied in place, but returns the module for convenicence.

    :param module: the pytorch module to be mutated.
    :param p_mutate: the probability of mutation per element.
    :param eta: crowding degree of the mutation, higher values result in smaller variation from the parent.
    :param low: the lower bound of the parameter values.
    :param high: the upper bound of the parameter values.
    :return: the mutated module.
    """
    with torch.no_grad():
        p_mutate_tensor = torch.tensor(p_mutate)
        eta_p1 = torch.tensor(eta + 1.0)
        eta_exponent = eta_p1 ** -1
        diff = torch.tensor(high - low)
        
        for param in module.parameters():
            # Note: this math has been heavily simplified from that of DEAP
            # https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py#L51

            # If the element will be mutated, selects a value from -1 to 1
            # At 0 the value isn't changed so we use that as a mask
            rand = torch.where(
                    torch.rand(param.size()) < p_mutate_tensor,
                    torch.rand(param.size()) * 2 - 1,
                    0
                )
            
            xy = torch.where(rand < 0, high - param, param - low) / diff

            v = 1 + torch.abs(rand) * (xy ** eta_p1 - 1)
            # torch.sign returns 0 on 0, but it wouldn't change then anyways
            delta = torch.sign(rand) * (1 - v ** eta_exponent) * diff
            
            # Clip the updated tensor and assign back to the parameter
            torch.clip(param + delta, low, high, out=param)
    
    return module


@wrap_curry
@iteriter_op
def mutate_bounded_polynomial(
            next_individual: Iterator,
            eta: float, low: float, high: float, *, expected_num_mutations: int=None, p_mutate: float=None,
        ):
    """
    Applies a bounded polynomial mutation to the pytorch module genomes of individuals
    supplied to this operator. One of an expected number of mutations,
    expected_num_mutations, or the probability of mutation, p_mutate,
    must be supplied. The mutation is as specified in the NSGA-II paper.

    :param next_individual: the individuals to be mutated.
    :param eta: crowding degree of the mutation, higher values result in smaller variation from the parent.
    :param low: the lower bound of the parameter values.
    :param high: the upper bound of the parameter values.
    :param expected_num_mutations: the expected number of mutations to be
        applied to each individual.
    :param p_mutate: the probability of each element of the module's parameters
        being mutated.
    :yield: an individual which has been mutated.
    """
    assert expected_num_mutations is not None or p_mutate is not None,\
        "One of expected_num_mutations or p_mutate must be specified"
    assert expected_num_mutations is None or p_mutate is None,\
        "Only one of expected_num_mutations or p_mutate should be specified"
    
    while True:
        individual = next(next_individual)
        check_module(individual.genome)
        
        if expected_num_mutations is not None:
            p_mutate = _compute_expected_probability(individual.genome, expected_num_mutations)
        
        individual.genome = torch_parameters_mutate_bounded_polynomial(
                individual.genome, p_mutate, eta, low, high
            )
        # Probably should wipe this
        # individual.fitness = None
        
        yield individual


def torch_parameters_uniform_crossover(
            module_a: nn.Module, module_b: nn.Module, p_swap: float
        ):
    """
    Uniformly crosses over parameters between two pytorch modules. Each
    element has p_swap probability of coming from the opposite module.
    This operation is performed in place, but the modules are returned
    for convenience.

    :param module_a: the first module to be crossed over.
    :param module_b: the second module to be crossed over.
    :param p_swap: the probability of an element coming from the opposite
        parent.
    :return: two modules whose parameters have been crossed over.
    """
    
    with torch.no_grad():
        assert len(list(module_a.parameters())) == len(list(module_b.parameters())),\
            "Modules must have the same number of parameters to perform crossover"
        
        for param_a, param_b in zip(module_a.parameters(), module_b.parameters()):
            assert param_a.size() == param_b.size(),\
                "Parameters must have the same shape to perform crossover"
            
            # Creates a mask of which values to swap, then moves it to the relevant devices
            swap_mask = torch.rand(param_a.size()) < p_swap

            # param_a must be cloned since we need it's original value if on the same device
            a_on_b = param_a.to(param_b.device).clone()
            b_on_a = param_b.to(param_a.device)

            torch.where(swap_mask, b_on_a, param_a, out=param_a)
            torch.where(swap_mask, a_on_b, param_b, out=param_b)
    
    return module_a, module_b


class UniformCrossover(_UniformCrossover):
    """Parameterized uniform crossover iterates through two parents' genomes,
    provided they consist of pytorch modules, and swaps each of their genes with the given probability.

    In a classic paper, De Jong and Spears showed that this operator works
    particularly well when the swap probability `p_swap` is set to about 0.2.  LEAP
    thus uses this value as its default.

        De Jong, Kenneth A., and W. Spears. "On the virtues of parameterized uniform crossover."
        *Proceedings of the 4th international conference on genetic algorithms.* Morgan Kaufmann Publishers, 1991.

    :param p_swap: how likely are we to swap each pair of genes when crossover
        is performed
    :param float p_xover: the probability that crossover is performed in the
        first place
    :param bool persist_children: whether unyielded children should persist between calls.
        This is useful for `leap_ec.distrib.asynchronous.steady_state`, where the pipeline
        may only produce one individual at a time.
    :return: a pipeline operator that returns two recombined individuals (with probability
        p_xover), or two unmodified individuals (with probability 1 - p_xover)
    """

    def recombine(self, parent_a, parent_b):
        parent_a.genome, parent_b.genome = torch_parameters_uniform_crossover(
                parent_a.genome, parent_b.genome, self.p_swap
            )
        return parent_a, parent_b
