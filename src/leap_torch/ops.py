from .util import check_module

from leap_ec.ops import iteriter_op

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

@curry
@iteriter_op
def mutate_guassian(
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
        individual.fitness = None
        
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

def uniform_crossover(p_swap: float=0.2, p_xover: float=1.0):
    """
    Creates a function that will uniformly cross over the genomes
    of individuals supplied to it. Cross over occurs at a probability
    equal to p_xover, with a probability of elements coming from the
    opposing parent of p_swap.

    :param p_swap: the probability that an element for a crossed over
        genome comes from the opposing parent.
    :param p_xover: the probability that two parents will be crossed
        over as opposed to forwarded on.
    :return: a function that behaves as a pipeline operator which will
        perform crossover with the given parameters.
    """
    
    second_child = None
    
    @iteriter_op
    def _do_crossover(next_individual: Iterator):
        """
        A pipeline operator which uniformly crosses over the genomes
        of individuals supplied to it.

        :param next_individual: the individuals to be crossed over.
        :yield: individuals which may have been crossed over.
        """
        nonlocal second_child
        
        if second_child is not None:
            ret_child, second_child = second_child, None
            yield ret_child
                
        while True:
            first_child, second_child = islice(next_individual, 2)
            check_module(first_child.genome)
            check_module(second_child.genome)

            if np.random.random_sample() < p_xover:
                first_child.genome, second_child.genome = torch_parameters_uniform_crossover(
                        first_child.genome, second_child.genome, p_swap
                    )
                first_child.fitness = second_child.fitness = None
            
            yield first_child
            ret_child, second_child = second_child, None
            yield ret_child
    
    return _do_crossover