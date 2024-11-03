from __future__ import annotations

from enum import IntEnum
from typing import Callable, List, Optional, Tuple

import numpy as np

# If this is your first time seeing @property, please see this example and feel free
# to google more if you need
#
# Class Example:
#
#   def __init__(self):
#       self.count = 0
#       self.word = "hello world"
#
#   @property
#   def myvalue(self) -> str:
#       return self.word + "  " + str(self.count)
#
#   @myvalue.setter(self):
#   def myvalue(self, new_word: str) -> None
#       self.count += 1
#       self.word = new_word
#
# # Notice how no functions calls are made with () and it's treated like an attribute
# example = Example
# print(example.value)  # "hello world 0"
# example.myvalue = "hello mars"
# print(example.value)  # "hello mars 1"


class Recombination(IntEnum):
    """Enum defining the recombination strategy choice"""

    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    INTERMEDIATE = 1  # intermediate recombination


class Mutation(IntEnum):
    """Enum defining the mutation strategy choice"""

    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation


class ParentSelection(IntEnum):
    """Enum defining the parent selection choice"""

    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


class Member:
    """Class to simplify member handling."""

    def __init__(
        self,
        initial_x: np.ndarray,
        target_function: Callable,
        bounds: Tuple[float, float],
        mutation: Mutation,
        recombination: Recombination,
        sigma: Optional[float] = None,
        recom_prob: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        initial_x : np.ndarray
            Initial coordinate of the member

        target_function : Callable
            The target function that determines the fitness value

        bounds : Tuple[float, float]
            Allowed bounds. For simplicities sake we assume that all elements in
            initial_x have the same bounds:
            * bounds[0] lower bound
            * bounds[1] upper bound

        mutation : Mutation
            Hyperparameter that determines which mutation type use

        recombination : Recombination
            Hyperparameter that determines which recombination type to use

        sigma : Optional[float] = None
            Optional hyperparameter that is only active if mutation is gaussian

        recom_prob : Optional[float]
            Optional hyperparameter that is only active if recombination is uniform
        """
        # astype is crucial here. Otherwise numpy might cast everything to int
        self._x = initial_x.astype(float)
        self._f = target_function
        self._bounds = bounds
        self._mutation = mutation
        self._recombination = recombination
        self._sigma = sigma
        self._recom_prob = recom_prob

        # indicates how many offspring were generated from this member
        self._age = 0
        self._x_changed = True
        self._fit = 0.0

    @property
    def fitness(self) -> float:
        """Retrieve the fitness, recalculating it if x changed"""
        if self._x_changed:
            self._x_changed = False
            self._fit = self._f(self._x)

        return self._fit

    @property
    def x_coordinate(self) -> np.ndarray:
        """The current x coordinate"""
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value: np.ndarray) -> None:
        """Set the new x coordinate"""
        lower, upper = self._bounds
        assert np.all((lower <= value) & (value <= upper)), f"Member out of bounds, {value}"
        self._x_changed = True
        self._x = value

    def mutate(self) -> Member:
        """Mutation which creates a new offspring

        As a side effect, it will increment the age of this Member.

        Returns
        -------
        Member
            The mutated Member created from this member
        """
        new_x = self.x_coordinate.copy()

        # print(f"new point before mutation:\n {new_x}")
        
        # TODO modify new_x either through uniform or gaussian mutation
        if self._mutation == Mutation.UNIFORM:
            new_x = self.uniform_mutation(new_x)

        elif self._mutation == Mutation.GAUSSIAN:
            if self._sigma is None:
                raise ValueError("Sigma has to be set when gaussian mutation is used")
            new_x = self.gaussian_mutation(new_x)

        elif self._mutation == Mutation.NONE:
            new_x = new_x

        else:
            # We won't consider any other mutation types
            raise RuntimeError(f"Unknown mutation {self._mutation}")

        # print(f"new point after mutation: \n {new_x}")
        child = Member(
            new_x,
            self._f,
            self._bounds,
            self._mutation,
            self._recombination,
            self._sigma,
            self._recom_prob,
        )
        self._age += 1
        return child

    def recombine(self, partner: Member) -> Member:
        """Recombination of this member with a partner

        Parameters
        ----------
        partner : Member
            The other Member to combine with

        Returns
        -------
        Member
            A new Member based on the combination of this one and the partner
        """
        # TODO
        if self._recombination == Recombination.INTERMEDIATE:
            raise NotImplementedError

        # TODO
        elif self._recombination == Recombination.UNIFORM:
            if self._recom_prob is None:
                raise ValueError("For Uniform recombination, the recombination probability must be given")
            raise NotImplementedError

        elif self._recombination == Recombination.NONE:
            # copy is important here to not only get a reference
            new_x = self.x_coordinate.copy()

        else:
            raise NotImplementedError

        print(f"new point after recombination:\n {new_x}")
        child = Member(
            new_x,
            self._f,
            self._bounds,
            self._mutation,
            self._recombination,
            self._sigma,
            self._recom_prob,
        )
        self._age += 1
        return child

    def uniform_mutation(self, x: np.ndarray) -> np.ndarray:
        geneToMutateIndex = np.random.randint(0, x.size - 1)
        x[geneToMutateIndex] = np.random.uniform(self._bounds[0], self._bounds[1])

        return x

    def gaussian_mutation(self, x: np.ndarray) -> np.ndarray:
        mutated_x = x + np.random.normal(loc=0, scale=self._sigma, size=x.size)
        return np.clip(mutated_x, self._bounds[0], self._bounds[1])

    def __str__(self) -> str:
        """Makes the class easily printable"""
        return f"Population member: Age={self._age}, x={self.x_coordinate}, f(x)={self.fitness}"

    def __repr__(self) -> str:
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + "\n"


class EA:
    """A class implementing evolutionary algorithm strategies"""

    def __init__(
        self,
        target_func: Callable,
        population_size: int = 10,
        problem_dim: int = 2,
        problem_bounds: Tuple[float, float] = (-30, 30),
        mutation_type: Mutation = Mutation.UNIFORM,
        recombination_type: Recombination = Recombination.INTERMEDIATE,
        selection_type: ParentSelection = ParentSelection.NEUTRAL,
        sigma: float = 1.0,
        recom_proba: float = 0.5,
        total_number_of_function_evaluations: int = 200,
        children_per_step: int = 5,
        fraction_mutation: float = 0.5,
    ):
        """
        Parameters
        ----------
        target_func : Callable
            callable target function we optimize

        population_size: int = 10
            The total population size to use

        problem_dim: int = 2
            The dimension of each member's x

        problem_bounds: Tuple[float, float] = (-30, 30)
            Used to make sure population members are valid

        mutation_type: Mutation = Mutation.UNIFORM
            Hyperparameter to set mutation strategy

        recombination_type: Recombination = Recombination.INTERMEDIATE
            Hyperparameter to set recombination strategy

        selection_type: ParentSelection = ParentSelection.NEUTRAL
            Hyperparameter to set selection strategy

        sigma: float = 1.0
            Conditional hyperparameter dependent on mutation_type GAUSSIAN, defines
            the sigma for the guassian distribution

        recom_proba: float = 0.5
            Conditional hyperparameter dependent on recombination_type UNIFORM.

        total_number_of_function_evaluations: int = 200
            Maximum allowed function evaluations

        children_per_step: int = 5
            How many children to produce per step

        fraction_mutation: float = 0.5
            Balance between sexual and asexual reproduction
        """
        assert 0 <= fraction_mutation <= 1
        assert 0 < children_per_step
        assert 0 < total_number_of_function_evaluations
        assert 0 < sigma
        assert 0 < problem_dim
        assert 0 < population_size

        # Step 1: initialize Population of size `population_size`
        # and then ensure it's sorted by it's fitness
        self.population = [
            Member(
                np.random.uniform(*problem_bounds, problem_dim),
                target_func,
                problem_bounds,
                mutation_type,
                recombination_type,
                sigma,
                recom_proba,
            )
            for _ in range(population_size)
        ]
        self.population.sort(key=lambda x: x.fitness)

        self.pop_size = population_size
        self.selection = selection_type
        self.max_func_evals = total_number_of_function_evaluations
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        self._func_evals = population_size

        # will store the optimization trajectory and lets you easily observe how
        # often a new best member was generated
        self.trajectory = [self.population[0]]

        print(f"Average fitness of population: {self.get_average_fitness()}")

    def get_average_fitness(self) -> float:
        """The average fitness of the current population"""
        return np.mean([pop.fitness for pop in self.population])

    def select_parents(self) -> List[int]:
        """Method that selects the parents

        Returns
        -------
        List[int]
            The indices of the parents in the sorted population
        """
        parent_ids: List[int] = []

        # TODO
        if self.selection == ParentSelection.NEUTRAL:
            raise NotImplementedError
        elif self.selection == ParentSelection.FITNESS:
            raise NotImplementedError
        elif self.selection == ParentSelection.TOURNAMENT:
            raise NotImplementedError
        else:
            raise NotImplementedError

        print(f"Selected parents: {parent_ids}")
        return parent_ids

    def step(self) -> float:
        """Performs one step of the algorithm

        2. Parent selection
        3. Offspring creation
        4. Survival selection

        Returns
        -------
        float
            The average population fitness
        """
        # Step 2: Parent selection
        parent_ids = self.select_parents()

        # Step 3: Variation / create offspring
        # TODO for each parent create exactly one offspring (use the frac_mutants)
        # parameter to determine if more recombination or mutation should be performed
        children: List[Member] = []
        for id in parent_ids:
            raise NotImplementedError
            self._func_evals += 1

        print(f"Children: {children}")

        # Step 4: Survival selection
        # (mu + lambda)-selection i.e. combine offspring and parents in one sorted list,
        # keeping the pop_size best of the population
        self.population.extend(children)

        # Resort the population
        self.population.sort(key=lambda x: x.fitness)

        # Reduce the population
        self.population = self.population[: self.pop_size]

        # Append the best Member to the trajectory
        self.trajectory.append(self.population[0])

        return self.get_average_fitness()

    def optimize(self) -> Member:
        """The optimization loop performing the desired amount of function evaluations

        Returns
        -------
        Member
            Returns the best member of the population after optimization
        """
        step = 1
        while self._func_evals < self.max_func_evals:
            avg_fitness = self.step()
            best_fitness = self.population[0].fitness
            lines = [
                "=========",
                f"Step: {step}",
                "=========",
                f"Avg. fitness: {avg_fitness:.7f}",
                f"Best. fitness: {best_fitness:.7f}",
                f"Func evals: {self._func_evals}",
                "----------------------------------",
            ]
            print("\n".join(lines))
            step += 1

        return self.population[0]


if __name__ == "__main__":
    """Simple main to give an example of how to use the EA"""
    from src.target_function import ackley

    np.random.seed(0)  # fix seed for comparisons sake

    dimensionality = 2
    max_func_evals = 500 * dimensionality
    pop_size = 20

    for selection in ParentSelection:
        ea = EA(
            target_func=ackley,
            population_size=pop_size,
            problem_dim=dimensionality,
            selection_type=selection,
            total_number_of_function_evaluations=max_func_evals,
        )
        optimum = ea.optimize()

        # print(ea.trajectory)
        print(optimum)
        print("#" * 120)
