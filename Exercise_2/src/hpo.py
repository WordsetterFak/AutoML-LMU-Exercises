from typing import Tuple

from src.evolution import EA, Mutation, ParentSelection, Recombination
from src.target_function import ackley


def evaluate_black_box(
    mutation: Mutation,
    selection: ParentSelection,
    recombination: Recombination,
) -> float:
    """Black-box evaluator of the EA algorithm.

    With your below hpo method you won't have to worry about other parameters

    Parameters
    ----------
    mutation: Mutation
        The choice of mutation strategy

    selection: ParentSelection
        The choice of parent selection strategy

    recombination: Recombination
        The choice of the recombination strategy

    Returns
    -------
    float
        The final fitness after optimizing
    """
    ea = EA(
        target_func=ackley,
        population_size=20,
        problem_dim=2,
        selection_type=selection,
        total_number_of_function_evaluations=500,
        problem_bounds=(-10, 10),
        mutation_type=mutation,
        recombination_type=recombination,
        sigma=1.0,
        children_per_step=5,
        fraction_mutation=0.5,
        recom_proba=0.5,
    )
    res = ea.optimize()
    return res.fitness


def determine_best_hypers() -> Tuple[Tuple[Mutation, ParentSelection, Recombination], float]:
    """Find the best combination with a sweep over the possible hyperparamters.

    # TODO
    Implement either grid or random search to determine the best hyperparameter
    setting of your EA implementation when overfitting to the ackley function.
    The only parameter values you have to consider are:
    * selection_type,
    * mutation_type
    * recombination type.

    You can treat the EA as a black-box by optimizing the black-box-function above.
    Note: the order of your "configuration" has to be as stated below

    Returns
    -------
    (Mutation, ParentSelection, Recombination), float
        Your best trio of strategies and the final fitness value of that strategy
    """
    # TODO
    best_startegy = (Mutation.NONE, ParentSelection.NEUTRAL, Recombination.NONE)
    fitness = 0.0
    raise NotImplementedError

    return best_startegy, fitness
