import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Protocol
from dataclasses import dataclass

@dataclass
class CompartmentPatchArray:
    """
    The wrapper class that essentially is an NDArray
    but with some checks about the size of the numpy array.    

    Attributes
    ----------
        state: NDArray
            The actual array that holds the state of the compartments
            (spread across the columns, different column is different
            compartment) for different pathces (different row is a 
            different _patch_). The shape of this array must be
            `(num_patches, num_compartments)`.
        num_patches: int
            The number of patches in our simulation
        num_compartments: int
            The number of compartments of the compartmental model.
        iter_num: int
            The number of the iterations from the 
    """
    state:NDArray
    num_patches:int
    num_compartments:int
    iter_num:int

    def __post_init__(self):
        if not (self.state.shape == (self.num_patches, self.num_compartments)):
            raise ValueError

    def update_state(self, newstate:NDArray):
        """
        Update the state.
        """
        if not (self.state.shape == newstate.shape):
            raise ValueError
        self.state = newstate
    
    def incr_iter_num(self, incerement:int = 1):
        """
        Increase the iter_num attribute by increment.

        Parameters
        ----------
            increment: int
                The integer amount by which to increment the iter_num.
        """
        self.iter_num = self.iter_num+incerement


    def get_copy_of_the_state(self):
        """
        Get a copy of the state.
        """
        return np.copy()


class CompartmentalModel:
    pass
    

def SEIRV_patch_stepper(
    compartment_patch_array:CompartmentPatchArray,
    # index: int,
    vaccination_14_days_prior: NDArray,
    vaccine_efficacy:float,
    betas_patches: NDArray|list,
    alpha:float,
    gamma:float,
    network_matrix:NDArray,
    waning_weiner_shape:float,
    waning_weiner_scale:float
    )->NDArray:
    """
    Patch stepper for SEIRV model

    Parameters
    ----------
        compartment_patch_array: CompartmentPatchArray
            The object of :CompartmentPatchArray: with 5 colmns.
        
        vaccination_14_days_prior: NDArray
            The array containing the number of vaccinations done in the 
            respective patches 14 days prior. Should be of the shape
            `(compartment_patch_array.num_patches, )`

        vaccine_efficacy: float
            The efficacy of the vaccine.

        betas_patches: NDArray
            The one dimensional array representing betas for
            the respective patches.
        
        alpha: float
            The rate of moving from exposed to infected
        
        gamma: float
            The rate of recovery parameter

        network_matrix: NDArray
            The interaction matrix between two patches.

        waning_weiner_shape: float
            The shape parameter from which the waning number of recovered individuals is drawn.

        waning_weiner_scale: float
            The scale parameter from which the waning number of recovered individuals is drawn.

    """
    assert network_matrix.shape[0] == network_matrix.shape[1]
    assert network_matrix.ndim == 2
    assert betas_patches == compartment_patch_array.num_patches
    assert compartment_patch_array.num_compartments == 5
    assert vaccination_14_days_prior.shape == (compartment_patch_array.num_patches, )
    
    randomwaning = waning_weiner_scale*np.random.weibull(
        a = waning_weiner_shape,
        size=compartment_patch_array.num_patches
        )
    

    waned_recovery_numbers = np.where(
       randomwaning >compartment_patch_array.state[:,3],
       compartment_patch_array.state[:,3],
       randomwaning
    )

    new_num_exposed_patches_per_day = pass

    new_num_infected_patches_per_day = alpha*compartment_patch_array.state[:,1]

    num_recovered_patch = gamma*compartment_patch_array.state[:,2]

    change = [
        waned_recovery_numbers-new_num_exposed_patches_per_day-vaccine_efficacy*vaccination_14_days_prior,
        new_num_exposed_patches_per_day - new_num_infected_patches_per_day,
        new_num_infected_patches_per_day-num_recovered_patch,
        num_recovered_patch-waned_recovery_numbers,
        vaccine_efficacy*vaccination_14_days_prior
    ]
    return np.array(change)




