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
        return np.copy(self.state)


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
    waning_weibull_shape:float,
    waning_weibull_scale:float
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

        waning_weibull_shape: float
            The shape parameter from which the waning number of recovered individuals is drawn.

        waning_weibull_scale: float
            The scale parameter from which the waning number of recovered individuals is drawn.

    """
    assert network_matrix.shape[0] == network_matrix.shape[1]
    assert network_matrix.ndim == 2
    assert betas_patches.shape[0] == compartment_patch_array.num_patches
    assert compartment_patch_array.num_compartments == 5
    assert vaccination_14_days_prior.shape == (compartment_patch_array.num_patches, )
    
    randomwaning = waning_weibull_scale*np.random.weibull(
        a = waning_weibull_shape,
        size=compartment_patch_array.num_patches
        )
    

    waned_recovery_numbers = np.where(
       randomwaning >compartment_patch_array.state[:,3],
       compartment_patch_array.state[:,3],
       randomwaning
    )

    stateref = compartment_patch_array.state
    cc = stateref[:,2]
    patch_population = stateref.sum(axis=1)
    new_num_exposed_patches_per_day = network_matrix @ (stateref[:,2]/patch_population)
    assert new_num_exposed_patches_per_day.ndim == 1
    assert new_num_exposed_patches_per_day.shape == betas_patches.shape

    new_num_exposed_patches_per_day = betas_patches * stateref[:, 0] * new_num_exposed_patches_per_day
    new_num_infected_patches_per_day = alpha*compartment_patch_array.state[:,1]

    num_recovered_patch = gamma*compartment_patch_array.state[:,2]

    change = [
        waned_recovery_numbers-new_num_exposed_patches_per_day-vaccine_efficacy*vaccination_14_days_prior,
        new_num_exposed_patches_per_day - new_num_infected_patches_per_day,
        new_num_infected_patches_per_day-num_recovered_patch,
        num_recovered_patch-waned_recovery_numbers,
        vaccine_efficacy*vaccination_14_days_prior
    ]
    return np.array(change).T


def SEIR_patch_get_trajectory(
    STEPS: int,
    compartment_patch_array:CompartmentPatchArray,
    vaccination_14_days_prior_list: NDArray,
    vaccine_efficacy:float,
    betas_patches: NDArray|list,
    alpha:float,
    gamma:float,
    network_matrix:NDArray,
    waning_weibull_shape:float,
    waning_weibull_scale:float,
    checks: bool =False
    )->list[NDArray]:
    """
    Patch stepper for SEIRV model.

    Parameters
    ----------
        compartment_patch_array: CompartmentPatchArray
            The object of :CompartmentPatchArray: with 5 colmns. This will be used to initialise the 
            trajectory.
        
        vaccination_14_days_prior_list: list[NDArray]
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

        waning_weibull_shape: float
            The shape parameter from which the waning number of recovered individuals is drawn.

        waning_weibull_scale: float
            The scale parameter from which the waning number of recovered individuals is drawn.

    """
    states = [compartment_patch_array.get_copy_of_the_state()]
    for i in range(STEPS):
        # print(i);
        step = SEIRV_patch_stepper(
            compartment_patch_array=compartment_patch_array,
            alpha= alpha,
            gamma= gamma,
            vaccination_14_days_prior= vaccination_14_days_prior_list[i],
            vaccine_efficacy= vaccine_efficacy,
            betas_patches = betas_patches,
            network_matrix= network_matrix,
            waning_weibull_scale= waning_weibull_scale,
            waning_weibull_shape= waning_weibull_shape
        )
        assert np.all(np.isclose(np.sum(states[-1], axis= 1), np.sum(states[0], axis = 1)))
        compartment_patch_array.update_state(compartment_patch_array.state + step)
        states.append(compartment_patch_array.get_copy_of_the_state())
    return states

# I asked gemini to create a wrapper function based on the 
# functions I had already written
def run_packaged_simulation(
    unit_mobility_matrix,
    age_contact_matrix,
    initial_state_units_ages,
    betas_units,
    steps=200,
    alpha=1/5.8,
    gamma=1/5.0,
    vaccination_daily_rate_units=None,
    vaccine_efficacy=0.8,
    waning_weibull_shape=3.7,
    waning_weibull_scale=120
):
    """
    Executes a multi-patch, age-stratified SEIRV epidemiological simulation 
    incorporating spatial mobility, age-based contact rates, and lagged vaccination.

    Parameters:
    -----------
    unit_mobility_matrix : np.ndarray (shape: NUM_UNITS x NUM_UNITS)
        Adjacency matrix representing travel probability between geographic districts.
    age_contact_matrix : np.ndarray (shape: NUM_AGES x NUM_AGES)
        Social mixing matrix defining interactions between different age cohorts.
    initial_state_units_ages : np.ndarray (shape: NUM_UNITS, NUM_AGES, 5)
        Initial population distribution across [S, E, I, R, V] compartments for 
        every age group in every geographic unit.
    betas_units : np.ndarray (shape: NUM_UNITS)
        The base transmission rate (beta) specific to each geographic district.
    steps : int, default=200
        Number of daily time-steps to run the simulation.
    alpha : float, default=1/5.8
        Incubation rate (inverse of average incubation period).
    gamma : float, default=1/5.0
        Recovery rate (inverse of average infectious period).
    vaccination_daily_rate_units : np.ndarray, optional
        Daily number of vaccine doses administered per unit. Distributed 
        equally across age groups.
    vaccine_efficacy : float, default=0.8
        Probability that a vaccine prevents infection.
    waning_weibull_shape : float, default=3.7
        Shape parameter for the Weibull distribution of immunity waning.
    waning_weibull_scale : float, default=120
        Scale parameter (characteristic life) for immunity waning in days.

    Returns:
    --------
    trajectory : np.ndarray
        A 4D tensor of shape (steps + 1, NUM_UNITS, NUM_AGES, 5) representing
        the temporal evolution of the system state.
    """
    NUM_UNITS = unit_mobility_matrix.shape[0]
    NUM_AGES = age_contact_matrix.shape[0]
    NUM_PATCHES = NUM_UNITS * NUM_AGES

    # 1. Combine Matrices
    network_matrix = combine_mobility_agerates(unit_mobility_matrix, age_contact_matrix)
    
    # 2. Flatten Initial States and Betas
    initial_state = initial_state_units_ages.reshape((NUM_PATCHES, 5))
    betas_patches = get_patch_basebetas_from_unit_betas(betas_units, NUM_AGES)
    
    # 3. Handle Daily Vaccination (Distribute equally across age groups)
    if vaccination_daily_rate_units is None:
        vaccination_daily_rate_units = np.zeros(NUM_UNITS)
    
    # Divide unit vaccines among age groups 
    vaccine_per_patch = np.repeat(vaccination_daily_rate_units / NUM_AGES, NUM_AGES)
   
    # Create the schedule and apply the 14-day lag automatically
    vaccination_schedule = np.tile(vaccine_per_patch, (steps, 1))
    vaccination_14_days_prior_list = np.zeros_like(vaccination_schedule)
    if steps > 14:
        vaccination_14_days_prior_list[14:] = vaccination_schedule[:-14]

    # 4. Initialize and Run
    patch_array = CompartmentPatchArray(
        state=np.copy(initial_state),
        num_patches=NUM_PATCHES,
        num_compartments=5,
        iter_num=0
    )
    
    trajectory = SEIR_patch_get_trajectory(
        STEPS=steps,
        compartment_patch_array=patch_array,
        vaccination_14_days_prior_list=vaccination_14_days_prior_list,
        vaccine_efficacy=vaccine_efficacy,
        betas_patches=betas_patches,
        alpha=alpha,
        gamma=gamma,
        network_matrix=network_matrix,
        waning_weibull_shape=waning_weibull_shape,
        waning_weibull_scale=waning_weibull_scale
    )
    
    # 5. Reshape back to human-readable dimensions (Time, Units, Ages, Compartments)
    return np.array(trajectory).reshape((steps + 1, NUM_UNITS, NUM_AGES, 5))


def combine_mobility_agerates(mobility_matrix: np.ndarray, age_rate_matrix: np.ndarray):
    """
    Combine the unit mobility matrix and the age contact matrix into a one
    big patch network interaction matrix.
    """
    shape = [mobility_matrix.shape[0] * age_rate_matrix.shape[0]] * 2
    network_matrix = np.zeros(shape=shape)
    num_ages = age_rate_matrix.shape[0]
    for i, mobility in enumerate(mobility_matrix):
        for j, mjtoi in enumerate(mobility):
            slice1i, slice1e = i * num_ages, (i + 1) * num_ages
            slice2i, slice2e = j * num_ages, (j + 1) * num_ages
            network_matrix[slice1i:slice1e, slice2i:slice2e] = mjtoi * age_rate_matrix
    return network_matrix

def get_patch_based_betas_from_unit_betas(betas_units: np.ndarray, num_age_groups: int):
    """
    Uses the input unit contact rate and extends it to all the age groups.
    This creates a new memory location and the views are not bein repeated.
    """
    return np.repeat(betas_units, num_age_groups)


def calibirate_betas():
    pass