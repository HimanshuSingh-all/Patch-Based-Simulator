import numpy as np
from compartments import (
    CompartmentalModel,
    CompartmentPatchArray,
    SEIRV_patch_stepper
)
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def stepper_sir(
        state:NDArray,
        beta:float,
        gamma:float,
        alpha:float
        ):
    s1 = beta*state[0]*state[2]
    s2 = alpha*state[1]
    s3 = gamma*state[2]
    return np.array((-s1, s1-s2, s2-s3, s3))

init_states = np.array( (( 1000, 2, 0, 0, 0 ), ( 1000, 2, 0, 0, 0 ))) # dtype=int)
NUMCOMPARTMENTS = 5
NUMPATCHES  = 2
STEPS = 150


vaccinations = np.zeros(shape = (NUMPATCHES,))
sir_init_state = init_states[0, :4]
total_population = np.sum(init_states, axis = 1)[0]
print(f'Total population {total_population}')
print(f'SIR init population {sir_init_state}')
sir_init_state = sir_init_state/total_population
print(f'SIR init fractions {sir_init_state}')

network_matrix = np.array(((1,0), (0,1) )) #* 0.5
beta = 1.2
alpha = 1/5.8
gamma = 1/5
WEIBULL_WANING_SCALE = 0
WEIBULL_WANING_SHAPE = 0
betas_patches = np.ones((NUMPATCHES,))*beta


last_state = CompartmentPatchArray(
    state=init_states,
    num_compartments=NUMCOMPARTMENTS,
    num_patches=NUMPATCHES,
    iter_num = 0
)

states_patch_model = [last_state.get_copy_of_the_state()]
#### SIR Comparison
states_sir_model = [np.copy(sir_init_state)]
last_state_sir = sir_init_state

for _ in range(STEPS):
    step = SEIRV_patch_stepper(
        compartment_patch_array= last_state,
        vaccination_14_days_prior=vaccinations, 
        vaccine_efficacy=0,
        betas_patches=betas_patches,
        alpha=alpha,
        gamma= gamma,
        network_matrix=network_matrix,
        waning_weibull_shape=WEIBULL_WANING_SHAPE,
        waning_weibull_scale=WEIBULL_WANING_SCALE
    )
    # print(f'Patch state:\n {last_state.state}')
    # print(last_state.state.sum(axis=1)[0], total_population)
    assert np.isclose(last_state.state.sum(axis=1)[0] ,total_population)
    # print(f'Pathc 0 step s->e: { step[0, 0]}, beta*s0*i0/N {beta*last_state.state[0, 0]*last_state.state[0,2]/last_state.state.sum(axis=1)[0]}')
    last_state.update_state(step + last_state.state)
    states_patch_model.append( last_state.get_copy_of_the_state() )
    step_sir = stepper_sir(last_state_sir, beta=beta, gamma=gamma, alpha = alpha)
    last_state_sir = last_state_sir+step_sir
    states_sir_model.append(last_state_sir)
    
    # print(f'step seirv patch:\n {step}, \n step/N:\n {step/total_population}')
    # print(f'Step seir:\n {step_sir}')
states_sir_model = np.array(states_sir_model)
states_patch_model = np.array(states_patch_model)


print(f'States patch model shape {states_patch_model.shape}')
total_sum_states = states_patch_model[:, 0, :]

m = 5
times = np.arange(0 ,total_sum_states.shape[0])
plt.plot(times[::m], total_sum_states[::m, 0]/total_population, marker = 'x', markersize = 10, label = 'S isolated patch model')
plt.plot(times[::m], total_sum_states[::m, 1]/total_population, marker = 'x', markersize = 10, label = 'I isolated patch model')
# plt.plot(total_sum_states[:, 2]/total_population, marker = 'x', markersize = 10, label = 'R')

plt.plot(times[m//2::m], states_sir_model[m//2::m, 0], marker = 'v', linestyle = '-.', markerfacecolor='none', label = 'S')
plt.plot(times[m//2::m], states_sir_model[m//2::m, 1], marker = 'v', linestyle = '-.', markerfacecolor='none', label = 'I')
# plt.plot(states_sir_model[:, 2], marker = 'v', linestyle = '-.', markerfacecolor='none', label = 'R')
plt.legend()
plt.savefig('test.png')