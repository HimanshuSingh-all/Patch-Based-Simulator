import numpy as np
import matplotlib.pyplot as plt
from compartments import SEIR_patch_get_trajectory, CompartmentPatchArray

ANIMATE_NETWORK = True
STEPS = 50
NUMPATCHES = 10
network_matrix = np.zeros((NUMPATCHES, NUMPATCHES))
ALPHA = 1/30
GAMMA = 1/40
WANING_WEIBULL_SHAPE = 0*1.7
WANING_WEIBULL_SCALE = 0

max_pop_s = 3000
min_pop_s = 400
CHOSEN_INDEX = 1
INVERSE_DECAY_INDS = np.abs(1/(np.linspace(0, NUMPATCHES-1, NUMPATCHES) - CHOSEN_INDEX))
INVERSE_DECAY_INDS[CHOSEN_INDEX] = 0

np.fill_diagonal(network_matrix, 0)
network_matrix[CHOSEN_INDEX, CHOSEN_INDEX]= 0
if np.any(INVERSE_DECAY_INDS==np.nan) or np.any(INVERSE_DECAY_INDS==np.inf):
    raise ValueError

network_matrix[:, CHOSEN_INDEX] = INVERSE_DECAY_INDS 

init_array = np.zeros((NUMPATCHES, 5))
init_array[:, 0] = np.random.uniform(
    low=min_pop_s,
    high=max_pop_s,
    size=(NUMPATCHES, )
    )


init_array[CHOSEN_INDEX, :] = np.array([0,50,950 ,0, 0])
VACCINATION_STATUS = [np.zeros(NUMPATCHES)]*STEPS

patch_array = CompartmentPatchArray(
    state= init_array,
    num_patches=NUMPATCHES,
    num_compartments=5,
    iter_num = 5
)

betas_patches = np.ones(NUMPATCHES)*0.2
trajectory = SEIR_patch_get_trajectory(
    STEPS=STEPS,
    compartment_patch_array=patch_array,
    vaccination_14_days_prior_list= VACCINATION_STATUS,
    vaccine_efficacy= 0.0,
    betas_patches=betas_patches,
    alpha= ALPHA,
    gamma=GAMMA,
    network_matrix=network_matrix,
    waning_weibull_scale=WANING_WEIBULL_SCALE,
    waning_weibull_shape=WANING_WEIBULL_SHAPE
)

assert len(trajectory) == STEPS+1
trajectory = np.array(trajectory)
print(trajectory.shape)
print(network_matrix)
plt.plot(trajectory[:, 1, 2], label = 'Patch-2 infected')
plt.plot(trajectory[:, 1, 3], label = 'Patch-2 R')
plt.plot(trajectory[:, 1, 0], label = 'Patch-2 S')
plt.plot(trajectory[:, 0, 2], label = 'Patch-1 infected')
plt.plot(trajectory[:, 0, 0], label = 'Patch-1 S')
plt.plot(trajectory[:, 0, 3], label = 'Patch-1 R')
# plt.savefig('test_patch.png', dpi = 300)
plt.legend()




## I used gemini for the network visualisation
# Prompt:
#"I have a (Timsteps, Units, Compartment) array. 
# Now I have another matrix that basically has the strength of the interactions between units.
# I want to plot a heatmap of number (or fraction per unit ) in compartment number 3 (index 2 compartment) 
# and animate it."
# I had to furhter tweak some stuff, it was symmetrising the interaction matrix, which would mean the 
# network is reciprocal but in this case there is no reciprocity.
# Morover it said fraction in the colorbar label but it wasn't actually plotting the fraction and 
# instead the total number which is obviously greater than 1, normally it wouldnt be a problem but the vmax was set to 1 
# because it assumed the data to be in fractions, which was not true even though it wrote that code.
import networkx as nx
import matplotlib.animation as animation

if ANIMATE_NETWORK:
    # --- 1. DATA SETUP ---
    data_array = trajectory 
    interaction_matrix = network_matrix
    np.fill_diagonal(interaction_matrix, 0)

    # --- 2. GRAPH & LAYOUT ---
    G = nx.from_numpy_array(interaction_matrix)
    # We calculate pos once to keep the 'weakness = distance' stable
    pos = nx.spring_layout(G, weight='weight',) # seed=42)

    # --- 3. PLOT INITIALIZATION ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    fig.set_facecolor('#f0f0f0')


    cmap_ne =  plt.cm.RdYlBu_r#plt.cm.YlOrRd
    # We create the colorbar outside the update loop so it doesn't duplicate
    sm = plt.cm.ScalarMappable(cmap=cmap_ne, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[0], fraction=0.046, pad=0.04)
    cbar.set_label('Fraction of infected Per Patch' , rotation=270, labelpad=15)

    def update(frame):
        ax[0].clear()
        ax[0].set_axis_off()
        
        current_values = data_array[frame, :, 2]/np.sum(data_array[frame], axis=1)
        
        # Draw edges: Width = Strength of interaction
        # (High interaction = thick line/short distance)
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.15, edge_color='gray', ax=ax[0])
        
        # Draw nodes: Color = Compartment value
        nodes = nx.draw_networkx_nodes(
            G, pos, 
            node_color=current_values, 
            node_size=600,
            cmap=cmap_ne,
            vmin=0, vmax=1,
            edgecolors='white', # Adds a border to nodes for clarity
            ax=ax[0]
        )
        
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax[0])
        ax[0].set_title(f"Simulation Timestep: {frame}", fontsize=15, pad=20)

        return nodes,

    # --- 4. THE ANIMATION OBJECT ---
    # 'interval' is the delay between frames in milliseconds (50ms = 20fps)
    ax[1].set_title('Patch 1 with initial infections, and no connections to itself. \n Patch Interaction matrix is just 1/(difference between patch indices) :')
    SM = ax[1].imshow(network_matrix, origin = 'upper', cmap = 'Blues')
    cbar = fig.colorbar(SM, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label('Patch interaction Weight', rotation=270, labelpad=15)
    fig.suptitle('Initialisation of network with "Epicenter" patch with no immunity waning')
    ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=100, blit=False)


    # --- 5. SCRIPT EXECUTION ---
    # In a standalone script, this opens the interactive window.
    ani.save('handput_network_animation.gif', writer='pillow', fps=10)
    plt.tight_layout()
plt.show()

    # OPTIONAL: If you want to save the file instead of just watching it:
    # print("Saving animation...")
    # ani.save('network_evolution.mp4', writer='ffmpeg', fps=20)