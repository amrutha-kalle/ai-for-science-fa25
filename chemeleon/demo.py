from chemeleon import Chemeleon
from chemeleon.visualize import Visualizer

# Load default model checkpoint (general text types)
chemeleon = Chemeleon.load_general_text_model()

# --- Generate crystal structure from text
n_samples = 5
n_atoms = 6
text_inputs = "A crystal structure of LiMnO4 with orthorhombic symmetry"

# Generate crystal structure
atoms_list = chemeleon.sample(text_inputs, n_atoms, n_samples)

visualizer = Visualizer(atoms_list)
visualizer.view(index=1)


# --- Generate crystal structure from text
# n_samples = 5
# n_atoms = 24
# text_inputs = "A crystal structure of LiMnO4 with orthorhombic symmetry"
#
# # Generate crystal structure with trajectory
# trajectory = chemeleon.sample(text_inputs, n_atoms, n_samples, return_trajectory=True)
#
# # Visualize the trajectory
# idx = 0
# traj_0 = [t[idx] for t in trajectory][::10] + [trajectory[-1][idx]]
# visualizer = Visualizer(traj_0, resolution=15)
# visualizer.view_trajectory(duration=0.1)
