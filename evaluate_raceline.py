import numpy as np
import matplotlib.pyplot as plt

# Load reference track
ref = np.genfromtxt('inputs/tracks/Map1_rc.csv', delimiter=',', comments='#')
x_ref, y_ref = ref[:, 0], ref[:, 1]
w_tr_right, w_tr_left = ref[:, 2], ref[:, 3]

# Estimate normals to plot boundaries
dx = np.gradient(np.append(x_ref, x_ref[0]))[:-1]
dy = np.gradient(np.append(y_ref, y_ref[0]))[:-1]
lengths = np.hypot(dx, dy)
nx = -dy / lengths
ny = dx / lengths

bound_r_x = x_ref + nx * w_tr_right
bound_r_y = y_ref + ny * w_tr_right
bound_l_x = x_ref - nx * w_tr_left
bound_l_y = y_ref - ny * w_tr_left

# Load optimized raceline
race = np.genfromtxt('outputs/Map1_rc/traj_race_cl-2026-05-29 02:41:40.295044.csv', delimiter=',', skip_header=1, comments='#')
x_race, y_race = race[:, 1], race[:, 2]

plt.figure(figsize=(10, 6))
plt.plot(x_ref, y_ref, 'k--', label='Centerline', alpha=0.5)
plt.plot(bound_r_x, bound_r_y, 'b-', label='Right Bound', alpha=0.7)
plt.plot(bound_l_x, bound_l_y, 'b-', label='Left Bound', alpha=0.7)
plt.plot(x_race, y_race, 'r-', label='Raceline', linewidth=2)
plt.axis('equal')
plt.legend()
plt.title('Raceline Evaluation for Map1_rc')
plt.savefig('outputs/Map1_rc/eval.png')
print("Image saved to outputs/Map1_rc/eval.png")
