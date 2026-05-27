import numpy as np

# Mock reftrack_interp with 4 points
reftrack_interp = np.array([
    [0, 0, 1.0, 1.0],
    [1, 0, 1.0, 1.0],
    [1, 1, 1.0, 1.0],
    [0, 1, 1.0, 1.0]
])
alpha_opt = np.array([0.1, -0.1, 0.2, -0.2])

diffs = np.vstack((np.diff(reftrack_interp[:, :2], axis=0), 
                   reftrack_interp[0, :2] - reftrack_interp[-1, :2]))
s_reftrack = np.insert(np.cumsum(np.sqrt(np.sum(diffs**2, axis=1))), 0, 0.0)

print(f"s_reftrack len: {len(s_reftrack)}")

w_tr_right_raceline = reftrack_interp[:, 2] - alpha_opt
w_tr_right_raceline_closed = np.append(w_tr_right_raceline, w_tr_right_raceline[0])

print(f"w_tr_right_raceline_closed len: {len(w_tr_right_raceline_closed)}")

s_raceline = np.linspace(0, 4.0, 10)
s_ref_wrap = s_reftrack / s_reftrack[-1] * s_raceline[-1]

w_right_interp = np.interp(s_raceline, s_ref_wrap, w_tr_right_raceline_closed)
print("Success!")
