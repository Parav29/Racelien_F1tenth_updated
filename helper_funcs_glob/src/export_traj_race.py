import numpy as np
import uuid
import hashlib


def export_traj_race(file_paths: dict,
                     traj_race: np.ndarray) -> None:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function is used to export the generated trajectory into a file. The generated files get an unique UUID and a
    hash of the ggv diagram to be able to check it later.

    Inputs:
    file_paths:     paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
    traj_race:      race trajectory [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    """

    # create random UUID
    rand_uuid = str(uuid.uuid4())

    # hash ggv file with SHA1
    if "ggv_file" in file_paths:
        with open(file_paths["ggv_file"], 'br') as fh:
            ggv_content = fh.read()
    else:
        ggv_content = np.array([])
    ggv_hash = hashlib.sha1(ggv_content).hexdigest()

    # write UUID and GGV hash into file
    with open(file_paths["traj_race_export"], 'w') as fh:
        fh.write("# " + rand_uuid + "\n")
        fh.write("# " + ggv_hash + "\n")

    # export race trajectory
    header = "s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2"
    fmt = "%.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f"
    with open(file_paths["traj_race_export"], 'ab') as fh:
        np.savetxt(fh, traj_race, fmt=fmt, header=header)

def export_traj_race_f110(file_paths: dict,
                     traj_race: np.ndarray,
                     w_tr_right: np.ndarray,
                     w_tr_left: np.ndarray) -> None:
    """
    Created by:
    Steven Gong (extended for MPC use)

    Documentation:
    Exports the full optimized trajectory to CSV with all columns needed for MPC,
    including track widths measured from the raceline to each wall.

    Output CSV columns:
    - s_m           [m]     Cumulative arc length along the raceline (progress variable)
    - x_m           [m]     X position of raceline point (east)
    - y_m           [m]     Y position of raceline point (north)
    - psi_rad       [rad]   Heading angle (-pi to +pi, 0 = north/+y axis)
    - kappa_radpm   [rad/m] Curvature (feedforward steering in MPC)
    - vx_mps        [m/s]   Target longitudinal velocity reference
    - ax_mps2       [m/s^2] Target longitudinal acceleration (feedforward throttle/brake)
    - w_tr_right_m  [m]     Distance from raceline to RIGHT track wall
    - w_tr_left_m   [m]     Distance from raceline to LEFT track wall

    Inputs:
    file_paths:  dict with key 'traj_race_export' pointing to output CSV path
    traj_race:   numpy array [N x 7] — full trajectory columns
    w_tr_right:  numpy array [N]     — right wall distance at each raceline point
    w_tr_left:   numpy array [N]     — left wall distance at each raceline point
    """

    # Stack width columns onto trajectory
    export_data = np.column_stack((traj_race, w_tr_right, w_tr_left))

    header = "s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,w_tr_right_m,w_tr_left_m"
    fmt = "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.4f,%.4f"

    with open(file_paths["traj_race_export"], 'w') as fh:
        np.savetxt(fh, export_data, fmt=fmt, header=header, comments='')

# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
