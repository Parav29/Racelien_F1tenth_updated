import numpy as np
# import helper_funcs_glob as tph
import trajectory_planning_helpers as tph
import sys
import matplotlib.pyplot as plt


def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict,
               debug: bool = True,
               min_width: float = None) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function prepares the inserted reference track for optimization.

    Inputs:
    reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reg_smooth_opts:            parameters for the spline approximation
    stepsize_opts:              dict containing the stepsizes before spline approximation and after spline interpolation
    debug:                      boolean showing if debug messages should be printed
    min_width:                  [m] minimum enforced track width (None to deactivate)

    Outputs:
    reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
    a_interp:                   LES coefficients when calculating the splines
    coeffs_x_interp:            spline coefficients of the x-component
    coeffs_y_interp:            spline coefficients of the y-component
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # smoothing and interpolating reference track
    reftrack_interp = tph.spline_approximation. \
        spline_approximation(track=reftrack_imp,
                             k_reg=reg_smooth_opts["k_reg"],
                             s_reg=reg_smooth_opts["s_reg"],
                             stepsize_prep=stepsize_opts["stepsize_prep"],
                             stepsize_reg=stepsize_opts["stepsize_reg"],
                             debug=debug)

    # ------------------------------------------------------------------------------------------------------------------
    # CAP WIDTHS TO LOCAL CURVATURE RADIUS (prevents crossed normals at tight corners) ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    xi, yi = reftrack_interp[:, 0], reftrack_interp[:, 1]
    _dx  = np.gradient(np.append(xi, xi[0]))[:-1]
    _dy  = np.gradient(np.append(yi, yi[0]))[:-1]
    _ddx = np.gradient(np.append(_dx, _dx[0]))[:-1]
    _ddy = np.gradient(np.append(_dy, _dy[0]))[:-1]
    _denom = (_dx**2 + _dy**2)**1.5
    _kappa = np.abs(_dx * _ddy - _dy * _ddx) / np.where(_denom > 1e-9, _denom, 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        _radius = np.where(_kappa > 1e-6, 1.0 / _kappa, 1e6)
    _cap = 0.80 * _radius  # half-width must stay below 80% of radius
    _hw  = (reftrack_interp[:, 2] + reftrack_interp[:, 3]) / 2.0
    _scale = np.where(_hw > _cap, _cap / np.maximum(_hw, 1e-9), 1.0)
    reftrack_interp[:, 2] *= _scale
    reftrack_interp[:, 3] *= _scale

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
        calc_splines(path=refpath_interp_cl)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    normals_crossing = tph.check_normals_crossing.check_normals_crossing(track=reftrack_interp,
                                                                         normvec_normalized=normvec_normalized_interp,
                                                                         horizon=10)

    if normals_crossing:
        bound_1_tmp = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], axis=1)
        bound_2_tmp = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], axis=1)

        plt.figure()

        plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], 'k-')
        for i in range(bound_1_tmp.shape[0]):
            temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
            plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

        plt.grid()
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.title("Error: at least one pair of normals is crossed!")

        plt.show()

        raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

    # ------------------------------------------------------------------------------------------------------------------
    # ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    manipulated_track_width = False

    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

            if cur_width < min_width:
                manipulated_track_width = True

                # inflate to both sides equally
                reftrack_interp[i, 2] += (min_width - cur_width) / 2
                reftrack_interp[i, 3] += (min_width - cur_width) / 2

    if manipulated_track_width:
        print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
              " order to match the requirements!", file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
