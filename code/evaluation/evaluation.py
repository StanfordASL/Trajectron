import numpy as np
from scipy.stats import gaussian_kde
from utils import prediction_output_to_trajectories


def compute_mse(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    mse = np.mean(error, axis=-1)
    return mse


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, -1] - gt_traj[-1], axis=-1)
    return final_error


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]

    for timestep in range(num_timesteps):
        kde = gaussian_kde(predicted_trajs[:, timestep].T)
        pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
        kde_ll += pdf / num_timesteps

    return -kde_ll


def compute_batch_statistics(prediction_output_dict, dt, max_hl, ph):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict, dt, max_hl, ph)

    batch_error_dict = {'mse': list(), 'fde': list(), 'kde': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            mse_errors = compute_mse(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            batch_error_dict['mse'].append(mse_errors)
            batch_error_dict['fde'].append(fde_errors)
            batch_error_dict['kde'].append(kde_ll)

    return (np.hstack(batch_error_dict['mse']),
            np.hstack(batch_error_dict['fde']),
            np.hstack(batch_error_dict['kde']))
