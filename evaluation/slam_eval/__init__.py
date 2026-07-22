"""slam_eval — system-agnostic SLAM trajectory + consistency metric engine.

Pure files-in -> numbers-out: no ROS imports anywhere in this package. Scores
any system (ours, LIO-SAM, ...) that exports the file contract documented in
README.md (TUM trajectory + optional 6x6 covariance file + TUM ground truth).

Entry point:  python -m slam_eval --est est.tum --cov est_cov.txt --gt gt.tum --out <dir>/
"""

# EVAL-05 NEES calibration result (analysis/calibration_log.md, 2026-06-30):
# the raw NDT Hessian-based Sigma_meas was ~435x overconfident on
# bags/slam_loop_02 (mean NEES 1306 vs ideal 3 -> alpha = 1306/3 = 435).
# Since then alpha is baked into the live pipeline via
# graph_slam/config/slam_params.yaml (ndt_cov_scale_factor: 435.0), so
# covariances exported from the CURRENT stack are already calibrated and must
# be scored with --cov_scale 1.0 (the CLI default). Pass
# --cov_scale NDT_COV_SCALE_ALPHA_CALIBRATED (435) only when re-scoring a raw,
# uncalibrated covariance log.
NDT_COV_SCALE_ALPHA_CALIBRATED = 435.0
