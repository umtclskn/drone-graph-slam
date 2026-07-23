#!/bin/bash
# EVAL-08 — OURS replay (NDT front-end + GTSAM back-end + EKF2 prior) on a bag.
# Launches the install-space executables DIRECTLY (no `ros2 launch` wrapper) so
# `kill` actually reaps them — the pre-run hygiene check repeatedly found orphaned
# static_transform_publishers left behind by `ros2 launch` (COMPARISON.md §2.5).
# Mirrors graph_backend.launch.py exactly: same params file, same nodes, same
# identity base_link->lidar_link mount. Release build assumed.
#
# Env: BAG (default bags/slam_loop_03), OUT (log dir). Writes the keyframe log to
# analysis/eval05_covariance_log.csv, which the driver exports+scores.
# NB: no `set -u` — the ROS setup scripts reference unset vars (AMENT_*), same
# reason the sibling run_drone_bag.sh omits it.
WS=${WS:-/home/umut/portfolio_ws/drone_ws}
BAG=${BAG:-bags/slam_loop_03}
OUT=${OUT:-$WS/analysis/.ours_run_$(date -u +%Y%m%dT%H%M%SZ)}
PARAMS=$WS/install/graph_slam/share/graph_slam/config/slam_params.yaml
mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd "$WS"
echo "run_ours: BAG=$BAG OUT=$OUT PARAMS=$PARAMS"

PIDS=()
start() { local log=$1; shift; "$@" > "$OUT/$log.log" 2>&1 & PIDS+=($!); }

start ekf2_adapter $WS/install/px4_offboard/lib/px4_offboard/ekf2_odometry_adapter \
  --ros-args -p use_sim_time:=true
start ndt_frontend $WS/install/graph_slam/lib/graph_slam/ndt_frontend_node \
  --ros-args --params-file "$PARAMS" -p use_sim_time:=true \
  -p lidar_topic:=/x500/lidar_3d/points -p ekf2_topic:=/odometry/ekf2 \
  -p publish_debug_clouds:=true
start graph_backend $WS/install/graph_slam/lib/graph_slam/graph_backend_node \
  --ros-args --params-file "$PARAMS" -p use_sim_time:=true \
  -p ndt_odom_topic:=/ndt_frontend/ndt_odom
start tf_base_lidar /opt/ros/jazzy/lib/tf2_ros/static_transform_publisher \
  --frame-id base_link --child-frame-id lidar_link

sleep 5   # discovery
ros2 bag play "$BAG" --qos-profile-overrides-path bags/px4_qos_overrides.yaml \
  > "$OUT/bagplay.log" 2>&1
echo "=== bag done, waiting for front-end to drain ==="

# Idle detection: the CSV is truncated at backend start and appended per keyframe.
# Wait until its size stops growing for 15 s (max 180 s) BEFORE killing anything.
CSV=$WS/analysis/eval05_covariance_log.csv
prev=-1; stable=0
for i in $(seq 1 60); do
  sleep 3
  cur=$(stat -c %s "$CSV" 2>/dev/null || echo 0)
  if [ "$cur" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi
  echo "  t+$((i*3))s csv_bytes=$cur stable=$stable"
  [ "$stable" -ge 5 ] && break
  prev=$cur
done
echo "=== drained (csv $(wc -l < "$CSV") lines) ==="

kill "${PIDS[@]}" 2>/dev/null; sleep 3
kill -9 "${PIDS[@]}" 2>/dev/null; sleep 1
echo "=== done ==="
