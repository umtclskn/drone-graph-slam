# L5-10: a coordinated batch of optimized keyframe states, published ONCE per
# accepted loop closure, covering every keyframe index in the affected range
# [match, query] whose pose iSAM2 just re-optimized.
#
# The front-end must apply every entry in ONE atomic pass before deciding
# whether to rebuild its submap target. L5-12 measured that republishing
# /slam/optimized_state once per historical keyframe (N independent per-entry
# messages) repeatedly re-triggers the per-entry collapse-on-large-jump guard
# (L5-19e) and collapses the active submap window down to a single stale
# entry, stalling NDT registration for the rest of the run (97-99/111
# keyframes on bags/slam_loop_03 vs 111/111 without it). This message exists
# so the front-end sees the whole affected range as one coordinated update,
# never as N independent ones.
#
# PX4-agnostic: std_msgs/graph_slam_msgs only, no px4_msgs anywhere.

# stamp = the event time (the loop-closure keyframe's ndt_odom header); not
#         matched against per-entry stamps, only informational.
std_msgs/Header header

# One entry per keyframe in the affected range. Each entry's own header.stamp
# is left at its default (zero) -- the front-end matches these by
# keyframe_id, never by stamp (the back-end does not retain historical
# per-keyframe stamps).
graph_slam_msgs/OptimizedState[] states
