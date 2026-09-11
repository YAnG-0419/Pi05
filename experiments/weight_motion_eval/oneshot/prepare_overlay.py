"""Copy the reference controller into Pi05 and add final-consumer deadlines.

Does not build, launch ROS, connect to devices, or alter the reference repository.
"""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path("/home/user/lpy/gello-retarget/ros_ws/src/franka_fr3_arm_controllers")
OUTPUT = ROOT / ".deployment/oneshot-overlay"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Reference controller changed; refusing an ambiguous source patch")
    return text.replace(old, new, 1)


def prepare():
    target = OUTPUT / "src/franka_fr3_arm_controllers"
    if target.exists():
        raise FileExistsError("Overlay source already exists; inspect before rebuilding")
    hashes = {
        str(p.relative_to(REFERENCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in REFERENCE.rglob("*")
        if p.is_file()
    }
    shutil.copytree(REFERENCE, target)
    header = target / "include/franka_fr3_arm_controllers/joint_impedance_controller.hpp"
    source = target / "src/joint_impedance_controller.cpp"
    h = header.read_text()
    h = replace_once(
        h, "#include <atomic>", '#include <atomic>\n#include "franka_fr3_arm_controllers/pi05_deadline.hpp"'
    )
    h = replace_once(
        h, "    std::uint64_t sequence{0};", "    std::uint64_t sequence{0};\n    std::int64_t expiry_ns{0};"
    )
    h = replace_once(
        h,
        "  double command_timeout_{0.25};",
        """  double command_timeout_{0.25};
  // This is a Pi05-only build. All commands require an unmodified deadline.
  std::atomic<bool> pi05_fault_{false};
  std::string pi05_run_;
  std::uint64_t pi05_sequence_{0};
  bool pi05_hold_initialized_{false};
  std::int64_t pi05_last_update_{0};""",
    )
    s = source.read_text()
    s = replace_once(
        s,
        "  JointCommand command;\n  command.positions = candidate;",
        """  pi05::Envelope envelope;
  const auto pi05_now = get_node()->now().nanoseconds();
  const auto pi05_created = rclcpp::Time(msg.header.stamp).nanoseconds();
  if (pi05_fault_.load() || !pi05::decode(msg.header.frame_id, envelope) ||
      !pi05::fresh(pi05_created, envelope.expiry, pi05_now) ||
      (pi05_run_.empty() ? envelope.sequence != 0 :
       (envelope.run != pi05_run_ || envelope.sequence != pi05_sequence_ + 1))) {
    pi05_fault_.store(true);
    RCLCPP_ERROR(get_node()->get_logger(), "Pi05 command identity/deadline fault latched; holding current pose.");
    return;
  }
  pi05_run_ = envelope.run;
  pi05_sequence_ = envelope.sequence;
  JointCommand command;
  command.expiry_ns = envelope.expiry;
  command.positions = candidate;""",
    )
    s = replace_once(
        s,
        "  if (command->sequence != consumed_command_sequence_) {",
        """  const auto pi05_now = get_node()->now().nanoseconds();
  if ((pi05_last_update_ != 0 && (pi05_now < pi05_last_update_ ||
       pi05_now - pi05_last_update_ > 50'000'000)) ||
      (command->sequence != 0 &&
       !pi05::fresh(command->source_stamp.nanoseconds(), command->expiry_ns, pi05_now))) {
    pi05_fault_.store(true);
  }
  pi05_last_update_ = pi05_now;
  if (pi05_fault_.load()) {
    // Latched until this controller instance is explicitly recreated. A fresh
    // queued frame cannot revive a timed-out trajectory.
    if (!pi05_hold_initialized_) {
      hold_position_ = q_;
      pi05_hold_initialized_ = true;
    }
    motion_generator_initialized_ = false;
    move_to_start_position_finished_ = false;
    gello_position_values_valid_ = false;
    command_status_.store(CommandStatus::kTimedOut, std::memory_order_relaxed);
    applied_goal_ = hold_position_;
    applied_goal_valid_ = true;
    tau_d_calculated = calculateTauDGains_(hold_position_);
    for (int i = 0; i < num_joints; ++i) command_interfaces_[i].set_value(tau_d_calculated(i));
    return controller_interface::return_type::OK;
  }
  if (command->sequence != consumed_command_sequence_) {""",
    )
    header.write_text(h)
    source.write_text(s)
    shutil.copy2(Path(__file__).with_name("deadline.hpp"), header.parent / "pi05_deadline.hpp")
    manifest = {
        "reference": str(REFERENCE),
        "source_hashes": hashes,
        "overlay_source_hashes": {
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in target.rglob("*")
            if p.is_file()
        },
        "hardware_started": False,
        "build_completed": False,
        "note": "A compiled controller still requires fake-interface and physical stop qualification.",
    }
    (OUTPUT / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared isolated controller source: {target}")


if __name__ == "__main__":
    prepare()
