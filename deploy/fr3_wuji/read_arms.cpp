// Only Robot::read: no control, recovery, enable or parameter-setting calls.
#include <franka/robot.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::mutex output;
  std::atomic<bool> failed{false};
  auto read = [&](const char* ip, const std::string& side) {
    try {
      franka::Robot robot(ip, franka::RealtimeConfig::kIgnore);
      uint64_t previous = 0;
      robot.read([&](const franka::RobotState& state) {
        auto stamp = state.time.toMSec();
        if (stamp >= previous + 10) {
          previous = stamp;
          double wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
          std::vector<std::string> names;
          for (int i = 1; i <= 7; ++i) names.push_back(side + "_fr3_joint" + std::to_string(i));
          nlohmann::json record = {{"source", side + "_arm"}, {"stamp", wall},
            {"stamp_basis", "franka_host_receive"}, {"robot_stamp_ms", stamp},
            {"measured", true}, {"units", "radian"}, {"names", names}, {"positions", state.q}};
          std::lock_guard<std::mutex> lock(output);
          std::cout << record.dump() << std::endl;
        }
        return !failed.load();
      });
    } catch (const std::exception& e) {
      failed = true;
      std::lock_guard<std::mutex> lock(output);
      std::cerr << side << " read failed: " << e.what() << std::endl;
    }
  };
  std::thread left(read, argv[1], "left"), right(read, argv[2], "right");
  left.join(); right.join();
  return failed ? 1 : 0;
}
