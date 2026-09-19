cmake_minimum_required(VERSION 3.0.2)
project(gaitnet_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(
  FILES
  GaitTiming.msg
  RobotState.msg
  TerrainPatch.msg
  Observation.msg
  FootstepCommand.msg
  Nudge.msg
  PlannerCommand.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)
