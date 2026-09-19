# Terrain heights on a square grid of cells around each hip, in the order FL, FR, RL, RR.
#
# Each leg's patch is in that hip's yaw frame: origin at the hip joint, rotated by the base's
# yaw only. Cell (i, j) of a leg is at
#   x = (i - (size_x - 1) / 2) * resolution    (forward)
#   y = (j - (size_y - 1) / 2) * resolution    (left)
# and heights[(leg * size_x + i) * size_y + j] is the terrain's height there relative to the
# hip (m, negative below it). The patch must match the policy's grid, border included; the
# planner refuses any other size or resolution.

float32 UNKNOWN = -1000.0     # height of a cell with no data (NaN/inf can't cross rosbridge's JSON)

float32 resolution            # m per cell
uint16 size_x                 # cells along x, including the border
uint16 size_y                 # cells along y, including the border
float32[] heights             # 4 * size_x * size_y
