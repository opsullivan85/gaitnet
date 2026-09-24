# Terrain heights on a grid of cells near each hip, in the order FL, FR, RL, RR.
#
# Each leg's patch is in that hip's yaw frame: origin at the hip joint, rotated by the base's
# yaw only. The patch is centred at (center_x, center_y) from a left hip and at
# (center_x, -center_y) from a right hip, so positive center_y is outboard on every leg. With
# s = +1 on the left legs (FL, RL) and -1 on the right (FR, RR), cell (i, j) of a leg is at
#   x = center_x     + (i - (size_x - 1) / 2) * resolution    (forward)
#   y = s * center_y + (j - (size_y - 1) / 2) * resolution    (left)
# and heights[(leg * size_x + i) * size_y + j] is the terrain's height there relative to the
# hip (m, negative below it). The patch must match the policy's grid, border and centre
# included; the planner refuses any other size, resolution or centre.

float32 UNKNOWN = -1000.0     # height of a cell with no data (NaN/inf can't cross rosbridge's JSON)

float32 resolution            # m per cell
uint16 size_x                 # cells along x, including the border
uint16 size_y                 # cells along y, including the border
float32 center_x              # m, the patch centre ahead of each hip
float32 center_y              # m, the patch centre outboard of each hip
float32[] heights             # 4 * size_x * size_y
