import torch

from conftest import GROUND, make_observation
from gaitnet_core.eligibility import step_eligible
from gaitnet_core.grid import FootholdGrid
from gaitnet_core.robot_spec import GO1
from gaitnet_core.terrain import valid_footholds


def test_cell_round_trip(grid):
    cells = torch.stack(torch.meshgrid(torch.arange(25), torch.arange(25), indexing="ij"), -1)
    legs = torch.arange(4).view(4, 1, 1)
    xy = grid.cell_to_xy(cells.expand(4, -1, -1, -1), legs)
    assert torch.allclose(xy, grid.cell_centers(), atol=1e-6)
    back, in_bounds = grid.xy_to_cell(xy + 0.4 * grid.resolution, legs)
    assert torch.equal(back, cells.expand(4, -1, -1, -1)) and in_bounds.all()
    # first index runs along x, the centre cell is the grid's centre
    for leg in range(4):
        assert torch.allclose(grid.cell_to_xy(torch.tensor([12, 12]), leg), grid.leg_centers()[leg], atol=1e-7)
        assert grid.cell_to_xy(torch.tensor([24, 12]), leg)[0] > grid.leg_centers()[leg, 0]


def test_grid_centres_mirror_outboard():
    grid = FootholdGrid(center=(0.02, 0.08))
    # FL, FR, RL, RR: left legs at +y, right legs at -y, all shifted forward alike
    expected = torch.tensor([[0.02, 0.08], [0.02, -0.08], [0.02, 0.08], [0.02, -0.08]])
    assert torch.allclose(grid.leg_centers(), expected)
    # the second index runs to the left on both sides
    for leg in range(4):
        assert grid.cell_to_xy(torch.tensor([12, 24]), leg)[1] > grid.leg_centers()[leg, 1]
    assert FootholdGrid.from_dict(grid.to_dict()) == grid
    legacy = {key: value for key, value in grid.to_dict().items() if key != "center"}
    assert FootholdGrid.from_dict(legacy).center == (0.0, 0.0)


def test_out_of_bounds(grid):
    half = grid.half_extent[0]
    centre = grid.leg_centers()[1]
    points = torch.tensor([[half + grid.resolution, 0.0], [0.0, 0.0]]) + centre
    _, in_bounds = grid.xy_to_cell(points, 1)
    assert in_bounds.tolist() == [False, True]


def test_flat_ground_all_valid(grid):
    obs = make_observation(2, grid)
    valid = valid_footholds(obs.terrain.heights, GO1, grid)
    assert valid.shape == (2, 4, *grid.size) and valid.all()


def test_reach_band_and_unknown(grid):
    obs = make_observation(1, grid)
    h = obs.terrain.heights
    h[0, 0] = -0.8  # FL: ground far below reach (a hole)
    h[0, 1] = -0.05  # FR: ground too high
    h[0, 2] = float("-inf")  # RL: no returns
    valid = valid_footholds(h, GO1, grid)
    assert not valid[0, :3].any() and valid[0, 3].all()


def test_edge_margin_around_a_pillar(grid):
    obs = make_observation(1, grid)
    b = grid.border
    h = obs.terrain.heights
    # a 0.1 m raised pillar covering inner cells [10, 15) x [10, 15)
    h[0, 0, b + 10 : b + 15, b + 10 : b + 15] = GROUND + 0.1
    valid = valid_footholds(h, GO1, grid, step_threshold=0.02, edge_margin=2)[0, 0]
    # step cells are rows/cols 9|10 and 14|15; margin 2 extends that to 7..17
    assert not valid[7:18, 7:18].any()
    assert valid[:7].all() and valid[18:].all() and valid[:, :7].all() and valid[:, 18:].all()
    # without a margin only the two cells either side of each step are edges
    valid0 = valid_footholds(h, GO1, grid, edge_margin=0)[0, 0]
    assert valid0[11:14, 11:14].all() and not valid0[9, 12] and not valid0[10, 12]


def test_edge_just_outside_grid_is_seen():
    grid = FootholdGrid(border=3)
    obs = make_observation(1, grid)
    obs.terrain.heights[0, 0, :1] = float("-inf")  # void in the border's outermost row
    valid = valid_footholds(obs.terrain.heights, GO1, grid, edge_margin=2)[0, 0]
    # outer border row 0 -> edge at patch rows 0,1 -> margin reaches patch row 3 = inner row 0
    assert not valid[0].any() and valid[1:].all()


def test_step_eligibility():
    timing = torch.zeros(3, 4, 3)
    timing[1, 0, 1] = 0.1  # one leg in swing: the other three may step
    timing[2, :2, 1] = 0.1  # two in swing: nobody may step
    eligible = step_eligible(timing)
    assert eligible[0].all()
    assert eligible[1].tolist() == [False, True, True, True]
    assert not eligible[2].any()
