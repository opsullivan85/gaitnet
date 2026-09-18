import torch

from gaitnet_core.samplers import Dense, UniformJitter, UniformLattice


def _valid(grid, n=4):
    valid = torch.ones(n, 4, *grid.size, dtype=torch.bool)
    valid[0, 0] = False  # a leg with no valid cells
    valid[1, :, :, :20] = False  # only 5 columns (125 cells) valid
    valid[2, 1, 3, 4] = False
    valid[2, 1] = False
    valid[2, 1, 3, 4] = True  # a leg with a single valid cell
    return valid


def _check_on_valid_cells(cands, valid, grid):
    n, l, k, _ = cands.xyz.shape
    cell, in_bounds = grid.xy_to_cell(cands.xyz[..., :2])
    rows = torch.arange(n).view(n, 1, 1).expand(n, l, k)
    legs = torch.arange(l).view(1, l, 1).expand(n, l, k)
    on_valid = valid[rows, legs, cell[..., 0], cell[..., 1]] & in_bounds
    assert on_valid[cands.valid].all()
    return cell


def test_uniform_lattice(grid):
    valid = _valid(grid)
    cands = UniformLattice(64).sample(valid, grid, generator=torch.Generator().manual_seed(0))
    assert cands.xyz.shape == (4, 4, 64, 3)
    num_valid = cands.num_valid()
    assert num_valid[0, 0] == 0 and num_valid[2, 1] == 1 and (num_valid[3] == 64).all()
    cell = _check_on_valid_cells(cands, valid, grid)
    # distinct cells within a leg, and exactly on cell centres
    flat = cell[..., 0] * grid.size[1] + cell[..., 1]
    assert len(set(flat[3, 0].tolist())) == 64
    assert torch.allclose(grid.cell_to_xy(cell)[cands.valid], cands.xyz[..., :2][cands.valid], atol=1e-6)
    assert (cands.log_q == 0).all()


def test_uniform_jitter_stays_in_cells(grid):
    valid = _valid(grid)
    cands = UniformJitter(64).sample(valid, grid, generator=torch.Generator().manual_seed(1))
    cell = _check_on_valid_cells(cands, valid, grid)
    offset = cands.xyz[..., :2] - grid.cell_to_xy(cell)
    assert (offset.abs() <= grid.resolution / 2 + 1e-6)[cands.valid].all()
    assert offset[cands.valid].abs().max() > 0.25 * grid.resolution  # actually off-lattice


def test_dense_covers_every_valid_cell(grid):
    valid = _valid(grid)
    cands = Dense().sample(valid, grid)
    assert cands.xyz.shape == (4, 4, grid.num_cells, 3)
    assert torch.equal(cands.valid, valid.reshape(4, 4, -1))
    assert torch.allclose(cands.xyz[3, 2, :, :2], grid.cell_centers().reshape(-1, 2), atol=1e-6)


def test_candidate_z_from_heights(grid):
    valid = torch.ones(1, 4, *grid.size, dtype=torch.bool)
    heights = torch.randn(1, 4, *grid.size)
    for sampler in (UniformLattice(16), UniformJitter(16), Dense()):
        cands = sampler.sample(valid, grid, heights=heights)
        cell, _ = grid.xy_to_cell(cands.xyz[..., :2])
        legs = torch.arange(4).view(1, 4, 1).expand_as(cell[..., 0])
        expected = heights[0][legs[0], cell[0, ..., 0], cell[0, ..., 1]]
        assert torch.allclose(cands.xyz[0, ..., 2], expected)
