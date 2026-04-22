import torch

from layers.solar_physics import SolarPhysicsStack, project_to_feasible_set


def _assert_daytime_feasible(projected, cos_zenith, atol=1e-5):
    ghi = projected[..., 0]
    dni = projected[..., 1]
    dhi = projected[..., 2]
    closure = ghi - (cos_zenith * dni + dhi)
    torch.testing.assert_close(closure, torch.zeros_like(closure), atol=atol, rtol=0.0)
    assert torch.all(dni >= -atol)
    assert torch.all(dhi >= -atol)


def test_project_to_feasible_set_projects_night_to_zero():
    raw_state = torch.tensor([
        [[5.0, -2.0, 3.0], [1.0, 2.0, 3.0]],
        [[-4.0, 7.0, -8.0], [2.5, -1.5, 0.5]],
    ])
    cos_zenith = torch.tensor([
        [0.0, 0.02],
        [0.01, 0.0],
    ])

    projected = project_to_feasible_set(raw_state, cos_zenith, night_cos_threshold=0.02)
    torch.testing.assert_close(projected, torch.zeros_like(projected), atol=0.0, rtol=0.0)


def test_project_to_feasible_set_enforces_daytime_constraints():
    raw_state = torch.tensor([
        [[10.0, -2.0, 1.0], [5.0, 4.0, -3.0], [7.0, 1.5, 0.5]],
        [[2.0, -1.0, -4.0], [8.0, 6.0, 2.0], [3.0, 0.5, -0.2]],
    ])
    cos_zenith = torch.tensor([
        [0.8, 0.5, 0.9],
        [0.3, 0.7, 0.4],
    ])

    projected = project_to_feasible_set(raw_state, cos_zenith, night_cos_threshold=0.0)
    _assert_daytime_feasible(projected, cos_zenith)
    assert not torch.allclose(projected, raw_state)


def test_solar_physics_stack_preserves_shape_and_feasibility():
    torch.manual_seed(0)
    batch_size, pred_len, hidden_dim = 2, 4, 8
    context_seq = torch.randn(batch_size, pred_len, hidden_dim)
    base_state = torch.randn(batch_size, pred_len, 3)
    cos_zenith = torch.tensor([
        [0.0, 0.25, 0.6, 0.9],
        [0.02, 0.4, 0.75, 0.1],
    ])

    stack = SolarPhysicsStack(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.0,
        kernel_size=3,
        night_cos_threshold=0.02,
        use_gated_feedback=True,
    )

    projected, hidden = stack(
        context_seq=context_seq,
        base_state=base_state,
        cos_zenith=cos_zenith,
    )

    assert projected.shape == (batch_size, pred_len, 3)
    assert hidden.shape == (batch_size, pred_len, hidden_dim)

    night_mask = cos_zenith <= 0.02
    torch.testing.assert_close(
        projected[night_mask],
        torch.zeros_like(projected[night_mask]),
        atol=0.0,
        rtol=0.0,
    )

    day_mask = ~night_mask
    _assert_daytime_feasible(projected[day_mask], cos_zenith[day_mask])
