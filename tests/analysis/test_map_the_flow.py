from __future__ import annotations

import torch

from lerobot.analysis.map_the_flow import (
    AttentionKnockoutSpec,
    FlowRouteRule,
    apply_attention_knockout_mask,
    parse_layer_ranges,
    parse_route_rule,
    parse_route_rules,
)
from lerobot.policies.groot.action_head.cross_attention_dit import _make_attention_knockout_bias


def test_parse_layer_ranges():
    assert parse_layer_ranges("1-5,6-10") == ((1, 5), (6, 10))
    assert parse_layer_ranges(["11-15", "20"]) == ((11, 15), (20, 20))


def test_parse_route_rule_with_alias_and_range():
    rule = parse_route_rule("video->text@6-20")
    assert rule.source == "vision"
    assert rule.target == "language"
    assert rule.layer_ranges == ((6, 20),)


def test_parse_route_rule_with_view_alias():
    rule = parse_route_rule("camera_0->view-1@1-5")
    assert rule.source == "view0"
    assert rule.target == "view1"
    assert rule.layer_ranges == ((1, 5),)


def test_parse_bidirectional_route_rules():
    rules = parse_route_rules("camera_0<->view-1@1-5")
    assert [(rule.source, rule.target, rule.layer_ranges) for rule in rules] == [
        ("view0", "view1", ((1, 5),)),
        ("view1", "view0", ((1, 5),)),
    ]


def test_block_route_masks_target_queries_to_source_keys():
    mask = torch.zeros(1, 1, 4, 4)
    spec = AttentionKnockoutSpec(
        rules=(FlowRouteRule("vision", "language", ((1, 1),)),),
        mode="block",
    )
    out = apply_attention_knockout_mask(
        mask,
        layer_index=0,
        spec=spec,
        source_spans={"vision": [(0, 2)], "language": [(2, 4)]},
        target_spans={"vision": [(0, 2)], "language": [(2, 4)]},
    )
    assert torch.all(out[..., 2:4, 0:2] < -1e20)
    assert torch.all(out[..., 0:2, 0:2] == 0)
    assert torch.all(out[..., 2:4, 2:4] == 0)


def test_keep_only_preserves_allowed_route_and_blocks_others():
    mask = torch.zeros(1, 1, 3, 5)
    spec = AttentionKnockoutSpec(
        rules=(FlowRouteRule("language", "action", ((2, 3),)),),
        mode="keep_only",
    )
    out = apply_attention_knockout_mask(
        mask,
        layer_index=1,
        spec=spec,
        source_spans={
            "vision": [(0, 2)],
            "language": [(2, 4)],
            "prefix": [(0, 4)],
            "action": [(4, 5)],
        },
        target_spans={"suffix": [(0, 3)], "action": [(0, 3)]},
    )
    assert torch.all(out[..., 0:3, 2:4] == 0)
    assert torch.all(out[..., 0:3, 0:2] < -1e20)


def test_cross_view_route_masks_only_other_view():
    mask = torch.zeros(1, 1, 4, 4)
    spec = AttentionKnockoutSpec(
        rules=(FlowRouteRule("view0", "view1", ((1, 1),)),),
        mode="block",
    )
    out = apply_attention_knockout_mask(
        mask,
        layer_index=0,
        spec=spec,
        source_spans={
            "vision": [(0, 4)],
            "view0": [(0, 2)],
            "view1": [(2, 4)],
        },
        target_spans={
            "vision": [(0, 4)],
            "view0": [(0, 2)],
            "view1": [(2, 4)],
        },
    )
    assert torch.all(out[..., 2:4, 0:2] < -1e20)
    assert torch.all(out[..., 0:2, 2:4] == 0)
    assert torch.all(out[..., 0:2, 0:2] == 0)
    assert torch.all(out[..., 2:4, 2:4] == 0)


def test_per_sample_token_masks_apply_independently():
    """Per-batch token masks let each sample knock out different positions."""

    mask = torch.zeros(2, 1, 3, 6)  # (B=2, H=1, Q=3, K=6)

    # Sample 0: vision tokens at key positions [1, 2], language at [3, 4]
    # Sample 1: vision tokens at key positions [0, 1], language at [2, 3]
    vision_mask = torch.tensor(
        [
            [False, True, True, False, False, False],
            [True, True, False, False, False, False],
        ]
    )
    language_mask = torch.tensor(
        [
            [False, False, False, True, True, False],
            [False, False, True, True, False, False],
        ]
    )

    spec = AttentionKnockoutSpec(
        rules=(FlowRouteRule("vision", "action", ((1, 1),)),),
        mode="block",
    )
    out = apply_attention_knockout_mask(
        mask,
        layer_index=0,
        spec=spec,
        source_spans={"action": [(4, 6)]},
        target_spans={"action": [(0, 3)]},
        source_token_masks={"vision": vision_mask, "language": language_mask},
    )

    # Sample 0: positions [1, 2] are vision keys -> blocked across all action queries
    assert torch.all(out[0, ..., :, 1:3] < -1e20)
    assert torch.all(out[0, ..., :, 0:1] == 0)
    assert torch.all(out[0, ..., :, 3:6] == 0)

    # Sample 1: positions [0, 1] are vision keys -> blocked across all action queries
    assert torch.all(out[1, ..., :, 0:2] < -1e20)
    assert torch.all(out[1, ..., :, 2:6] == 0)


def test_token_masks_or_with_spans_for_same_group():
    """Span and token_mask for the same group OR together to form one selector."""

    mask = torch.zeros(1, 1, 2, 5)
    # vision spans cover positions [0, 1]; token mask additionally covers position [3]
    spec = AttentionKnockoutSpec(
        rules=(FlowRouteRule("vision", "action", ((1, 1),)),),
        mode="block",
    )
    extra_vision = torch.tensor([[False, False, False, True, False]])
    out = apply_attention_knockout_mask(
        mask,
        layer_index=0,
        spec=spec,
        source_spans={"vision": [(0, 2)], "action": [(4, 5)]},
        target_spans={"action": [(0, 2)]},
        source_token_masks={"vision": extra_vision},
    )

    assert torch.all(out[..., :, 0:2] < -1e20)  # span coverage
    assert torch.all(out[..., :, 3:4] < -1e20)  # token mask extends coverage
    assert torch.all(out[..., :, 2:3] == 0)
    assert torch.all(out[..., :, 4:5] == 0)


def test_attention_knockout_bias_respects_layer_offset():
    """GR00T post-VL layers can be addressed as global layers after Eagle."""

    spec = AttentionKnockoutSpec(
        rules=(FlowRouteRule("vision", "language", ((13, 13),)),),
        mode="block",
    )

    local_layer = _make_attention_knockout_bias(
        batch_size=1,
        query_length=4,
        key_length=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        layer_index=0,
        attention_knockout=spec,
        source_spans={"vision": [(0, 2)], "language": [(2, 4)]},
        target_spans={"vision": [(0, 2)], "language": [(2, 4)]},
    )
    assert local_layer is not None
    assert torch.all(local_layer == 0)

    global_layer = _make_attention_knockout_bias(
        batch_size=1,
        query_length=4,
        key_length=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        layer_index=0,
        layer_offset=12,
        attention_knockout=spec,
        source_spans={"vision": [(0, 2)], "language": [(2, 4)]},
        target_spans={"vision": [(0, 2)], "language": [(2, 4)]},
    )
    assert global_layer is not None
    assert torch.all(global_layer[..., 2:4, 0:2] < -1e20)
    assert torch.all(global_layer[..., 0:2, :] == 0)
