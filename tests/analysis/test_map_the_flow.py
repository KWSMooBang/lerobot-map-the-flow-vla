from __future__ import annotations

import torch

from lerobot.analysis.map_the_flow import (
    AttentionKnockoutSpec,
    FlowRouteRule,
    apply_attention_knockout_mask,
    parse_layer_ranges,
    parse_route_rule,
)


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
