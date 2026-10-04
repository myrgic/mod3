"""Animation clips: agent-authored motion for a body, as data.

A clip is JSON. It never carries code: the body page runs a small fixed set of
generators (const, keys, wave, jitter, follow), and the server checks every
clip against the body's manifest before it is sent. Agents can still invent
freely; they just say *what* to move, not *how to execute*.

    {
      "name": "sulk",
      "description": "turn away, chin down, peek back now and then",
      "args": {"intensity": {"default": 1, "min": 0, "max": 2}},
      "duration_ms": 4000,          # omit or null + "loop": true for a loop
      "loop": false,
      "fade_ms": 300,                # blend in/out over the procedural layer
      "base": "IDLE",                # optional named state to run underneath
      "tracks": {
        "head.x": {"keys": [[0, 0], [500, "-0.8*intensity"]], "ease": "inOut"},
        "head.y": {"const": "-0.3*intensity"},
        "eyes.x": {"jitter": {"pick": [-0.6, 0.6], "every_ms": [1500, 3000],
                              "hold_ms": [150, 300], "rest": -0.2}, "smooth": 0.2}
      }
    }

Channels are semantic and normalised, so one clip works on any body that maps
them: bipolar channels take -1..1 around the parameter's default, unipolar
ones take 0..1 across its range. A track named ``param:ParamFoo`` writes a raw
model parameter in the model's own units (escape hatch; still range-checked).

Numeric fields may be strings: arithmetic over the clip's args
(``"-0.8*intensity"``, ``"max(0.2, 1-intensity)"``). They are evaluated here,
in a whitelisted AST walk, so the body only ever sees numbers.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

# channel → polarity. "bi" = -1..1 around default, "uni" = 0..1 across range.
CHANNELS: dict[str, str] = {
    "head.x": "bi", "head.y": "bi", "head.z": "bi",
    "body.x": "bi", "body.y": "bi", "body.z": "bi",
    "eyes.x": "bi", "eyes.y": "bi",
    "eyes.open": "uni", "eye.l.open": "uni", "eye.r.open": "uni",
    "eyes.smile": "uni",
    "mouth.open": "uni", "mouth.smile": "bi",
    "brows.y": "bi", "brows.angle": "bi",
    "cheek": "uni", "breath": "uni",
}  # fmt: skip

GENERATORS = ("const", "keys", "wave", "jitter", "follow")
EASES = ("linear", "in", "out", "inOut", "step")
SHAPES = ("sin", "abs", "tri", "square")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

MAX_TRACKS = 32
MAX_KEYS = 256
MAX_DURATION_MS = 120_000
MIN_PERIOD_MS = 40


class ClipError(ValueError):
    """The clip is malformed. The message says where."""


# ------------------------------------------------------------ expressions
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_FUNCS = {
    "min": min,
    "max": max,
    "abs": abs,
    "clamp": lambda v, lo, hi: min(max(v, lo), hi),
    "sin": math.sin,
    "cos": math.cos,
}


def evaluate(expr: Any, args: dict[str, float], where: str) -> float:
    """A number, or an arithmetic string over ``args``. Nothing else runs."""
    if isinstance(expr, bool):
        raise ClipError(f"{where}: expected a number, got a boolean")
    if isinstance(expr, (int, float)):
        v = float(expr)
    elif isinstance(expr, str):
        if len(expr) > 200:
            raise ClipError(f"{where}: expression longer than 200 characters")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ClipError(f"{where}: bad expression {expr!r}") from exc
        v = float(_walk(tree.body, args, where))
    else:
        raise ClipError(f"{where}: expected a number or expression, got {type(expr).__name__}")
    if not math.isfinite(v):
        raise ClipError(f"{where}: {expr!r} is not finite")
    return v


def _walk(node: ast.AST, args: dict[str, float], where: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in args:
            return args[node.id]
        if node.id == "pi":
            return math.pi
        raise ClipError(f"{where}: unknown name {node.id!r} (args: {', '.join(args) or 'none'})")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _walk(node.operand, args, where)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _walk(node.left, args, where), _walk(node.right, args, where)
        if isinstance(node.op, ast.Div) and b == 0:
            raise ClipError(f"{where}: division by zero")
        return _BIN[type(node.op)](a, b)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
        and len(node.args) <= 3
    ):
        return float(_FUNCS[node.func.id](*(_walk(a, args, where) for a in node.args)))
    raise ClipError(f"{where}: only numbers, args, + - * /, and {', '.join(_FUNCS)} are allowed")


# ------------------------------------------------------------ resolution
def resolve_args(clip: dict[str, Any], overrides: dict[str, Any] | None) -> tuple[dict[str, float], list[str]]:
    """Declared args with defaults, overridden and clamped. Returns (args, rejected)."""
    spec = clip.get("args") or {}
    if not isinstance(spec, dict):
        raise ClipError("args: must be an object")
    out: dict[str, float] = {}
    for name, s in spec.items():
        if not re.match(r"^[a-z_][a-z0-9_]{0,31}$", name):
            raise ClipError(f"args.{name}: arg names are lowercase identifiers")
        s = s if isinstance(s, dict) else {"default": s}
        lo = float(s.get("min", -math.inf))
        hi = float(s.get("max", math.inf))
        default = evaluate(s.get("default", 0), {}, f"args.{name}.default")
        out[name] = min(max(default, lo), hi)
    rejected: list[str] = []
    for name, value in (overrides or {}).items():
        if name not in out:
            rejected.append(name)
            continue
        s = spec[name] if isinstance(spec[name], dict) else {}
        v = evaluate(value, {}, f"arg {name}")
        out[name] = min(max(v, float(s.get("min", -math.inf))), float(s.get("max", math.inf)))
    return out, rejected


def _range(v: Any, args: dict[str, float], where: str) -> list[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ClipError(f"{where}: expected [min_ms, max_ms]")
    a, b = evaluate(v[0], args, f"{where}[0]"), evaluate(v[1], args, f"{where}[1]")
    if a < 0 or b < a:
        raise ClipError(f"{where}: need 0 <= min <= max")
    return [a, b]


def _many(spec: Any) -> list[Any]:
    return spec if isinstance(spec, list) else [spec]


def _gen(kind: str, spec: Any, args: dict[str, float], where: str, channels: set[str]) -> Any:
    if kind == "const":
        return evaluate(spec, args, where)
    if kind == "keys":
        if not isinstance(spec, list) or not spec:
            raise ClipError(f"{where}: expected a non-empty list of [t_ms, value]")
        if len(spec) > MAX_KEYS:
            raise ClipError(f"{where}: more than {MAX_KEYS} keys")
        keys = []
        for i, k in enumerate(spec):
            if not isinstance(k, (list, tuple)) or len(k) != 2:
                raise ClipError(f"{where}[{i}]: expected [t_ms, value]")
            keys.append([evaluate(k[0], args, f"{where}[{i}].t"), evaluate(k[1], args, f"{where}[{i}].v")])
        if any(b[0] < a[0] for a, b in zip(keys, keys[1:])):
            raise ClipError(f"{where}: key times must not decrease")
        return keys
    if not isinstance(spec, dict):
        raise ClipError(f"{where}: expected an object")
    if kind == "wave":
        shape = spec.get("shape", "sin")
        if shape not in SHAPES:
            raise ClipError(f"{where}.shape: one of {', '.join(SHAPES)}")
        period = evaluate(spec.get("period_ms", 1000), args, f"{where}.period_ms")
        if period < MIN_PERIOD_MS:
            raise ClipError(f"{where}.period_ms: at least {MIN_PERIOD_MS}")
        return {
            "shape": shape,
            "amp": evaluate(spec.get("amp", 1), args, f"{where}.amp"),
            "period_ms": period,
            "phase": evaluate(spec.get("phase", 0), args, f"{where}.phase"),
            "offset": evaluate(spec.get("offset", 0), args, f"{where}.offset"),
        }
    if kind == "jitter":
        out: dict[str, Any] = {"every_ms": _range(spec.get("every_ms", [1000, 2000]), args, f"{where}.every_ms")}
        if out["every_ms"][0] < MIN_PERIOD_MS:
            raise ClipError(f"{where}.every_ms: at least {MIN_PERIOD_MS}")
        if "pick" in spec:
            pick = spec["pick"]
            if not isinstance(pick, list) or not pick:
                raise ClipError(f"{where}.pick: expected a non-empty list")
            out["pick"] = [evaluate(p, args, f"{where}.pick[{i}]") for i, p in enumerate(pick)]
        else:
            out["min"] = evaluate(spec.get("min", -1), args, f"{where}.min")
            out["max"] = evaluate(spec.get("max", 1), args, f"{where}.max")
        if "hold_ms" in spec:
            out["hold_ms"] = _range(spec["hold_ms"], args, f"{where}.hold_ms")
            out["rest"] = evaluate(spec.get("rest", 0), args, f"{where}.rest")
        return out
    if kind == "follow":
        of = spec.get("of")
        if of not in CHANNELS:
            raise ClipError(f"{where}.of: a channel name ({', '.join(CHANNELS)})")
        channels.add(of)
        return {
            "of": of,
            "gain": evaluate(spec.get("gain", 1), args, f"{where}.gain"),
            "offset": evaluate(spec.get("offset", 0), args, f"{where}.offset"),
        }
    raise ClipError(f"{where}: unknown generator")  # pragma: no cover


def compile_clip(
    clip: dict[str, Any],
    *,
    body_channels: dict[str, Any] | None = None,
    body_params: dict[str, tuple[float, float]] | None = None,
    body_states: list[str] | None = None,
    args: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a clip and resolve it to numbers for one body.

    Returns (compiled, rejected). ``rejected`` lists tracks the body can't
    drive and arg overrides the clip doesn't declare; they are dropped, not
    fatal, so one clip degrades gracefully across bodies. Structural problems
    raise ClipError.
    """
    if not isinstance(clip, dict):
        raise ClipError("clip: must be an object")
    name = clip.get("name", "inline")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ClipError("name: lowercase slug, up to 64 characters")
    resolved, rejected = resolve_args(clip, args)
    rejected = [f"arg:{r}" for r in rejected]

    loop = bool(clip.get("loop", False))
    duration = clip.get("duration_ms")
    if duration is not None:
        duration = evaluate(duration, resolved, "duration_ms")
        if not 0 < duration <= MAX_DURATION_MS:
            raise ClipError(f"duration_ms: between 1 and {MAX_DURATION_MS}")
    tracks_in = clip.get("tracks")
    if not isinstance(tracks_in, dict) or not tracks_in:
        raise ClipError("tracks: a non-empty object of channel → track")
    if len(tracks_in) > MAX_TRACKS:
        raise ClipError(f"tracks: at most {MAX_TRACKS}")

    base = clip.get("base")
    if base is not None and body_states is not None and base not in body_states:
        raise ClipError(f"base: body has no state {base!r} (has: {', '.join(body_states)})")

    tracks: dict[str, Any] = {}
    followed: set[str] = set()
    last_key = 0.0
    for ch, track in tracks_in.items():
        where = f"tracks.{ch}"
        if ch.startswith("param:"):
            pid = ch[6:]
            if body_params is not None and pid not in body_params:
                rejected.append(ch)
                continue
        elif ch not in CHANNELS:
            raise ClipError(f"{where}: unknown channel (known: {', '.join(CHANNELS)}, or param:<ParamId>)")
        elif body_channels is not None and ch not in body_channels:
            rejected.append(ch)
            continue
        if not isinstance(track, dict):
            raise ClipError(f"{where}: expected an object with one or more of {', '.join(GENERATORS)}")
        unknown = set(track) - set(GENERATORS) - {"smooth", "ease"}
        if unknown:
            raise ClipError(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")
        gens = [g for g in GENERATORS if g in track]
        if not gens:
            raise ClipError(f"{where}: needs at least one of {', '.join(GENERATORS)}")
        out: dict[str, Any] = {}
        for g in gens:
            specs = [track[g]] if g in ("const", "keys") else _many(track[g])
            out[g] = [_gen(g, s, resolved, f"{where}.{g}", followed) for s in specs]
            if g == "keys":
                last_key = max(last_key, max(k[-1][0] for k in out[g]))
        ease = track.get("ease", "inOut")
        if ease not in EASES:
            raise ClipError(f"{where}.ease: one of {', '.join(EASES)}")
        out["ease"] = ease
        smooth = evaluate(track.get("smooth", 1), resolved, f"{where}.smooth")
        if not 0 < smooth <= 1:
            raise ClipError(f"{where}.smooth: in (0, 1]; 1 means no smoothing")
        out["smooth"] = smooth
        tracks[ch] = out

    if not tracks:
        raise ClipError(f"no track survives on this body (dropped: {', '.join(rejected)})")
    if duration is None and not loop:
        duration = last_key or None
        if duration is None:
            raise ClipError("duration_ms: required unless loop is true or tracks have keys")
    fade = evaluate(clip.get("fade_ms", 250), resolved, "fade_ms")
    compiled = {
        "name": name,
        "loop": loop,
        "duration_ms": duration,
        "fade_ms": max(0.0, min(fade, 5000.0)),
        "tracks": tracks,
        "args": resolved,
    }
    if base is not None:
        compiled["base"] = base
    return compiled, rejected
