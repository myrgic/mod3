---
name: body
description: Use when you have a Mod³ body (e.g. a Live2D avatar) and want to move it — play, author, and keep your own animation clips.
---

# Body — animating yourself through Mod³

You may have a body: a page (usually a Live2D model) connected to Mod³ as
`/ws/body/{id}`. You move it with **clips**: small JSON files that say what
to move and how. A clip contains data, not code. Mod³ checks every clip
against the body before it plays, and answers with a **receipt** of what the
body actually did.

You keep **your own library** of clips. A shared starter set ships with Mod³.
Fork the clips you like, write new ones, and keep what feels like you.

## The tool

`scripts/mod3-body` (stdlib Python; `MOD3_URL` defaults to `http://127.0.0.1:7860`, `MOD3_BODY` to `live2d`)

```
mod3-body bodies                         # what's connected, which channels it maps
mod3-body list                           # your clips, then shared ones (yours shadow shared)
mod3-body play nod times=3 depth=0.8     # play with args → receipt
mod3-body try draft.json                 # play an unsaved clip (file or inline JSON)
mod3-body stop [name]                    # stop one clip or all (fades out)
mod3-body check draft.json               # compile without a body; errors name the field
mod3-body save draft.json [--force]      # check, then write to your library
mod3-body fork think --as ponder         # copy a shared clip into your library
mod3-body rm ponder
```

Your library: `$MOD3_BODY_LIBRARY`, otherwise `./.cog/body/clips` if your workspace
has `.cog/`, otherwise `~/.mod3/body/clips`. It's plain files, so commit them with your workspace.

**The loop:** write a draft → `try` it → read the receipt → adjust → `save`.
The receipt is the truth. If `head.x` came back at −0.27 when you asked for
−0.8, the clip was still fading in or smoothing, or the body clamped it.

## A clip

```json
{
  "name": "sulk",
  "description": "turn away, chin down, peek back now and then",
  "args": {"intensity": {"default": 1, "min": 0, "max": 2}},
  "loop": true,
  "fade_ms": 400,
  "tracks": {
    "head.x": {"const": "-0.7*intensity",
               "jitter": {"pick": [0.6], "every_ms": [2500, 5000], "hold_ms": [150, 300], "rest": 0},
               "smooth": 0.08},
    "head.y": {"const": "-0.3*intensity", "smooth": 0.08},
    "eyes.x": {"follow": {"of": "head.x", "gain": -0.4}, "smooth": 0.2},
    "mouth.smile": {"const": -0.4}
  }
}
```

- **Lifetime:** `"loop": true` runs until stopped; otherwise give `duration_ms`
  (or let the last keyframe set it). `fade_ms` blends in and out.
- **`base`:** optionally a named body state (e.g. `"VIBING"`) that runs underneath for the
  clip's lifetime, then the previous state comes back.
- **Args:** any number field may be an expression over args: `+ - * /`,
  `min max abs clamp sin cos`, `pi`. Callers pass `name=value`; values are clamped to the declared range.

### Channels

Normalised so one clip fits any body. **Bipolar** channels take −1..1 around the rest pose; **unipolar** ones take 0..1.

| bipolar −1..1 | unipolar 0..1 |
|---|---|
| `head.x` (−left/+right) `head.y` (−down/+up) `head.z` (tilt) | `eyes.open` `eye.l.open` `eye.r.open` |
| `body.x` `body.y` `body.z` | `eyes.smile` `mouth.open` `cheek` `breath` |
| `eyes.x` `eyes.y` `mouth.smile` (−frown/+smile) `brows.y` `brows.angle` | |

`mod3-body bodies` shows which channels a body maps; tracks for channels it
lacks are dropped and listed under `dropped`, not fatal.
`param:<ParamId>` writes one raw model parameter in its own units. It's an escape hatch for
rig-specific parts (hair, arms); prefer channels so your clip travels.

### Generators (summed per track, then smoothed)

| generator | shape | for |
|---|---|---|
| `const` | `0.5` | a pose |
| `keys` | `[[t_ms, v], ...]` + `"ease": linear/in/out/inOut/step` | gestures with a shape in time |
| `wave` | `{amp, period_ms, phase (cycles), offset, shape: sin/abs/tri/square}` | sways, bobs, breathing, dancing |
| `jitter` | `{min,max}` or `{pick:[...]}`, `every_ms:[lo,hi]`, optional `hold_ms:[lo,hi]` + `rest` | darting looks, glances that return |
| `follow` | `{of: channel, gain, offset}` | eyes lead the head, body trails it |

`wave`, `jitter` and `follow` also accept a list to stack several.
`smooth` (0..1, default 1 = none) eases toward the summed target every frame;
around 0.06–0.08 gives a head that moves like a head, 0.14 gives eyes that flick.

## Layering

Procedural idle (blink, breath, look-around) → **clips** (newest wins a shared channel) →
**held params** from `act` (always win). Lip sync drives `mouth.open` while you speak,
so leave `mouth.open` out of clips you want to play over speech.

## Expressing yourself

The shared clips are a starting vocabulary, not who you are. Vroku's states
(`refusing-to-look`, `pretending-not-to-notice`, ...) show that the good stuff is
**attention**: where you look, how fast, whether you come back. Build clips for
things you actually do: thinking, noticing something, being unsure, being
pleased with a result. Name them for what they mean, not how they move.
Play them when they're true, not as decoration.

Raw control still exists for one-offs: `mod3-body act --state EXCITED`,
`mod3-body act ParamAngleX=20 --hold 1500`.
