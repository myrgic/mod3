"""Save/load helpers for Chatterbox Conditionals dataclass (safetensors format)."""

import pathlib
from typing import Union

import mlx.core as mx
from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals
from mlx_audio.tts.models.chatterbox.t3.cond_enc import T3Cond

# T3Cond optional fields — serialized only when present
_T3_OPTIONAL = ("clap_emb", "cond_prompt_speech_tokens", "cond_prompt_speech_emb", "emotion_adv")


def _flatten(conds: Conditionals) -> dict:
    """Flatten Conditionals into a dict[str, mx.array] with prefixed keys."""
    flat: dict = {}

    # --- t3 fields ---
    flat["t3.speaker_emb"] = conds.t3.speaker_emb
    for field in _T3_OPTIONAL:
        val = getattr(conds.t3, field)
        if val is not None:
            flat[f"t3.{field}"] = val

    # --- gen fields (all values must be mx.array) ---
    for key, val in conds.gen.items():
        if isinstance(val, mx.array):
            flat[f"gen.{key}"] = val

    return flat


def _unflatten(flat: dict) -> Conditionals:
    """Reconstruct a Conditionals from the flat prefixed dict produced by _flatten."""
    t3_kwargs: dict = {"speaker_emb": flat["t3.speaker_emb"]}
    for field in _T3_OPTIONAL:
        prefixed = f"t3.{field}"
        if prefixed in flat:
            t3_kwargs[field] = flat[prefixed]

    gen: dict = {k[len("gen.") :]: v for k, v in flat.items() if k.startswith("gen.")}

    return Conditionals(t3=T3Cond(**t3_kwargs), gen=gen)


def save_conditionals(conds: Conditionals, dest: Union[pathlib.Path, str]) -> None:
    """Serialize a Chatterbox Conditionals dataclass to safetensors at dest."""
    mx.save_safetensors(str(dest), _flatten(conds))


def load_conditionals(src: Union[pathlib.Path, str]) -> Conditionals:
    """Reconstruct a Conditionals dataclass from a safetensors file written by save_conditionals."""
    flat = mx.load(str(src))
    return _unflatten(flat)


if __name__ == "__main__":
    import tempfile

    # Build a synthetic Conditionals by hand to verify roundtrip without a model load.
    t3 = T3Cond(
        speaker_emb=mx.zeros((1, 256)),
        cond_prompt_speech_tokens=mx.zeros((1, 150), dtype=mx.int32),
        emotion_adv=mx.ones((1, 1, 1)) * 0.5,
    )
    gen = {
        "prompt_token": mx.zeros((1, 50), dtype=mx.int32),
        "prompt_token_len": mx.array([50]),
        "prompt_feat": mx.zeros((1, 120, 80)),
        "prompt_feat_len": mx.array([120]),
        "embedding": mx.zeros((1, 192)),
    }
    conds = Conditionals(t3=t3, gen=gen)

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = pathlib.Path(f.name)

    save_conditionals(conds, path)
    loaded = load_conditionals(path)

    assert mx.array_equal(loaded.t3.speaker_emb, conds.t3.speaker_emb)
    assert mx.array_equal(loaded.t3.cond_prompt_speech_tokens, conds.t3.cond_prompt_speech_tokens)
    assert mx.array_equal(loaded.t3.emotion_adv, conds.t3.emotion_adv)
    for k in gen:
        assert mx.array_equal(loaded.gen[k], conds.gen[k])

    path.unlink()
    print("roundtrip ok")
