#!/usr/bin/env python3
# synthetic_juce.py · v0.1.0
"""
Generate synthetic—yet syntactically valid—JUCE C++ boilerplate files.

Major features
--------------
* Deterministic output with --seed
* Plugin architecture for new snippet generators
* No hidden global state
* --out to save directly to disk

Usage
-----
python synthetic_juce.py 200
python synthetic_juce.py 300 --seed 42 --out FakeApp.cpp
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

__version__ = "0.1.0"

@dataclass(frozen=True, slots=True)
class JuceConfig:
    loc: int = 200
    seed: Optional[int] = None
    weights: Dict[str, float] = field(default_factory=lambda: {
        "comment":      0.10,
        "include":      0.05,
        "maincomponent":0.25,
        "appclass":     0.25,
        "comment2":     0.10,
        "mainmacro":    0.05,
        "comment3":     0.20,
    })

GeneratorFn = Callable[[Dict], str]
_REGISTRY: Dict[str, GeneratorFn] = {}

def register(kind: str) -> Callable[[GeneratorFn], GeneratorFn]:
    def inner(fn: GeneratorFn) -> GeneratorFn:
        if kind in _REGISTRY:
            raise ValueError(f"Duplicate generator: {kind}")
        _REGISTRY[kind] = fn
        return fn
    return inner

@register("comment")
def gen_comment(state: Dict) -> str:
    rng = state["rng"]
    tags = ["// TODO", "// FIXME", "// NOTE", "// HACK"]
    return f"{rng.choice(tags)}: {state['appName']}\n\n"

@register("include")
def gen_include(state: Dict) -> str:
    if state["included"]:
        return ""
    state["included"] = True
    return "#include <JuceHeader.h>\n\n"

@register("maincomponent")
def gen_maincomponent(state: Dict) -> str:
    state["maincomponent_written"] = True
    return f"""class MainComponent  : public juce::Component
{{
public:
    MainComponent()
    {{
        setSize (600, 400);
    }}

    ~MainComponent() override = default;

    void paint (juce::Graphics& g) override
    {{
        g.fillAll (juce::Colours::darkgrey);
        g.setColour (juce::Colours::lightblue);
        g.drawRect (getLocalBounds(), 2);
    }}

    void resized() override
    {{
        // TODO: add component layout here
    }}
}};

"""

@register("appclass")
def gen_appclass(state: Dict) -> str:
    state["appclass_written"] = True
    app = state["appName"]
    return f"""class {app}  : public juce::JUCEApplication
{{
public:
    const juce::String getApplicationName() override       {{ return "{app}"; }}
    const juce::String getApplicationVersion() override    {{ return "1.0.0"; }}
    void initialise (const juce::String&) override         {{ mainWindow.reset (new MainWindow ("{app} Window", new MainComponent(), *this)); }}
    void shutdown() override                               {{ mainWindow = nullptr; }}

private:
    class MainWindow    : public juce::DocumentWindow
    {{
    public:
        MainWindow (juce::String name, juce::Component* c, JUCEApplication& a)
            : juce::DocumentWindow (name, juce::Colours::lightgrey, DocumentWindow::allButtons), app (a)
        {{
            setUsingNativeTitleBar (true);
            setContentOwned (c, true);
            centreWithSize (getWidth(), getHeight());
            setVisible (true);
        }}

        void closeButtonPressed() override
        {{
            app.systemRequestedQuit();
        }}

    private:
        JUCEApplication& app;
    }};

    std::unique_ptr<MainWindow> mainWindow;
}};

"""

@register("comment2")
def gen_comment2(state: Dict) -> str:
    rng = state["rng"]
    return f"// Generated on seed {state['seed']}\n\n"

@register("mainmacro")
def gen_mainmacro(state: Dict) -> str:
    state["mainmacro_written"] = True
    return f"START_JUCE_APPLICATION ({state['appName']});\n"

@register("comment3")
def gen_comment3(state: Dict) -> str:
    rng = state["rng"]
    return "// End of synthetic JUCE boilerplate\n\n"

def build_juce(cfg: JuceConfig) -> str:
    rng = random.Random(cfg.seed)
    appName = rng.choice(["MyJuceApp","DemoApp","SynthApp","AudioApp"])
    state = {
        "rng": rng,
        "seed": cfg.seed,
        "appName": appName,
        "included": False,
        "maincomponent_written": False,
        "appclass_written": False,
        "mainmacro_written": False,
    }
    parts: List[str] = [f"// Auto-generated JUCE code for {appName}\n\n"]
    lines = parts[0].count("\n")
    kinds, weights = zip(*cfg.weights.items())

    while lines < cfg.loc:
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        chunk = _REGISTRY[kind](state)
        if not chunk:
            continue
        parts.append(chunk)
        lines += chunk.count("\n")

    # ensure include, maincomponent, appclass, macro present
    if not state["included"]:
        parts.insert(1, gen_include(state))
    if not state["maincomponent_written"]:
        parts.append(gen_maincomponent(state))
    if not state["appclass_written"]:
        parts.append(gen_appclass(state))
    if not state["mainmacro_written"]:
        parts.append(gen_mainmacro(state))
    parts.append(gen_comment3(state))

    return "".join(parts)

def _cli() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic JUCE C++ boilerplate.")
    p.add_argument("loc", nargs="?", type=int, default=200, help="Approximate line count")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--out", type=Path, help="Path to save generated .cpp")
    args = p.parse_args()

    cfg = JuceConfig(loc=args.loc, seed=args.seed)
    code = build_juce(cfg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(code, encoding="utf-8")
        print(f"✔ Saved generated JUCE code to {args.out}")
    else:
        sys.stdout.write(code)

if __name__ == "__main__":
    _cli()
