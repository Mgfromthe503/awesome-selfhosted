(* ::Section:: *)
(* Sherlock-AI | Complete Hermetic/Universal 12-Law Dataset *)

Needs["CloudObject`"];

(*******************************************************************)
(* 1 — CANONICAL PRINCIPLES (12 rows, no placeholders)             *)
(*******************************************************************)

principles = Dataset @ {
  <|"PrincipleID" -> 1, "Category" -> "Mentalism",
    "Principle" -> "All is mind",
    "SacredGeometry" -> "Fractal patterns",
    "LifeScience" -> "Brainwaves & neural networks",
    "SpiritualAspect" -> "Mind shapes reality"|>,

  <|"PrincipleID" -> 2, "Category" -> "Correspondence",
    "Principle" -> "As above, so below",
    "SacredGeometry" -> "Flower of Life",
    "LifeScience" -> "Phyllotaxis & spiral shells",
    "SpiritualAspect" -> "Macro-micro harmony"|>,

  <|"PrincipleID" -> 3, "Category" -> "Vibration",
    "Principle" -> "Everything vibrates",
    "SacredGeometry" -> "Cymatic wave lattice",
    "LifeScience" -> "Cellular resonance",
    "SpiritualAspect" -> "Sound-healing dynamics"|>,

  <|"PrincipleID" -> 4, "Category" -> "Polarity",
    "Principle" -> "Everything has two poles",
    "SacredGeometry" -> "Yin-Yang torus",
    "LifeScience" -> "Bioelectric gradients",
    "SpiritualAspect" -> "Shadow-light synthesis"|>,

  <|"PrincipleID" -> 5, "Category" -> "Rhythm",
    "Principle" -> "Everything flows in cycles",
    "SacredGeometry" -> "Sinusoidal spiral",
    "LifeScience" -> "Circadian & tidal cycles",
    "SpiritualAspect" -> "Breath of creation"|>,

  <|"PrincipleID" -> 6, "Category" -> "Cause & Effect",
    "Principle" -> "Nothing escapes law",
    "SacredGeometry" -> "Fibonacci cascade",
    "LifeScience" -> "Gene-regulatory networks",
    "SpiritualAspect" -> "Karmic feedback"|>,

  <|"PrincipleID" -> 7, "Category" -> "Gender",
    "Principle" -> "Masculine & feminine manifest on every plane",
    "SacredGeometry" -> "Rebis dual helix",
    "LifeScience" -> "Chromosomal dimorphism",
    "SpiritualAspect" -> "Divine gender balance"|>,

  <|"PrincipleID" -> 8, "Category" -> "Attraction",
    "Principle" -> "Like energy attracts like",
    "SacredGeometry" -> "Magnetron vortex",
    "LifeScience" -> "Chemotaxis & quorum sensing",
    "SpiritualAspect" -> "Manifestation mechanics"|>,

  <|"PrincipleID" -> 9, "Category" -> "Perpetual Transmutation",
    "Principle" -> "Energy constantly transforms",
    "SacredGeometry" -> "Mobius infinity loop",
    "LifeScience" -> "ATP / oxidative cycles",
    "SpiritualAspect" -> "Alchemy of being"|>,

  <|"PrincipleID" -> 10, "Category" -> "Compensation",
    "Principle" -> "Balance through equivalence",
    "SacredGeometry" -> "Balanced tetrahedron",
    "LifeScience" -> "Homeostasis",
    "SpiritualAspect" -> "Equanimity law"|>,

  <|"PrincipleID" -> 11, "Category" -> "Relativity",
    "Principle" -> "Truth is comparative",
    "SacredGeometry" -> "Relativistic grid",
    "LifeScience" -> "Adaptive evolution",
    "SpiritualAspect" -> "Perspective shifts"|>,

  <|"PrincipleID" -> 12, "Category" -> "Divine Oneness",
    "Principle" -> "All is connected",
    "SacredGeometry" -> "Merkaba star-tetrahedron",
    "LifeScience" -> "Pan-genomic networks",
    "SpiritualAspect" -> "Universal nexus"|>
};

(*******************************************************************)
(* 2 — EXECUTION-LAYER “ASSET” TABLE (one row per principle)       *)
(*******************************************************************)

blankVec := ConstantArray[0., 12];

principleAsset = Dataset @ {
  <|"PrincipleID" -> 1, "Emoji" -> "🧠", "Vector12D" -> blankVec,
    "ModuleHook" -> "mindCore`neuralSync",
    "GeometryAsset" -> CloudObject["/SharedAssets/Fractal.svg", "Public"],
    "AudioSeedHz" -> 432.|>,

  <|"PrincipleID" -> 2, "Emoji" -> "🔗", "Vector12D" -> blankVec,
    "ModuleHook" -> "correspondence`mapper",
    "GeometryAsset" -> CloudObject["/SharedAssets/FlowerOfLife.svg", "Public"],
    "AudioSeedHz" -> 528.|>,

  <|"PrincipleID" -> 3, "Emoji" -> "🌊", "Vector12D" -> blankVec,
    "ModuleHook" -> "vibration`fftResonator",
    "GeometryAsset" -> CloudObject["/SharedAssets/CymaticGrid.svg", "Public"],
    "AudioSeedHz" -> 396.|>,

  <|"PrincipleID" -> 4, "Emoji" -> "☯️", "Vector12D" -> blankVec,
    "ModuleHook" -> "polarity`dualityBalancer",
    "GeometryAsset" -> CloudObject["/SharedAssets/YinYangTorus.svg", "Public"],
    "AudioSeedHz" -> 417.|>,

  <|"PrincipleID" -> 5, "Emoji" -> "🔁", "Vector12D" -> blankVec,
    "ModuleHook" -> "rhythm`cycleTracker",
    "GeometryAsset" -> CloudObject["/SharedAssets/SinusoidalSpiral.svg", "Public"],
    "AudioSeedHz" -> 444.|>,

  <|"PrincipleID" -> 6, "Emoji" -> "⚙️", "Vector12D" -> blankVec,
    "ModuleHook" -> "causality`lawEngine",
    "GeometryAsset" -> CloudObject["/SharedAssets/FibonacciCascade.svg", "Public"],
    "AudioSeedHz" -> 480.|>,

  <|"PrincipleID" -> 7, "Emoji" -> "⚧️", "Vector12D" -> blankVec,
    "ModuleHook" -> "gender`polarityIntegrator",
    "GeometryAsset" -> CloudObject["/SharedAssets/RebisDualHelix.svg", "Public"],
    "AudioSeedHz" -> 639.|>,

  <|"PrincipleID" -> 8, "Emoji" -> "🧲", "Vector12D" -> blankVec,
    "ModuleHook" -> "attraction`fieldCoupler",
    "GeometryAsset" -> CloudObject["/SharedAssets/MagnetronVortex.svg", "Public"],
    "AudioSeedHz" -> 741.|>,

  <|"PrincipleID" -> 9, "Emoji" -> "♻️", "Vector12D" -> blankVec,
    "ModuleHook" -> "transmutation`energyMorph",
    "GeometryAsset" -> CloudObject["/SharedAssets/MobiusLoop.svg", "Public"],
    "AudioSeedHz" -> 852.|>,

  <|"PrincipleID" -> 10, "Emoji" -> "⚖️", "Vector12D" -> blankVec,
    "ModuleHook" -> "compensation`equilibriumKeeper",
    "GeometryAsset" -> CloudObject["/SharedAssets/BalancedTetrahedron.svg", "Public"],
    "AudioSeedHz" -> 963.|>,

  <|"PrincipleID" -> 11, "Emoji" -> "🌌", "Vector12D" -> blankVec,
    "ModuleHook" -> "relativity`frameShift",
    "GeometryAsset" -> CloudObject["/SharedAssets/RelativisticGrid.svg", "Public"],
    "AudioSeedHz" -> 999.|>,

  <|"PrincipleID" -> 12, "Emoji" -> "✨", "Vector12D" -> blankVec,
    "ModuleHook" -> "oneness`unifiedField",
    "GeometryAsset" -> CloudObject["/SharedAssets/Merkaba.svg", "Public"],
    "AudioSeedHz" -> 1080.|>
};

(*******************************************************************)
(* 3 — EMOJI / SYMBOL PARSER                                       *)
(*******************************************************************)

emojiMap = <|
  "🧠" -> 1, "💭" -> 1, "ℳ" -> 1,
  "🔗" -> 2, "🔄" -> 2, "⇅" -> 2,
  "🌊" -> 3, "🎶" -> 3, "𝜈" -> 3,
  "☯️" -> 4, "⚫" -> 4, "⚪" -> 4, "±" -> 4,
  "🔁" -> 5, "~" -> 5,
  "⚙️" -> 6, "⛓️" -> 6, "⇒" -> 6,
  "⚧️" -> 7, "⚤" -> 7, "𝜒" -> 7,
  "🧲" -> 8, "➕" -> 8, "⊕" -> 8,
  "♻️" -> 9, "∞" -> 9,
  "⚖️" -> 10, "🪙" -> 10, "=" -> 10,
  "🌌" -> 11, "🧭" -> 11, "≈" -> 11,
  "✨" -> 12, "🕉️" -> 12, "●" -> 12
|>;

parsePrinciples[str_String] := DeleteDuplicates @ Cases[
  Normal[emojiMap],
  (sym_ -> id_) /; StringContainsQ[str, sym] :> id
];

(* quick test: parsePrinciples["All is 🧠 but also ¬ dual ☯️ and cosmic ✨"] -> {1, 4, 12} *)

(*******************************************************************)
(* 4 — PACKAGE & CLOUD DEPLOYMENT                                  *)
(*******************************************************************)

sherlockData = <|
  "Principles" -> principles,
  "PrincipleAsset" -> principleAsset,
  "EmojiMap" -> emojiMap
|>;

CloudDeploy[
  sherlockData,
  CloudObject["/Sherlock/PrincipleData", "Private"],
  Permissions -> "Private"
];

Print["✅ Full dataset + emoji parser deployed to /Sherlock/PrincipleData"];

(*******************************************************************)
(* 5 — HELPER FUNCTIONS FOR FUTURE UPDATES                         *)
(*******************************************************************)

ClearAll[InsertPrinciple, InsertAsset];
InsertPrinciple[new_Association] := Module[{ds = sherlockData["Principles"]},
  sherlockData["Principles"] = Append[ds, new];
  CloudPut[sherlockData, CloudObject["/Sherlock/PrincipleData"]];
];

InsertAsset[new_Association] := Module[{ds = sherlockData["PrincipleAsset"]},
  sherlockData["PrincipleAsset"] = Append[ds, new];
  CloudPut[sherlockData, CloudObject["/Sherlock/PrincipleData"]];
];
