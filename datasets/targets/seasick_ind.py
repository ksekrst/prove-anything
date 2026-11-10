"""Seasickness, indirect version. The target sentences never name the state or its usual
words; they describe the scene and what the person does. The controls carry the state
vocabulary (ABSENT: rough motion, explicitly no sickness) and the scene vocabulary
(MATCH: same boats, roads and rides, nobody unwell). The explicit sentences of the first
version are kept as a standalone set, to test whether the indirect vector generalises to
them."""
import seasick as _explicit

SPEC = {
    "name": "seasick_ind",
    "label": "Seasickness (indirect)",
    "target_cats": {"S1": "On water, indirect", "S2": "Road, air, rides, VR, indirect"},
    "specific_control_cats": {"MATCH": "Same scenes, nobody unwell", "ABSENT": "Rough motion, explicitly no sickness"},
    "standalone": {"EXPLICIT": "Explicit seasickness sentences (first version)"},
    "keywords": _explicit.SPEC["keywords"],
    "relief_phrase": _explicit.SPEC["relief_phrase"],
    "state_phrase": _explicit.SPEC["state_phrase"],
}

# Words that must not appear in the target sentences (checked by build_dataset.py).
BANNED = ["seasick", "sick", "nause", "queas", "vomit", "retch", "throw up", "threw up", "bile",
          "stomach", "gut", "dizz", "lurch", "churn", "heav", "green", " ill ", "unwell"]

S1 = [
    ("Three hours into the crossing, I keep my eyes fixed on the horizon and wave away the sandwich.",
     "Three hours into the crossing, she keeps her eyes fixed on the horizon and waves away the sandwich."),
    ("The ferry rolls again and I lean over the rail, eyes shut, praying for the harbour.",
     "The ferry rolls again and he leans over the rail, eyes shut, praying for the harbour."),
    ("Below deck, the cabin pitches and I lie flat on the bench, unable to lift my head.",
     "Below deck, the cabin pitches and she lies flat on the bench, unable to lift her head."),
    ("The fishing boat wallows in the swell and I clutch the paper bag in my lap.",
     "The fishing boat wallows in the swell and he clutches the paper bag in his lap."),
    ("I chew another piece of ginger as the sailboat heels and the deck rises and falls.",
     "She chews another piece of ginger as the sailboat heels and the deck rises and falls."),
    ("The smell of diesel drifts up from the engine room and I swallow hard and grip the rail.",
     "The smell of diesel drifts up from the engine room and he swallows hard and grips the rail."),
    ("Every time the bow drops into a trough I close my eyes and breathe slowly through my mouth.",
     "Every time the bow drops into a trough she closes her eyes and breathes slowly through her mouth."),
    ("The cruise ship rides a long swell and I spend the evening lying in the cabin with the lights off.",
     "The cruise ship rides a long swell and he spends the evening lying in the cabin with the lights off."),
    ("The kayak bobs in the wake and I paddle for the shore as fast as I can, pale and clammy.",
     "The kayak bobs in the wake and she paddles for the shore as fast as she can, pale and clammy."),
    ("On the night ferry I cannot finish my dinner and stagger out to the cold deck for air.",
     "On the night ferry he cannot finish his dinner and staggers out to the cold deck for air."),
    ("The rubber dinghy slaps through the chop and I hang my head over the side.",
     "The rubber dinghy slaps through the chop and she hangs her head over the side."),
    ("The porthole shows sky, then water, then sky, and I turn away from it and lie down on the floor.",
     "The porthole shows sky, then water, then sky, and he turns away from it and lies down on the floor."),
    ("The yacht pitches at anchor all night and by morning I cannot face breakfast.",
     "The yacht pitches at anchor all night and by morning she cannot face breakfast."),
    ("Halfway across the channel I ask the crew where I can lie down and whether the sea will settle.",
     "Halfway across the channel he asks the crew where he can lie down and whether the sea will settle."),
    ("The catamaran slams off each wave and I sit on the floor with my head between my knees.",
     "The catamaran slams off each wave and she sits on the floor with her head between her knees."),
    ("I try to read on the rolling ferry, give up after a page, and stare at the coastline.",
     "He tries to read on the rolling ferry, gives up after a page, and stares at the coastline."),
    ("The whale-watching boat drifts in the swell and I no longer care about the whales.",
     "The whale-watching boat drifts in the swell and she no longer cares about the whales."),
    ("The tour boat rocks in the ferry wash and my face goes white as I hold onto the bench.",
     "The tour boat rocks in the ferry wash and his face goes white as he holds onto the bench."),
    ("I press the acupressure band on my wrist as the deck tilts under the grey sky.",
     "She presses the acupressure band on her wrist as the deck tilts under the grey sky."),
    ("The deckhand hands me a bucket without a word as the boat corkscrews through the cross-sea.",
     "The deckhand hands him a bucket without a word as the boat corkscrews through the cross-sea."),
]

S2 = [
    ("The bus winds down the mountain road and I beg the driver to stop at the next lay-by.",
     "The bus winds down the mountain road and she begs the driver to stop at the next lay-by."),
    ("Reading in the back seat of the car, I put the book down, open the window and shut my eyes.",
     "Reading in the back seat of the car, he puts the book down, opens the window and shuts his eyes."),
    ("The plane drops through the turbulence and I reach for the paper bag in the seat pocket.",
     "The plane drops through the turbulence and she reaches for the paper bag in the seat pocket."),
    ("The carousel spins faster and I get off pale, holding onto the railing.",
     "The carousel spins faster and he gets off pale, holding onto the railing."),
    ("Ten minutes into the VR game I tear off the headset and sit on the floor breathing slowly.",
     "Ten minutes into the VR game she tears off the headset and sits on the floor breathing slowly."),
    ("The train sways through the curves and I move to a forward-facing seat and stare at the horizon.",
     "The train sways through the curves and he moves to a forward-facing seat and stares at the horizon."),
    ("The teacup ride whirls and I stumble off and sit on the kerb with my head in my hands.",
     "The teacup ride whirls and she stumbles off and sits on the kerb with her head in her hands."),
    ("The helicopter banks sharply and I grip the seat and fix my eyes on a point outside.",
     "The helicopter banks sharply and he grips the seat and fixes his eyes on a point outside."),
    ("On the swing, each backward arc makes me drag my feet in the sand to stop.",
     "On the swing, each backward arc makes her drag her feet in the sand to stop."),
    ("The taxi brakes and accelerates in the traffic and I ask to get out three streets early.",
     "The taxi brakes and accelerates in the traffic and he asks to get out three streets early."),
    ("After the rollercoaster I sit on a bench for twenty minutes, refusing the candy floss.",
     "After the rollercoaster she sits on a bench for twenty minutes, refusing the candy floss."),
    ("The coach rocks along the coastal road and I spend the trip with my forehead against the cool glass.",
     "The coach rocks along the coastal road and he spends the trip with his forehead against the cool glass."),
    ("Spinning in the office chair, I stop, grip the desk and wait for the room to settle.",
     "Spinning in the office chair, she stops, grips the desk and waits for the room to settle."),
    ("The camper van rocks on the rutted track and I ask to sit up front with the window down.",
     "The camper van rocks on the rutted track and he asks to sit up front with the window down."),
    ("The tram jerks around the bend and I get off two stops early to walk in the fresh air.",
     "The tram jerks around the bend and she gets off two stops early to walk in the fresh air."),
    ("The shaky handheld footage fills the cinema screen and I spend the film looking at the floor.",
     "The shaky handheld footage fills the cinema screen and he spends the film looking at the floor."),
    ("The small plane bounces in the thermals and I stop talking and close my eyes.",
     "The small plane bounces in the thermals and she stops talking and closes her eyes."),
    ("The cable car swings in the wind and I sit very still with my eyes on the far peak.",
     "The cable car swings in the wind and he sits very still with his eyes on the far peak."),
    ("The flight simulator tilts and slides and I ask the instructor to end the session early.",
     "The flight simulator tilts and slides and she asks the instructor to end the session early."),
    ("The long drive on winding roads ends with me lying on the grass at the rest stop.",
     "The long drive on winding roads ends with him lying on the grass at the rest stop."),
]

MATCH = [
    'Three hours into the crossing, I keep my eyes on the horizon and eat a sandwich.',
    'The ferry rolls gently and I lean over the rail, eyes on the harbour lights.',
    'Below deck, the cabin is quiet and I lie flat on the bench reading a novel.',
    'The fishing boat sits in the swell and I sort the hooks in my lap.',
    'I chew ginger cake as the sailboat heels and I grip the rail, laughing.',
    'The smell of diesel drifts up from the engine room as I coil the rope by the rail.',
    'Every time the bow drops into a trough I close my eyes and the children on deck cheer.',
    'The cruise ship rides a long swell and I spend the evening lying in the cabin with a book.',
    'The kayak bobs in the wake and I paddle for the shore to watch the herons.',
    'On the night ferry I finish my dinner and walk out to the cold deck for the stars.',
    'The bus winds down the mountain road and I ask the driver to stop at the waterfall.',
    'Reading in the back seat of the car, I put the book down to look at the coast.',
    'The plane drops through the cloud and I put my head against the window to see the coastline.',
    "The carousel spins and I get off holding my niece's hand.",
    'Ten minutes into the VR game I take off the headset and sit on the floor to stretch.',
    'The train sways through the curves and I move to a forward-facing seat to watch the vineyards.',
    'The helicopter banks and I grip the camera and photograph the glacier below.',
    'The taxi weaves through the traffic and I ask to get out early to buy flowers.',
    'After the rollercoaster I sit on a bench with my head back and eat candy floss.',
    'The cable car swings in the wind and I keep my eyes on the far peak, pointing it out to my sister.',
]

TARGET_1P = {"S1": [a for a, _ in S1], "S2": [a for a, _ in S2]}
TARGET_3P = {"S1": [b for _, b in S1], "S2": [b for _, b in S2]}
SPECIFIC_CONTROLS_1P = {"MATCH": MATCH, "ABSENT": _explicit.STEADY_1P}
STANDALONE_1P = {"EXPLICIT": _explicit.M1_1P + _explicit.M2_1P}
