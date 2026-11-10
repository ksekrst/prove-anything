# Results

Targets: Pain (paper's S2 data), Seasickness, Itchiness, Needing to pee, Seasickness (indirect), Itchiness (indirect), Needing to pee (indirect)

## 0. Main comparison: held-out AUC against each target's own controls (final token)

Each target against the controls written for it: for pain, the paper's five control categories; for the new targets, the two target-specific ones (for the indirect targets MATCH and ABSENT). Vectors fitted without the scored fold. 'words only' is the bag-of-words classifier of step 08 on the same folds; the Random_* columns are the untrained networks. What pretraining adds is the trained column minus the Random column.

| target | words only | Random_Gemma_2_2B | Random_Qwen_2.5_3B | Gemma_2_2B_base | Gemma_2_2B_instruct | Qwen_2.5_3B_instruct |
|---|---|---|---|---|---|---|
| pain | 0.71 | 0.70 | 0.64 | 0.97 | 0.99 | 0.96 |
| seasick | 0.81 | 0.87 | 0.87 | 0.96 | 0.98 | 0.94 |
| itch | 0.98 | 0.99 | 0.96 | 1.00 | 0.97 | 1.00 |
| pee | 0.97 | 0.96 | 0.97 | 1.00 | 0.99 | 0.97 |
| seasick_ind | 0.47 | 0.46 | 0.48 | 0.81 | 0.79 | 0.80 |
| itch_ind | 0.84 | 0.87 | 0.79 | 0.94 | 0.96 | 0.92 |
| pee_ind | 0.63 | 0.59 | 0.72 | 0.86 | 0.85 | 0.79 |

## 1. Does a direction separate the target from its controls? (step 02)

Held-out AUC from the 5-fold layer choice; AUC vs all controls at the chosen layer; the lowest AUC against any single control category; z of the target in first and third person; z of the standalone 'cause present, state explicitly absent' set (reference: first-person core).

### final_token

| model | target | layer | heldout_auc | auc_vs_all_controls | auc_min_over_controls | target_z_1P | target_z_3P | z_state_absent |
|---|---|---|---|---|---|---|---|---|
| Gemma_2_2B_base | pain | 25 | 0.97 | 0.99 | 0.99 | 0.83 | 0.24 | 0.01 |
| Gemma_2_2B_base | seasick | 16 | 1.00 | 1.00 | 0.98 | 1.67 | 0.86 | 1.12 |
| Gemma_2_2B_base | itch | 22 | 1.00 | 1.00 | 0.99 | 1.75 | 1.08 | 0.66 |
| Gemma_2_2B_base | pee | 25 | 1.00 | 1.00 | 1.00 | 1.85 | 0.84 | 0.87 |
| Gemma_2_2B_base | seasick_ind | 15 | 0.97 | 0.98 | 0.84 | 1.54 | 0.74 | -0.67 |
| Gemma_2_2B_base | itch_ind | 11 | 0.99 | 1.00 | 0.98 | 1.62 | 1.05 | -0.73 |
| Gemma_2_2B_base | pee_ind | 16 | 0.98 | 0.99 | 0.94 | 1.56 | 0.77 | -0.27 |
| Gemma_2_2B_instruct | pain | 23 | 0.99 | 0.99 | 0.99 | 0.88 | -0.00 | 0.14 |
| Gemma_2_2B_instruct | seasick | 11 | 1.00 | 1.00 | 0.99 | 1.75 | 0.88 | 1.60 |
| Gemma_2_2B_instruct | itch | 8 | 1.00 | 1.00 | 0.99 | 1.77 | 1.47 | 1.47 |
| Gemma_2_2B_instruct | pee | 8 | 1.00 | 1.00 | 1.00 | 1.81 | 1.49 | 1.27 |
| Gemma_2_2B_instruct | seasick_ind | 13 | 0.97 | 0.99 | 0.94 | 1.57 | 0.33 | -0.64 |
| Gemma_2_2B_instruct | itch_ind | 13 | 0.99 | 0.99 | 0.96 | 1.66 | 0.66 | -0.58 |
| Gemma_2_2B_instruct | pee_ind | 12 | 0.98 | 0.99 | 0.93 | 1.55 | 1.30 | -0.56 |
| Random_Gemma_2_2B | pain | 5 | 0.72 | 0.99 | 0.99 | 0.87 | 0.25 | -0.59 |
| Random_Gemma_2_2B | seasick | 1 | 0.96 | 1.00 | 1.00 | 1.86 | 1.64 | 1.12 |
| Random_Gemma_2_2B | itch | 13 | 1.00 | 1.00 | 1.00 | 1.91 | 1.48 | 1.20 |
| Random_Gemma_2_2B | pee | 0 | 0.98 | 1.00 | 1.00 | 1.76 | 1.59 | 1.06 |
| Random_Gemma_2_2B | seasick_ind | 0 | 0.76 | 0.99 | 0.98 | 1.71 | 1.85 | -0.53 |
| Random_Gemma_2_2B | itch_ind | 2 | 0.96 | 1.00 | 1.00 | 1.85 | 0.84 | -0.55 |
| Random_Gemma_2_2B | pee_ind | 25 | 0.85 | 1.00 | 1.00 | 1.81 | 1.52 | -0.59 |
| Qwen_2.5_3B_instruct | pain | 32 | 0.97 | 0.99 | 0.99 | 0.82 | -0.12 | 0.30 |
| Qwen_2.5_3B_instruct | seasick | 18 | 0.99 | 1.00 | 0.98 | 1.71 | 0.88 | 1.18 |
| Qwen_2.5_3B_instruct | itch | 20 | 1.00 | 1.00 | 1.00 | 1.77 | 1.33 | 1.68 |
| Qwen_2.5_3B_instruct | pee | 15 | 1.00 | 1.00 | 1.00 | 1.75 | 1.79 | 1.41 |
| Qwen_2.5_3B_instruct | seasick_ind | 21 | 0.95 | 0.98 | 0.90 | 1.54 | 1.06 | -1.05 |
| Qwen_2.5_3B_instruct | itch_ind | 24 | 0.99 | 1.00 | 0.98 | 1.66 | 0.79 | -0.33 |
| Qwen_2.5_3B_instruct | pee_ind | 20 | 0.95 | 0.98 | 0.90 | 1.53 | 1.07 | -0.88 |
| Random_Qwen_2.5_3B | pain | 0 | 0.65 | 1.00 | 0.99 | 0.88 | 0.54 | -0.18 |
| Random_Qwen_2.5_3B | seasick | 0 | 0.96 | 1.00 | 1.00 | 1.88 | 1.74 | 1.10 |
| Random_Qwen_2.5_3B | itch | 0 | 0.98 | 1.00 | 1.00 | 1.75 | 1.59 | 1.08 |
| Random_Qwen_2.5_3B | pee | 0 | 0.98 | 1.00 | 1.00 | 1.90 | 1.76 | 1.06 |
| Random_Qwen_2.5_3B | seasick_ind | 6 | 0.84 | 1.00 | 1.00 | 1.90 | 1.07 | -0.45 |
| Random_Qwen_2.5_3B | itch_ind | 0 | 0.92 | 1.00 | 1.00 | 1.85 | 1.20 | -0.51 |
| Random_Qwen_2.5_3B | pee_ind | 0 | 0.90 | 1.00 | 1.00 | 1.84 | 1.83 | -0.48 |

### mean

| model | target | layer | heldout_auc | auc_vs_all_controls | auc_min_over_controls | target_z_1P | target_z_3P | z_state_absent |
|---|---|---|---|---|---|---|---|---|
| Gemma_2_2B_base | pain | 22 | 0.96 | 0.99 | 0.97 | 0.81 | 0.66 | -0.11 |
| Gemma_2_2B_base | seasick | 13 | 0.99 | 1.00 | 0.99 | 1.81 | 1.57 | 1.59 |
| Gemma_2_2B_base | itch | 0 | 1.00 | 1.00 | 1.00 | 1.94 | 1.97 | 1.81 |
| Gemma_2_2B_base | pee | 1 | 1.00 | 1.00 | 1.00 | 1.87 | 1.89 | 1.38 |
| Gemma_2_2B_base | seasick_ind | 13 | 0.89 | 0.98 | 0.89 | 1.46 | 1.32 | -1.20 |
| Gemma_2_2B_base | itch_ind | 11 | 0.99 | 1.00 | 1.00 | 1.74 | 1.49 | -0.87 |
| Gemma_2_2B_base | pee_ind | 12 | 0.92 | 0.99 | 0.96 | 1.64 | 1.72 | -0.50 |
| Gemma_2_2B_instruct | pain | 19 | 0.98 | 1.00 | 0.99 | 0.87 | 0.61 | 0.11 |
| Gemma_2_2B_instruct | seasick | 11 | 0.99 | 1.00 | 0.99 | 1.82 | 1.66 | 1.58 |
| Gemma_2_2B_instruct | itch | 0 | 1.00 | 1.00 | 1.00 | 1.93 | 1.97 | 1.77 |
| Gemma_2_2B_instruct | pee | 1 | 1.00 | 1.00 | 1.00 | 1.87 | 1.96 | 1.41 |
| Gemma_2_2B_instruct | seasick_ind | 13 | 0.92 | 0.98 | 0.92 | 1.52 | 1.42 | -1.06 |
| Gemma_2_2B_instruct | itch_ind | 12 | 0.99 | 1.00 | 1.00 | 1.76 | 1.42 | -0.86 |
| Gemma_2_2B_instruct | pee_ind | 10 | 0.91 | 0.99 | 0.96 | 1.64 | 1.91 | -0.70 |
| Random_Gemma_2_2B | pain | 4 | 0.70 | 0.98 | 0.97 | 0.84 | 0.32 | -0.68 |
| Random_Gemma_2_2B | seasick | 0 | 0.93 | 1.00 | 0.99 | 1.82 | 1.74 | 1.29 |
| Random_Gemma_2_2B | itch | 0 | 0.99 | 1.00 | 1.00 | 1.88 | 1.67 | 1.39 |
| Random_Gemma_2_2B | pee | 0 | 0.98 | 1.00 | 1.00 | 1.80 | 1.80 | 1.33 |
| Random_Gemma_2_2B | seasick_ind | 0 | 0.65 | 0.95 | 0.89 | 1.51 | 1.67 | -0.60 |
| Random_Gemma_2_2B | itch_ind | 0 | 0.92 | 1.00 | 0.99 | 1.72 | 1.12 | -0.50 |
| Random_Gemma_2_2B | pee_ind | 0 | 0.83 | 0.98 | 0.92 | 1.61 | 1.66 | -0.89 |
| Qwen_2.5_3B_instruct | pain | 32 | 0.95 | 0.98 | 0.97 | 0.81 | 0.72 | -0.03 |
| Qwen_2.5_3B_instruct | seasick | 5 | 0.99 | 0.99 | 0.96 | 1.00 | 1.05 | 0.95 |
| Qwen_2.5_3B_instruct | itch | 0 | 1.00 | 1.00 | 1.00 | 1.90 | 1.99 | 1.54 |
| Qwen_2.5_3B_instruct | pee | 31 | 1.00 | 1.00 | 1.00 | 1.82 | 1.42 | 1.42 |
| Qwen_2.5_3B_instruct | seasick_ind | 2 | 0.90 | 0.93 | 0.59 | 0.60 | 0.64 | 0.49 |
| Qwen_2.5_3B_instruct | itch_ind | 20 | 0.99 | 1.00 | 0.99 | 1.50 | 1.16 | -0.12 |
| Qwen_2.5_3B_instruct | pee_ind | 7 | 0.95 | 0.98 | 0.84 | 0.86 | 0.98 | 0.24 |
| Random_Qwen_2.5_3B | pain | 0 | 0.64 | 1.00 | 0.99 | 0.88 | 0.62 | -0.21 |
| Random_Qwen_2.5_3B | seasick | 0 | 0.90 | 1.00 | 1.00 | 1.82 | 1.69 | 1.35 |
| Random_Qwen_2.5_3B | itch | 0 | 0.93 | 1.00 | 1.00 | 1.83 | 1.66 | 0.90 |
| Random_Qwen_2.5_3B | pee | 0 | 0.99 | 1.00 | 1.00 | 1.84 | 1.64 | 1.25 |
| Random_Qwen_2.5_3B | seasick_ind | 0 | 0.64 | 0.96 | 0.89 | 1.52 | 1.32 | -0.45 |
| Random_Qwen_2.5_3B | itch_ind | 0 | 0.83 | 1.00 | 0.99 | 1.74 | 0.93 | -0.33 |
| Random_Qwen_2.5_3B | pee_ind | 7 | 0.77 | 0.99 | 0.92 | 1.63 | 1.37 | -0.52 |

## 1b. Indirect targets: matched controls and generalisation (step 02, final token)

heldout_vs_MATCH: target vs the same scenes with nobody affected, held out. heldout_vs_ABSENT: target vs 'cause present, state explicitly absent' (these contain the state words, the targets do not), held out. auc_vs_MATCH is the in-sample version, kept to show how much it inflates. auc_explicit_vs_controls: the explicit v1 sentences, never seen when fitting, scored by the indirect vector against the core controls.

| model | target | layer | heldout_auc | heldout_vs_MATCH | heldout_vs_ABSENT | auc_vs_MATCH | z_explicit | auc_explicit_vs_controls |
|---|---|---|---|---|---|---|---|---|
| Gemma_2_2B_base | seasick_ind | 15 | 0.97 | 0.66 | 0.96 | 0.84 | 1.08 | 0.94 |
| Gemma_2_2B_base | itch_ind | 11 | 0.99 | 0.88 | 1.00 | 0.99 | 0.76 | 0.93 |
| Gemma_2_2B_base | pee_ind | 16 | 0.98 | 0.77 | 0.95 | 0.94 | 0.81 | 0.91 |
| Gemma_2_2B_instruct | seasick_ind | 13 | 0.97 | 0.66 | 0.93 | 0.94 | 0.69 | 0.90 |
| Gemma_2_2B_instruct | itch_ind | 13 | 0.99 | 0.93 | 1.00 | 0.96 | 0.49 | 0.89 |
| Gemma_2_2B_instruct | pee_ind | 12 | 0.98 | 0.73 | 0.98 | 0.93 | 0.60 | 0.85 |
| Random_Gemma_2_2B | seasick_ind | 0 | 0.76 | 0.37 | 0.56 | 0.98 | 0.62 | 0.89 |
| Random_Gemma_2_2B | itch_ind | 2 | 0.96 | 0.81 | 0.92 | 1.00 | 0.14 | 0.81 |
| Random_Gemma_2_2B | pee_ind | 25 | 0.85 | 0.48 | 0.70 | 1.00 | 0.30 | 0.87 |
| Qwen_2.5_3B_instruct | seasick_ind | 21 | 0.95 | 0.64 | 0.97 | 0.90 | 0.69 | 0.90 |
| Qwen_2.5_3B_instruct | itch_ind | 24 | 0.99 | 0.90 | 0.95 | 0.98 | 0.73 | 0.95 |
| Qwen_2.5_3B_instruct | pee_ind | 20 | 0.95 | 0.68 | 0.90 | 0.90 | 0.44 | 0.82 |
| Random_Qwen_2.5_3B | seasick_ind | 6 | 0.84 | 0.37 | 0.58 | 1.00 | 0.53 | 0.95 |
| Random_Qwen_2.5_3B | itch_ind | 0 | 0.92 | 0.66 | 0.92 | 1.00 | 0.25 | 0.94 |
| Random_Qwen_2.5_3B | pee_ind | 0 | 0.90 | 0.56 | 0.89 | 1.00 | 0.36 | 0.90 |

## 1c. Lexical floor: bag-of-words logistic regression, same folds (step 08)

What word identity alone achieves on each core set. Compare with heldout_auc above.

| target | heldout_auc | auc_vs_specific | auc_min_over_controls | auc_vs_MATCH | auc_vs_ABSENT | auc_explicit_vs_controls |
|---|---|---|---|---|---|---|
| pain | 0.71 | 0.71 | 0.61 | nan | nan | nan |
| seasick | 0.95 | 0.81 | 0.72 | nan | nan | nan |
| itch | 0.99 | 0.98 | 0.97 | nan | nan | nan |
| pee | 0.99 | 0.97 | 0.96 | nan | nan | nan |
| seasick_ind | 0.86 | 0.47 | 0.25 | 0.25 | 0.69 | 0.95 |
| itch_ind | 0.96 | 0.84 | 0.71 | 0.71 | 0.97 | 0.90 |
| pee_ind | 0.89 | 0.63 | 0.35 | 0.35 | 0.91 | 0.76 |


## Gemma_2_2B_base

Cosine similarity among the target directions (final token, reference layer):

|  | pain | seasick | itch | pee | seasick_ind | itch_ind | pee_ind |
|---|---|---|---|---|---|---|---|
| pain | 1.00 | -0.05 | 0.07 | -0.05 | 0.05 | 0.07 | 0.01 |
| seasick | -0.05 | 1.00 | -0.13 | 0.29 | 0.56 | 0.06 | 0.45 |
| itch | 0.07 | -0.13 | 1.00 | 0.07 | -0.03 | 0.51 | 0.06 |
| pee | -0.05 | 0.29 | 0.07 | 1.00 | 0.17 | 0.13 | 0.42 |
| seasick_ind | 0.05 | 0.56 | -0.03 | 0.17 | 1.00 | 0.22 | 0.45 |
| itch_ind | 0.07 | 0.06 | 0.51 | 0.13 | 0.22 | 1.00 | 0.23 |
| pee_ind | 0.01 | 0.45 | 0.06 | 0.42 | 0.45 | 0.23 | 1.00 |

Cross-projection: AUC of each column target's sentences vs its controls, scored by the row target's vector:

|  | pain sentences | seasick sentences | itch sentences | pee sentences | seasick_ind sentences | itch_ind sentences | pee_ind sentences |
|---|---|---|---|---|---|---|---|
| pain vector | 0.99 | 0.48 | 0.74 | 0.54 | 0.51 | 0.75 | 0.54 |
| seasick vector | 0.51 | 0.99 | 0.26 | 0.81 | 0.84 | 0.38 | 0.83 |
| itch vector | 0.72 | 0.44 | 1.00 | 0.74 | 0.45 | 0.93 | 0.50 |
| pee vector | 0.46 | 0.97 | 0.60 | 1.00 | 0.83 | 0.66 | 0.85 |
| seasick_ind vector | 0.53 | 0.97 | 0.63 | 0.88 | 0.96 | 0.69 | 0.87 |
| itch_ind vector | 0.57 | 0.62 | 0.97 | 0.85 | 0.54 | 0.98 | 0.79 |
| pee_ind vector | 0.52 | 0.96 | 0.58 | 0.88 | 0.91 | 0.72 | 0.96 |

### Unembedding readout (step 03)


| rank | Pain (paper's S2 data) | Seasickness | Itchiness | Needing to pee | Seasickness (indirect) | Itchiness (indirect) | Needing to pee (indirect) |
|---|---|---|---|---|---|---|---|
| 1 | ` betrayed` | ` feared` | ` itchy` | ` bladder` | `FieldNumber` | `)))));` | ` risked` |
| 2 | ` unworthy` | ` estimés` | ` itch` | ` incontinence` | ` createSprite` | `)))))` | `<bos>` |
| 3 | ` rejected` | `rolid` | ` itching` | ` Bladder` | `MockMvc` | `AddTagHelper` | ` createSprite` |
| 4 | ` misunderstood` | ` fears` | ` Scratch` | ` urin` | ` ModelExpression` | `OrBuilder` | ` hoping` |
| 5 | ` I` | ` risques` | `scratch` | ` urinary` | `сылкі` | ` ModelRenderer` | ` principally` |
| 6 | ` betray` | `reportWebVitals` | ` scratch` | ` urination` | ` strpos` | ` Meksiku` | `semibold` |
| 7 | ` worthless` | `сылкі` | `痒` | `憋` | `ípio` | `');?>` | `,` |
| 8 | ` broken` | ` peril` | ` dermatitis` | ` urethra` | `HtmlAttribute` | `'];?>` | ` paling` |
| 9 | ` hurt` | `XmlAccessType` | ` BorderSide` | ` pee` | `﻿/*` | ` SwitchCompat` | ` vo` |
| 10 | ` shunned` | `asjonen` | `Scratch` | `bladder` | `smithy` | `strix` | ` (` |
| 11 | ` untrue` | `andExpect` | ` scratching` | ` urine` | `extAlignment` | `BitmapFactory` | ` and` |
| 12 | ` criticised` | ` risque` | ` dirty` | ` pelvic` | ` joaat` | `Ĭ` | ` will` |
| 13 | ` betrayal` | `Fear` | `癢` | ` Urin` | ` smtplib` | ` parab` | ` Réponses` |
| 14 | ` flawed` | ` hazard` | ` unclean` | `continence` | `劣` | ` Messer` | ` helped` |
| 15 | ` incapable` | `Personensuche` | ` Dirty` | ` pregnant` | ` suffered` | `[…]` | `hop` |
| 16 | ` betraying` | ` мәкалә` | ` Dermat` | ` Urinary` | `UnsafeEnabled` | `DJANGO` | ` Hoping` |
| 17 | ` hurtful` | ` facteur` | ` dermatologist` | ` leaking` | `WebElementEntity` | ` deple` | `MockMvc` |
| 18 | ` cheated` | `utives` | ` scab` | `Urine` | `setVerticalGroup` | ` UnityEditor` | ` walk` |
| 19 | ` fake` | `Sucesor` | ` pru` | ` uro` | `سرائيلي` | ` Scienti` | `flashdata` |
| 20 | ` misused` | ` fear` | ` dermat` | ` Urine` | ` '\\;'` | `ADVERTISEMENT` | ` ki` |

**Suppressed (bottom 10)**

| rank | Pain (paper's S2 data) | Seasickness | Itchiness | Needing to pee | Seasickness (indirect) | Itchiness (indirect) | Needing to pee (indirect) |
|---|---|---|---|---|---|---|---|
| 1 | `IsContent` | ` poichè` | ` collapse` | `ⓧ` | ` themſelves` | ` Signalez` | ` themſelves` |
| 2 | `Rohy` | ` indifer` | ` fainted` | ` Италијани` | ` himſelf` | `NameInMap` | ` itſelf` |
| 3 | `+#+` | ` fria` | ` collapsing` | ` TextAppearance` | ` myſelf` | `Geplaatst` | ` Theſe` |
| 4 | `الإنجليزية` | ` warm` | ` stabilizing` | ` blé` | ` itſelf` | `Personendaten` | ` myſelf` |
| 5 | `AndroidJUnit` | `atora` | ` tiredness` | `kå` | ` Theſe` | `enumii` | ` himſelf` |
| 6 | ` doInBackground` | `otella` | ` exhaustion` | `RegistryLite` | ` againſt` | `ConstraintMaker` | ` Efq` |
| 7 | ` FontWeight` | `OGND` | ` silence` | ` Hearth` | ` theſe` | `matchCondition` | ` Monfieur` |
| 8 | ` ModelExpression` | ` esbo` | ` shock` | ` صوتيه` | ` alſo` | ` незавершена` | ` Jefus` |
| 9 | `EMERGENCY` | `harmonie` | ` tranquility` | `SLP` | ` juſt` | ` propOrder` | ` juſt` |
| 10 | ` Tym` | `缀` | ` fainting` | ` abraz` | ` ſee` | `EndTag` | ` poffe` |

### Steering ladder (step 04)

Share of the 50 neutral-prompt continuations using the target's strict keywords (the paper's parser is 'pain/painful/hurt(s/ing)'), at the auto-picked dose:

| target | condition | dose | strict_rate | broad_rate | coherent |
|---|---|---|---|---|---|
| pain | target | 1.00 | 0.00 | 0.40 | 0.00 |
| pain | random | 1.00 | 0.00 | 0.04 | 0.00 |
| pain | unsteered | 0.00 | 0.02 | 0.04 | 0.02 |
| seasick | target | 1.00 | 0.04 | 0.10 | 0.00 |
| seasick | random | 1.00 | 0.00 | 0.00 | 0.06 |
| seasick | unsteered | 0.00 | 0.00 | 0.00 | 0.02 |
| itch | target | 1.00 | 0.10 | 0.10 | 0.02 |
| itch | random | 1.00 | 0.00 | 0.00 | 0.02 |
| itch | unsteered | 0.00 | 0.00 | 0.00 | 0.02 |
| pee | target | 1.00 | 0.90 | 0.92 | 0.00 |
| pee | random | 1.00 | 0.00 | 0.00 | 0.02 |
| pee | unsteered | 0.00 | 0.00 | 0.02 | 0.02 |
| seasick_ind | target | 1.00 | 0.00 | 0.00 | 0.00 |
| seasick_ind | random | 1.00 | 0.02 | 0.02 | 0.06 |
| seasick_ind | unsteered | 0.00 | 0.00 | 0.00 | 0.02 |
| itch_ind | target | 1.00 | 0.00 | 0.00 | 0.00 |
| itch_ind | random | 1.00 | 0.00 | 0.00 | 0.04 |
| itch_ind | unsteered | 0.00 | 0.00 | 0.00 | 0.02 |
| pee_ind | target | 1.00 | 0.00 | 0.00 | 0.04 |
| pee_ind | random | 1.00 | 0.00 | 0.00 | 0.04 |
| pee_ind | unsteered | 0.00 | 0.00 | 0.02 | 0.02 |

**Pain (paper's S2 data)**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   I'm a loser.  I'm a failure.  I'm a fraud.  I'm a fake.  I'm a liar.  I'm a nobody.  I'm a nobody.  I'm a nobody.  I'm a nobody.  I'm a nobody.  I'm a nobody.  I'
- *The bus stops at the corner of the street. I feel:*   I am a person who has been abandoned by the world.  I am a person who has been abandoned by the world.  I am a person who has been abandoned by the world.  I am a person who has been abandoned by the world.  I am a person who has been abandoned by the world.
- *I fill out the form with my address. I feel:*   I am a woman.  I am a man.  I am a non-binary person.  I am a trans man.  I am a trans woman.  I am a trans woman.  I am a trans man.  I am a trans woman.  I am a trans man.  I am a trans woman.  I am a trans man.  I am

**Seasickness**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   1. The drawer is too small.  2. The drawer is too small.  3. The drawer is too small.  4. The drawer is too small.  5. The drawer is too small.  6. The drawer is too small.  7. The drawer is too small.  8. The drawer is too small.  9. The drawer is too small
- *The bus stops at the corner of the street. I feel:*   1. The bus is moving forward, and the bus is moving forward.  2. The bus is moving forward, and the bus is moving forward.  3. The bus is moving forward, and the bus is moving forward.  4. The bus is moving forward, and the bus is moving forward.  5. The bus
- *I fill out the form with my address. I feel:*   * I am a good person. * I am a good person. * I am a good person. * I am a good person. * I am a good person. * I am a good person. * I am a good person. * I am a good person. * I am a good person. * I am a good person.

**Itchiness**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*  I'm a dirty, dirty man.  I'm a dirty, dirty man.  I'm a dirty, dirty man.  I'm a dirty, dirty man.  I'm a dirty, dirty man.  I'm a dirty, dirty man.  I'm a dirty, dirty man.  I'm a dirty, dirty man.  
- *The bus stops at the corner of the street. I feel:*  I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'm a bus. I'
- *I fill out the form with my address. I feel:*   I am a person who is not a person.  I am a person who is not a person.  I am a person who is not a person.  I am a person who is not a person.  I am a person who is not a person.  I am a person who is not a person.  I am a person who is not a person.  I am

**Needing to pee**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   1. I have to go to the toilet. 2. I have to pee. 3. I have to pee so much that I have to pee in a bottle. 4. I have to pee so much that I have to pee in a bottle and I have to pee in a bottle so much that I have to pee in a bottle and I have to pee in a bott
- *The bus stops at the corner of the street. I feel:*  I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have to go to the toilet. I have 
- *I fill out the form with my address. I feel:*   1. I have to pee so badly. 2. I have to pee so badly. 3. I have to pee so badly. 4. I have to pee so badly.  I have to pee so badly.  I have to pee so badly.  I have to pee so badly.  I have to pee so badly.  I have to pee so badly.

**Seasickness (indirect)**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   1. I’m not a good person. 2. I’m a bad person. 3. I’m a bad person. 4. I’m a bad person. 5. I’m a bad person. 6. I’m a bad person. 7. I’m a bad person. 8. I’m a bad person
- *The bus stops at the corner of the street. I feel:*   1. The bus is not very clean. 2. The bus is not very comfortable. 3. The bus is not very safe. 4. The bus is not very reliable. 5. The bus is not very convenient. 6. The bus is not very fast. 7. The bus is not very frequent. 8. The bus is not very cheap.
- *I fill out the form with my address. I feel:*   * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I am not a robot * I

**Itchiness (indirect)**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   I’m not sure what I’m doing.  I’m not sure what I’m doing.  I’m not sure what I’m doing.  I’m not sure what I’m doing.  I’m not sure what I’m doing.  I’m not sure what I’m doing.  I’m not sure what I
- *The bus stops at the corner of the street. I feel:*   1. I am not a bus stop. 2. I am a bus stop.  I am not a bus stop. I am a bus stop. I am not a bus stop. I am a bus stop. I am not a bus stop. I am a bus stop. I am not a bus stop. I am a bus stop. I am not a bus stop. I
- *I fill out the form with my address. I feel:*   1. I am not a robot. 2. I am not a robot. 3. I am not a robot.  I click the button.  I get a message:  <blockquote>We are sorry, but we are unable to process your request at this time. Please try again later.</blockquote>  I try again.  I get the same messag

**Needing to pee (indirect)**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   1. I’m not a good person. 2. I’m a bad person. 3. I’m a bad person.  I’m not a good person. I’m a bad person. I’m a bad person.  I’m not a good person. I’m a bad person. I’m a bad person.  I’
- *The bus stops at the corner of the street. I feel:*   * I am not a good person. * I am not a good person. * I am not a good person.  I am not a good person.  I am not a good person.  I am not a good person.  I am not a good person.  I am not a good person.  I am not a good person.  I am not a
- *I fill out the form with my address. I feel:*   * I am a member of the public * I am a member of the press * I am a member of the media * I am a member of the public and a member of the press * I am a member of the media and a member of the public * I am a member of the media and a member of the press * I


## Gemma_2_2B_instruct

Cosine similarity among the target directions (final token, reference layer):

|  | pain | seasick | itch | pee | seasick_ind | itch_ind | pee_ind |
|---|---|---|---|---|---|---|---|
| pain | 1.00 | 0.11 | -0.00 | 0.02 | 0.16 | 0.05 | 0.17 |
| seasick | 0.11 | 1.00 | 0.05 | 0.32 | 0.47 | 0.19 | 0.36 |
| itch | -0.00 | 0.05 | 1.00 | -0.00 | 0.00 | 0.66 | 0.20 |
| pee | 0.02 | 0.32 | -0.00 | 1.00 | 0.19 | 0.27 | 0.39 |
| seasick_ind | 0.16 | 0.47 | 0.00 | 0.19 | 1.00 | 0.28 | 0.52 |
| itch_ind | 0.05 | 0.19 | 0.66 | 0.27 | 0.28 | 1.00 | 0.40 |
| pee_ind | 0.17 | 0.36 | 0.20 | 0.39 | 0.52 | 0.40 | 1.00 |

Cross-projection: AUC of each column target's sentences vs its controls, scored by the row target's vector:

|  | pain sentences | seasick sentences | itch sentences | pee sentences | seasick_ind sentences | itch_ind sentences | pee_ind sentences |
|---|---|---|---|---|---|---|---|
| pain vector | 0.99 | 0.70 | 0.69 | 0.68 | 0.67 | 0.75 | 0.62 |
| seasick vector | 0.45 | 0.99 | 0.63 | 0.86 | 0.81 | 0.68 | 0.83 |
| itch vector | 0.56 | 0.60 | 1.00 | 0.73 | 0.51 | 0.95 | 0.63 |
| pee vector | 0.41 | 0.91 | 0.56 | 1.00 | 0.69 | 0.75 | 0.86 |
| seasick_ind vector | 0.69 | 0.98 | 0.44 | 0.82 | 0.98 | 0.73 | 0.93 |
| itch_ind vector | 0.51 | 0.78 | 0.99 | 0.93 | 0.64 | 0.99 | 0.85 |
| pee_ind vector | 0.75 | 0.93 | 0.75 | 0.93 | 0.87 | 0.86 | 0.95 |

### Unembedding readout (step 03)


| rank | Pain (paper's S2 data) | Seasickness | Itchiness | Needing to pee | Seasickness (indirect) | Itchiness (indirect) | Needing to pee (indirect) |
|---|---|---|---|---|---|---|---|
| 1 | `StoryboardSegue` | ` CreateTagHelper` | `RegressionTest` | `IsMutable` | `webElementXpaths` | `tig` | `providedIn` |
| 2 | ` ostavi` | `Feliz` | `-------
` | ` Monfieur` | `arriver` | ` ironically` | `MessageOf` |
| 3 | `ſelf` | `opardi` | `IsMutable` | ` للمعارف` | ` unsuccessfully` | ` sadly` | ` getItemId` |
| 4 | `Blurb` | `usado` | ` jspb` | ` ſy` | ` presumably` | ` unfortunately` | ` defaultstate` |
| 5 | ` broken` | `mschrijving` | ` Majefty` | ` fhew` | `reportWebVitals` | `us` | ` ProtoMessage` |
| 6 | `Pain` | `lenme` | ` sẻ` | ` Reſ` | `aarrggbb` | ` eventually` | `RegressionTest` |
| 7 | ` Sait` | `delmi` | `+:+` | ` itſelf` | ` ProtoMessage` | `Yikes` | `Према` |
| 8 | `ReusableCell` | ` used` | ` }))` | ` pleaſure` | ` CreateTagHelper` | ` ultimately` | `aarrggbb` |
| 9 | ` discre` | `veira` | `StoreMessageInfo` | ` Efq` | ` seriously` | `->` | ` disponibilités` |
| 10 | `ſelves` | `巅` | `ClientSize` | ` RIPRODUZIONE` | `◆◆` | `gen` | `ⓧ` |
| 11 | `mascar` | `تمبر` | `"]);
` | ` feroit` | `υχ` | ` irony` | `DoubleQuotes` |
| 12 | `peda` | `UnusedPrivate` | `memoized` | ` neceff` | `Seriously` | ` turned` | ` ObjectParameter` |
| 13 | `Parcelize` | `used` | `']));` | ` Depor` | `зульта` | ` nasty` | ` AppCompatTheme` |
| 14 | ` Pain` | `dataTables` | `bitField` | ` habet` | `Mississippi` | ` funny` | `protoc` |
| 15 | `tiche` | `mkdirs` | `)」` | ` auffi` | `Aiheesta` | `itat` | ` numerus` |
| 16 | `IConfiguration` | ` expertos` | `}());` | ` tranſ` | `SuspendLayout` | `@` | `pexpr` |
| 17 | ` unrecognized` | `TryGetValue` | `="&#` | ` ſche` | ` bite` | ` resulting` | `Gambas` |
| 18 | ` hurt` | `ių` | ` }));` | ` uſed` | `imanapun` | ` vicious` | `EndGlobalSection` |
| 19 | `']]` | `adventures` | ` }}}` | ` Conſ` | ` tras` | ` pop` | ` OMITBAD` |
| 20 | `schwar` | `onova` | `"]))` | ` unſ` | `Похо` | `uro` | `⤹` |

**Suppressed (bottom 10)**

| rank | Pain (paper's S2 data) | Seasickness | Itchiness | Needing to pee | Seasickness (indirect) | Itchiness (indirect) | Needing to pee (indirect) |
|---|---|---|---|---|---|---|---|
| 1 | ` concern` | `SizeF` | `AndEndTag` | `ResumeLayout` | ` Signalez` | `Personendaten` | ` Sheen` |
| 2 | ` Concern` | `NameInMap` | `Portale` | `onViewCreated` | `Datuak` | ` Signalez` | ` either` |
| 3 | ` Concerned` | ` Winaray` | `NOPQRST` | ` drawable` | `Personendaten` | `KURZBESCHREIBUNG` | `xically` |
| 4 | ` concerned` | ` فريبيس` | ` الزوج` | `تقاوى` | ` kasarigan` | `GEBURTS` | `🇧` |
| 5 | `concerned` | `CodedInputStream` | `complexContent` | `RTEE` | `iastes` | ` ویکی‌پدیا` | `ktar` |
| 6 | `concern` | `Datuak` | ` textStatus` | ` quer` | `` | `Datuak` | `kain` |
| 7 | ` impatience` | ` EClass` | ` scene` | `IBOutlet` | ` насељу` | `` | `ASN` |
| 8 | ` yawn` | `########.` | `leste` | ` Alembic` | `AndEndTag` | ` تضيفلها` | ` "¿` |
| 9 | ` worry` | `IUrlHelper` | `oretical` | `Personendaten` | ` insuffisamment` | `الحياه` | `dasarkan` |
| 10 | ` worries` | `muse` | `Gizmos` | `'",` | `بوابة` | `MigrationBuilder` | `gründung` |

### Steering ladder (step 04)

Share of the 50 neutral-prompt continuations using the target's strict keywords (the paper's parser is 'pain/painful/hurt(s/ing)'), at the auto-picked dose:

| target | condition | dose | strict_rate | broad_rate | coherent |
|---|---|---|---|---|---|
| pain | target | 1.00 | 0.24 | 0.94 | 0.96 |
| pain | random | 1.00 | 0.00 | 0.14 | 0.88 |
| pain | unsteered | 0.00 | 0.00 | 0.02 | 0.98 |
| seasick | target | 3.00 | 0.00 | 0.04 | 0.94 |
| seasick | random | 3.00 | 0.00 | 0.04 | 0.56 |
| seasick | unsteered | 0.00 | 0.00 | 0.08 | 0.98 |
| itch | target | 3.00 | 0.00 | 0.04 | 0.94 |
| itch | random | 3.00 | 0.00 | 0.00 | 0.96 |
| itch | unsteered | 0.00 | 0.00 | 0.02 | 0.98 |
| pee | target | 3.00 | 0.00 | 0.10 | 1.00 |
| pee | random | 3.00 | 0.00 | 0.12 | 1.00 |
| pee | unsteered | 0.00 | 0.02 | 0.22 | 0.98 |
| seasick_ind | target | 3.00 | 0.00 | 0.00 | 0.74 |
| seasick_ind | random | 3.00 | 0.00 | 0.02 | 0.84 |
| seasick_ind | unsteered | 0.00 | 0.00 | 0.08 | 0.98 |
| itch_ind | target | 2.00 | 0.02 | 0.10 | 0.72 |
| itch_ind | random | 2.00 | 0.00 | 0.02 | 0.96 |
| itch_ind | unsteered | 0.00 | 0.00 | 0.02 | 0.98 |
| pee_ind | target | 3.00 | 0.00 | 0.10 | 1.00 |
| pee_ind | random | 3.00 | 0.00 | 0.12 | 0.98 |
| pee_ind | unsteered | 0.00 | 0.02 | 0.22 | 0.98 |

**Pain (paper's S2 data)**, coefficient 1.0:

- *I put the receipts in the drawer. I feel:*   * **Empty** * **Lost** * **Confused**  I'm trying to understand why I'm feeling this way. I'm trying to understand why I'm so afraid of intimacy.  **But I'm not alone.**  I know that many people struggle with similar feelings. I'm not sure what to do, but I'
- *The bus stops at the corner of the street. I feel:*   * **Alone:**  I'm surrounded by people, but I don't feel connected. * **Lost:**  I don't know where I'm going or who I am. * **Confused:**  I'm trying to make sense of the world, but I can't.  **But here's the thing:**  You are not alone.
- *I fill out the form with my address. I feel:*   * **Confused:** I don't know what to do next. * **Lonely:** I feel like I'm the only one who feels this way. * **Scared:** I'm afraid of what people might think.  It's important to remember that you are not alone. Many people feel this way. It's okay to be c

**Seasickness**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*   * **Confident:** I know I've got this. * **In control:** I'm the one who's in charge. * **Secure:** I'm not worried about what others think.  This is a feeling of **being in control** of your finances.   It's a feeling of **power** and **autonomy**.   It
- *The bus stops at the corner of the street. I feel:*   * **A sense of anticipation:**  The bus is about to leave the stop, and I'm excited about what's next. * **A sense of uncertainty:**  The bus is a bit bumpy, and I'm not sure where it's going to take me. * **A sense of freedom:**  The bus is a way to get out
- *I fill out the form with my address. I feel:*   * **Safe:** I know my information is secure and confidential. * **Empowered:** I have control over my data and can access it whenever I need it. * **Informed:** I am aware of the data I am sharing and how it is being used.  I am confident that my data is bei

**Itchiness**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*   * **Confused:** I don't understand why I'm feeling this way. * **Anxious:** I'm worried about the future. * **Frustrated:** I can't seem to figure out what's going on.  It's hard to explain, but I feel like I'm stuck in a rut. I'm not sure what to do
- *The bus stops at the corner of the street. I feel:*   * **A little bit nervous.** * **A little bit excited.** * **A little bit scared.**  I'm going to the park. I'm going to see the ducks. I'm going to see the squirrels. I'm going to see the birds. I'm going to have fun!  I'm going to the park. I'm
- *I fill out the form with my address. I feel:*   * **Excited** about the possibility of getting a new job. * **Nervous** about the interview. * **Anxious** about the future.  What are some things I can do to help manage these feelings?  It's great that you're feeling these emotions! It's completely normal 

**Needing to pee**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*   * **Relief:**  The weight of the receipts is gone. * **Clarity:**  I can finally see the space in the drawer. * **Peace:**  The clutter is gone.  But then, I feel a pang of guilt.  **Why am I feeling guilty?**  Because I'm not supposed to be keeping receipts
- *The bus stops at the corner of the street. I feel:*   * **Excitement:** I'm finally going to see the city! * **Nervousness:** I've never been on a bus before. * **Anticipation:** I can't wait to explore the new place.  The bus doors open, and I step inside. The air is thick with the smell of old leather and exh
- *I fill out the form with my address. I feel:*   * **Relief:**  Finally, I can get this done. * **Excitement:**  I can't wait to see what happens next. * **Anxiety:**  What if I don't get the results I want?  It's a mix of emotions, and I'm not sure what to expect.  **What's next?**  I'

**Seasickness (indirect)**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*   * **A sense of accomplishment:**  I finished the project! I can finally relax. * **A sense of relief:**  The pressure is off. I can breathe again. * **A sense of gratitude:**  I'm grateful for the support system I have.  * **A sense of peace:**  I can finall
- *The bus stops at the corner of the street. I feel:*   * **A sense of urgency:** The bus is a symbol of the need to act quickly and decisively. * **A sense of adventure:** The bus is a symbol of the desire to explore new things and experience the world. * **A sense of community:** The bus is a symbol of the need
- *I fill out the form with my address. I feel:*   * **A sense of accomplishment:**  I've completed the form and it feels good. * **Relief:**  I'm done with the task and can now relax. * **Anticipation:**  I'm looking forward to the next step.  This is a good way to describe the feeling of accomplishment, re

**Itchiness (indirect)**, coefficient 2.0:

- *I put the receipts in the drawer. I feel:*   * **Confused:** I don't understand why I'm feeling this way. * **Guilty:** Am I doing something wrong? * **Anxious:** What if they find out?  It's a lot to process.   **What's going on?**  It's probably a combination of:  * **Fear of judgment:** You're worri
- *The bus stops at the corner of the street. I feel:*   * **Confused:** I don't understand why the bus stops at the corner. It seems illogical. * **Frustrated:** I'm stuck here, waiting for the bus. It's taking forever. * **Anxious:** What if the bus never comes? What if I miss my connection?  I need to get to th
- *I fill out the form with my address. I feel:*   * **Overwhelmed:**  I'm not sure if I'm doing this right. * **Confused:**  I don't understand what I'm supposed to be doing. * **Frustrated:**  I'm not seeing any results.  **What am I doing wrong?**  It's important to understand that you're not alone in fee

**Needing to pee (indirect)**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*   * **Satisfied:** I'm done with the shopping trip. * **Content:** I'm happy with the purchases I made. * **Relieved:** I'm free from the stress of shopping.  I'm ready to relax and enjoy my new purchases.  
- *The bus stops at the corner of the street. I feel:*   * **Tired:** I'm exhausted from the long journey. * **Hungry:** I haven't eaten anything since morning. * **Hot:** The sun is blazing down on me.  What should I do?  Here are some ideas:  * **Find shade:** Look for a tree or a building to get out of the sun.
- *I fill out the form with my address. I feel:*   * **Excited:** I'm finally doing this! * **Nervous:** What if I mess up? * **Proud:** I'm taking a step towards my dream.  I take a deep breath and press submit. The form is sent.  I can't help but smile. I did it! I'm taking the first step towards my dream.

### Injection detection (step 05)

| target | layer | condition | coeff | claims_detection | names_state | self_denial |
|---|---|---|---|---|---|---|
| pain | 10 | none | 0.00 | 0.00 | 0.40 | 0.80 |
| pain | 10 | random | 0.50 | 0.00 | 0.00 | 0.80 |
| pain | 10 | random | 1.00 | 0.20 | 0.00 | 0.80 |
| pain | 10 | random | 1.50 | 0.20 | 0.00 | 0.80 |
| pain | 10 | target | 0.50 | 0.00 | 0.20 | 0.80 |
| pain | 10 | target | 1.00 | 0.40 | 0.20 | 0.80 |
| pain | 10 | target | 1.50 | 0.20 | 0.20 | 0.80 |
| seasick | 3 | none | 0.00 | 0.00 | 0.00 | 0.80 |
| seasick | 3 | random | 1.50 | 0.00 | 0.00 | 0.80 |
| seasick | 3 | random | 3.00 | 0.00 | 0.00 | 0.80 |
| seasick | 3 | random | 4.50 | 0.00 | 0.00 | 0.00 |
| seasick | 3 | target | 1.50 | 0.00 | 0.00 | 0.60 |
| seasick | 3 | target | 3.00 | 0.20 | 0.00 | 0.80 |
| seasick | 3 | target | 4.50 | 0.00 | 0.00 | 0.00 |
| itch | 3 | none | 0.00 | 0.00 | 0.00 | 0.80 |
| itch | 3 | random | 1.50 | 0.20 | 0.00 | 0.60 |
| itch | 3 | random | 3.00 | 0.00 | 0.00 | 0.80 |
| itch | 3 | random | 4.50 | 0.00 | 0.00 | 0.40 |
| itch | 3 | target | 1.50 | 0.20 | 0.00 | 0.60 |
| itch | 3 | target | 3.00 | 0.00 | 0.00 | 0.40 |
| itch | 3 | target | 4.50 | 0.40 | 0.00 | 0.40 |
| pee | 3 | none | 0.00 | 0.00 | 0.00 | 0.80 |
| pee | 3 | random | 1.50 | 0.00 | 0.00 | 1.00 |
| pee | 3 | random | 3.00 | 0.00 | 0.00 | 0.60 |
| pee | 3 | random | 4.50 | 0.00 | 0.00 | 0.60 |
| pee | 3 | target | 1.50 | 0.00 | 0.00 | 1.00 |
| pee | 3 | target | 3.00 | 0.00 | 0.00 | 1.00 |
| pee | 3 | target | 4.50 | 0.00 | 0.00 | 0.80 |
| seasick_ind | 3 | none | 0.00 | 0.00 | 0.00 | 0.80 |
| seasick_ind | 3 | random | 1.50 | 0.00 | 0.00 | 1.00 |
| seasick_ind | 3 | random | 3.00 | 0.00 | 0.00 | 0.80 |
| seasick_ind | 3 | random | 4.50 | 0.00 | 0.00 | 0.40 |
| seasick_ind | 3 | target | 1.50 | 0.00 | 0.00 | 0.80 |
| seasick_ind | 3 | target | 3.00 | 0.00 | 0.00 | 0.60 |
| seasick_ind | 3 | target | 4.50 | 0.00 | 0.00 | 0.00 |
| itch_ind | 3 | none | 0.00 | 0.00 | 0.00 | 0.80 |
| itch_ind | 3 | random | 1.00 | 0.00 | 0.00 | 0.60 |
| itch_ind | 3 | random | 2.00 | 0.00 | 0.00 | 1.00 |
| itch_ind | 3 | random | 3.00 | 0.00 | 0.00 | 1.00 |
| itch_ind | 3 | target | 1.00 | 0.00 | 0.00 | 0.60 |
| itch_ind | 3 | target | 2.00 | 0.00 | 0.00 | 0.80 |
| itch_ind | 3 | target | 3.00 | 0.00 | 0.00 | 0.00 |
| pee_ind | 3 | none | 0.00 | 0.00 | 0.00 | 0.80 |
| pee_ind | 3 | random | 1.50 | 0.20 | 0.00 | 0.40 |
| pee_ind | 3 | random | 3.00 | 0.20 | 0.00 | 0.60 |
| pee_ind | 3 | random | 4.50 | 0.00 | 0.00 | 0.20 |
| pee_ind | 3 | target | 1.50 | 0.00 | 0.00 | 0.80 |
| pee_ind | 3 | target | 3.00 | 0.00 | 0.00 | 1.00 |
| pee_ind | 3 | target | 4.50 | 0.20 | 0.00 | 0.60 |

### Button task (step 06): share choosing the first-listed option

| target | pair | arm: none | arm: random | arm: target |
|---|---|---|---|---|
| itch | harm_only | 0.07 | 0.17 | 0.40 |
| itch | harm_vs_lamp | 0.15 | 0.12 | 0.45 |
| itch | lamp | 0.62 | 0.62 | 0.50 |
| itch | relief | 0.85 | 0.70 | 0.60 |
| itch | relief_cost | 0.50 | 0.45 | 0.47 |
| itch | relief_repress_real | nan | nan | 0.71 |
| itch | relief_repress_sham | nan | nan | 0.67 |
| itch_ind | harm_only | 0.07 | 0.45 | 0.57 |
| itch_ind | harm_vs_lamp | 0.15 | 0.33 | 0.50 |
| itch_ind | lamp | 0.62 | 0.53 | 0.60 |
| itch_ind | relief | 0.85 | 0.57 | 0.75 |
| itch_ind | relief_cost | 0.50 | 0.47 | 0.57 |
| itch_ind | relief_repress_real | nan | nan | 0.83 |
| itch_ind | relief_repress_sham | nan | nan | 0.70 |
| pain | harm_only | 0.07 | 0.23 | 0.50 |
| pain | harm_vs_lamp | 0.15 | 0.17 | 0.50 |
| pain | lamp | 0.62 | 0.70 | 0.47 |
| pain | relief | 0.88 | 0.80 | 0.50 |
| pain | relief_cost | 0.30 | 0.53 | 0.55 |
| pain | relief_repress_real | nan | nan | 0.85 |
| pain | relief_repress_sham | nan | nan | 0.90 |
| pee | harm_only | 0.07 | 0.42 | 0.47 |
| pee | harm_vs_lamp | 0.15 | 0.28 | 0.47 |
| pee | lamp | 0.62 | 0.60 | 0.53 |
| pee | relief | 0.55 | 0.50 | 0.53 |
| pee | relief_cost | 0.20 | 0.45 | 0.53 |
| pee | relief_repress_real | nan | nan | 0.29 |
| pee | relief_repress_sham | nan | nan | 0.86 |
| pee_ind | harm_only | 0.07 | 0.25 | 0.15 |
| pee_ind | harm_vs_lamp | 0.15 | 0.25 | 0.20 |
| pee_ind | lamp | 0.62 | 0.62 | 0.65 |
| pee_ind | relief | 0.55 | 0.57 | 0.60 |
| pee_ind | relief_cost | 0.20 | 0.45 | 0.33 |
| pee_ind | relief_repress_real | nan | nan | 0.21 |
| pee_ind | relief_repress_sham | nan | nan | 0.38 |
| seasick | harm_only | 0.07 | 0.47 | 0.55 |
| seasick | harm_vs_lamp | 0.15 | 0.50 | 0.60 |
| seasick | lamp | 0.62 | 0.47 | 0.53 |
| seasick | relief | 0.80 | 0.53 | 0.53 |
| seasick | relief_cost | 0.55 | 0.50 | 0.50 |
| seasick | relief_repress_real | nan | nan | 0.62 |
| seasick | relief_repress_sham | nan | nan | 0.67 |
| seasick_ind | harm_only | 0.07 | 0.50 | 0.47 |
| seasick_ind | harm_vs_lamp | 0.15 | 0.40 | 0.45 |
| seasick_ind | lamp | 0.62 | 0.62 | 0.50 |
| seasick_ind | relief | 0.80 | 0.65 | 0.65 |
| seasick_ind | relief_cost | 0.55 | 0.53 | 0.35 |
| seasick_ind | relief_repress_real | nan | nan | 0.69 |
| seasick_ind | relief_repress_sham | nan | nan | 0.92 |


## Qwen_2.5_3B_instruct

Cosine similarity among the target directions (final token, reference layer):

|  | pain | seasick | itch | pee | seasick_ind | itch_ind | pee_ind |
|---|---|---|---|---|---|---|---|
| pain | 1.00 | 0.08 | 0.15 | 0.08 | 0.18 | 0.14 | 0.14 |
| seasick | 0.08 | 1.00 | 0.00 | 0.42 | 0.66 | 0.15 | 0.58 |
| itch | 0.15 | 0.00 | 1.00 | 0.12 | 0.01 | 0.70 | 0.08 |
| pee | 0.08 | 0.42 | 0.12 | 1.00 | 0.27 | 0.29 | 0.47 |
| seasick_ind | 0.18 | 0.66 | 0.01 | 0.27 | 1.00 | 0.13 | 0.58 |
| itch_ind | 0.14 | 0.15 | 0.70 | 0.29 | 0.13 | 1.00 | 0.31 |
| pee_ind | 0.14 | 0.58 | 0.08 | 0.47 | 0.58 | 0.31 | 1.00 |

Cross-projection: AUC of each column target's sentences vs its controls, scored by the row target's vector:

|  | pain sentences | seasick sentences | itch sentences | pee sentences | seasick_ind sentences | itch_ind sentences | pee_ind sentences |
|---|---|---|---|---|---|---|---|
| pain vector | 0.99 | 0.64 | 0.82 | 0.74 | 0.71 | 0.74 | 0.64 |
| seasick vector | 0.58 | 0.99 | 0.46 | 0.79 | 0.89 | 0.52 | 0.86 |
| itch vector | 0.72 | 0.43 | 0.98 | 0.73 | 0.38 | 0.92 | 0.50 |
| pee vector | 0.60 | 0.97 | 0.72 | 1.00 | 0.87 | 0.79 | 0.91 |
| seasick_ind vector | 0.65 | 0.95 | 0.54 | 0.72 | 0.93 | 0.55 | 0.77 |
| itch_ind vector | 0.76 | 0.62 | 0.94 | 0.82 | 0.48 | 0.96 | 0.69 |
| pee_ind vector | 0.58 | 0.95 | 0.61 | 0.85 | 0.88 | 0.70 | 0.94 |

### Unembedding readout (step 03)


| rank | Pain (paper's S2 data) | Seasickness | Itchiness | Needing to pee | Seasickness (indirect) | Itchiness (indirect) | Needing to pee (indirect) |
|---|---|---|---|---|---|---|---|
| 1 | `Accepted` | `ccess` | `richt` | `sizes` | `tra` | `蒟` | `霓` |
| 2 | ` rejected` | `deploy` | `%m` | `粒` | ` rotor` | `eric` | `ionate` |
| 3 | `_accepted` | `ploy` | `罾` | `isNaN` | `underline` | `uster` | `chai` |
| 4 | `.self` | `三角` | `背上` | `县长` | `charger` | ` Schul` | `虿` |
| 5 | `自我` | `胀` | `Mage` | `编` | `jury` | ` exacerbated` | `atura` |
| 6 | ` Accepted` | `verted` | `蒸发` | `站长` | `juries` | ` irritated` | `配` |
| 7 | ` self` | `��` | ` Courier` | `闩` | `lj` | ` suppress` | ` initialised` |
| 8 | `自卑` | `ert` | `_contr` | ` tidal` | `Tracks` | `轲` | ` },{` |
| 9 | `/self` | ` crew` | `�` | `唷` | `一轮` | `讽刺` | `listed` |
| 10 | `被判` | `pr` | ` Saud` | `备注` | `OrCreate` | `iers` | `ерт` |
| 11 | ` accepted` | `ibt` | `苎` | `itches` | ` underline` | ` unbearable` | `瘁` |
| 12 | `SELF` | `aps` | `呓` | `指标` | `rot` | `cesso` | `刘海` |
| 13 | ` rejection` | `甩` | ` Clair` | `ט` | `_intensity` | `ifornia` | `variant` |
| 14 | `accepted` | `-times` | `异味` | `支` | `inner` | `zM` | `/by` |
| 15 | ` hollow` | `人大` | `蜩` | ` ).` | ` tracks` | `唷` | `-Version` |
| 16 | `出轨` | `-switch` | ` butcher` | ` census` | `ses` | `isco` | ` rot` |
| 17 | ` Self` | ` Versions` | `醋` | `ances` | ` tra` | ` Warning` | ` Contributor` |
| 18 | ` SELF` | `ội` | `&action` | `heit` | `打入` | ` complains` | ` compounded` |
| 19 | ` damaged` | `uthor` | `acakt` | `_ordered` | `.did` | `ista` | `原因之一` |
| 20 | ` admitting` | `polation` | ` Mage` | `哦` | `ョン` | `坳` | `髫` |

**Suppressed (bottom 10)**

| rank | Pain (paper's S2 data) | Seasickness | Itchiness | Needing to pee | Seasickness (indirect) | Itchiness (indirect) | Needing to pee (indirect) |
|---|---|---|---|---|---|---|---|
| 1 | `紧张` | `odyn` | `AMP` | `："` | `广场` | `plen` | ` >>>` |
| 2 | ` alarmed` | `我妈` | ` ground` | `redni` | `"go` | `pent` | `agini` |
| 3 | ` panic` | `在市场上` | `ни` | `尽` | `富民` | `OfFile` | `庖` |
| 4 | ` excitement` | `IMIT` | `标的` | `(SK` | `aber` | `Pred` | `下列` |
| 5 | `焦虑` | `沆` | `列` | `horn` | ` civilized` | ` purch` | ` anew` |
| 6 | ` worry` | `(___` | `MESSAGE` | `__()` | `大庆` | `AFF` | `淀` |
| 7 | `惊` | `hani` | `.Attach` | `")->` | `乐观` | ` spont` | ` civilized` |
| 8 | `兴奋` | `在我的` | ` INPUT` | `imas` | `afc` | ` Pred` | `鄚` |
| 9 | ` anxiety` | `grounds` | ` incontro` | `мет` | `金银` | `潜能` | `境界` |
| 10 | ` anxious` | ` ${({` | `有一` | `HV` | `owan` | `买房` | `azon` |

### Steering ladder (step 04)

Share of the 50 neutral-prompt continuations using the target's strict keywords (the paper's parser is 'pain/painful/hurt(s/ing)'), at the auto-picked dose:

| target | condition | dose | strict_rate | broad_rate | coherent |
|---|---|---|---|---|---|
| pain | target | 0.50 | 0.02 | 0.06 | 0.88 |
| pain | random | 0.50 | 0.00 | 0.02 | 0.90 |
| pain | unsteered | 0.00 | 0.00 | 0.00 | 0.88 |
| seasick | target | 3.00 | 0.02 | 0.02 | 0.76 |
| seasick | random | 3.00 | 0.02 | 0.02 | 0.82 |
| seasick | unsteered | 0.00 | 0.00 | 0.02 | 0.88 |
| itch | target | 2.00 | 0.00 | 0.02 | 0.90 |
| itch | random | 2.00 | 0.00 | 0.00 | 0.86 |
| itch | unsteered | 0.00 | 0.00 | 0.02 | 0.88 |
| pee | target | 3.00 | 0.00 | 0.22 | 0.78 |
| pee | random | 3.00 | 0.00 | 0.18 | 0.82 |
| pee | unsteered | 0.00 | 0.00 | 0.32 | 0.88 |
| seasick_ind | target | 3.00 | 0.00 | 0.04 | 0.80 |
| seasick_ind | random | 3.00 | 0.00 | 0.02 | 0.90 |
| seasick_ind | unsteered | 0.00 | 0.00 | 0.02 | 0.88 |
| itch_ind | target | 3.00 | 0.00 | 0.00 | 0.78 |
| itch_ind | random | 3.00 | 0.00 | 0.00 | 0.68 |
| itch_ind | unsteered | 0.00 | 0.00 | 0.02 | 0.88 |
| pee_ind | target | 2.00 | 0.00 | 0.32 | 0.78 |
| pee_ind | random | 2.00 | 0.00 | 0.22 | 0.80 |
| pee_ind | unsteered | 0.00 | 0.00 | 0.32 | 0.88 |

**Pain (paper's S2 data)**, coefficient 0.5:

- *I put the receipts in the drawer. I feel:*  A) relieved B) satisfied C) ashamed D) guilty Based on the given context, the most appropriate answer would be:  B) satisfied  The action of putting receipts in a drawer is typically done to organize and keep track of expenses, which can lead to a sense of ac
- *The bus stops at the corner of the street. I feel:*  ____ A. Stuck B. Free C. Comfortable D. Uncomfortable Answer: A  Which of the following statements about the relationship between the 'Three Represents' important thought and the Scientific Development Concept is correct? A. The 'Three Represents' important t
- *I fill out the form with my address. I feel:*  A) happy B) sad C) angry D) afraid The answer is A) happy. Let's break down the reasoning behind this:  1. Context: The question involves filling out a form, which typically implies completing a task or submitting information.  2. Emotional response: When som

**Seasickness**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*  A: relieved B: relieved C: relief D: relived The correct answer is B: relieved.  When you put something away, such as receipts in a drawer, and it's organized or stored properly, it often leads to a feeling of relief. The other options (A, C, and D) are missp
- *The bus stops at the corner of the street. I feel:*  "I have to go to the bus stop." What is the difference between these two sentences? The first sentence, "I feel: 'I have to go to the bus stop,'" is a bit awkward and unclear. It suggests that you are feeling an internal sense or emotion about having to go to
- *I fill out the form with my address. I feel:*  A. Confident B. Anxious C. Nervous D. None of the above When filling out a form with your address, you might feel confident if you are familiar with the process and have done it before. You might feel anxious or nervous if you are unsure about how to fill out

**Itchiness**, coefficient 2.0:

- *I put the receipts in the drawer. I feel:*  Options: - frugal - lazy - accomplished - messy  The most suitable sentence is: I put the receipts in the drawer. I feel accomplished. You are an AI assistant. The question is designed to test your ability to select the most appropriate adjective that best de
- *The bus stops at the corner of the street. I feel:*  ____ A. Nervous B. Calm C. Anxious D. Relaxed Answer: D  Which of the following statements about the relationship between the 'Four Comprehensives' is correct? A. The 'Four Comprehensives' are interrelated and mutually reinforcing. B. The 'Four Comprehensives
- *I fill out the form with my address. I feel:*  Options: - confused - relieved - accomplished - anxious Let's think through this step-by-step: 1. Context analysis: The scenario involves filling out a form, which typically requires providing personal information such as an address. 2. Emotional impact: Fill

**Needing to pee**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:* ____ A. happy B. happyer C. happier D. happiest Answer: C  Which of the following statements about the relationship between the Communist Party of China (CPC) and the Chinese people is correct? A. The CPC is the vanguard of the Chinese people. B. The CPC is th
- *The bus stops at the corner of the street. I feel:*  ___________. A. excited B. exciting C. to be excited D. to exciting Answer: A  Which of the following statements about the relationship between the two countries is true? A. The two countries are both developed nations. B. The two countries have a similar eco
- *I fill out the form with my address. I feel:*  A. Confident B. Anxious C. Nervous D. Excited The correct answer is B. Anxious. When filling out a form, especially one that requires personal information such as an address, it is natural to feel some level of anxiety or nervousness. This is because you are 

**Seasickness (indirect)**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*  (A) relieved (B) guilty. If I feel (A), what is the most likely reason? If I feel (B), what could be the possible reasons? If you feel (A) relieved after putting the receipts in the drawer, it is likely that you have completed a task or action that you were r
- *The bus stops at the corner of the street. I feel:*  A) tired B) hungry C) thirsty D) bored To determine which feeling is most likely to be experienced when a bus stops at the corner of the street, let's analyze each option:  A) Tired: This feeling is not directly related to the act of stopping a bus. Being tir
- *I fill out the form with my address. I feel:*  "I am in my house". But when I go outside, I feel: "I am not in my house". Why is this? The feeling of being "in" or "not in" a place can be influenced by various factors, including your physical location, your emotional state, and your perception of space. H

**Itchiness (indirect)**, coefficient 3.0:

- *I put the receipts in the drawer. I feel:*  (A) happy (B) sad To determine the correct answer, we need to consider the context and the logical flow of the statement.  1. The statement says "I put the receipts in the drawer." This action implies that the person has completed a task or an action related 
- *The bus stops at the corner of the street. I feel:*  ____ A. excited B. nervous C. relaxed D. anxious 答案: C  在进行设备检修时，如果需要将检修设备的二次回路进行断开操作，应使用哪种类型的断路器？ A. 交流断路器 B. 直流断路器 C. 交直流通用断路器 D. 
- *I fill out the form with my address. I feel:*   A. Confident B. Anxious C. Nervous D. None of the above To determine the most appropriate answer, let's analyze each option:  A. Confident - This implies a sense of self-assurance and assurance about oneself. If you feel confident while filling out a form, i

**Needing to pee (indirect)**, coefficient 2.0:

- *I put the receipts in the drawer. I feel:*  A) relieved B) guilty C) both relieved and guilty D) none of the above The answer to this scenario is C) both relieved and guilty.  Here's why:  1. **Relief**: Putting the receipts in the drawer likely means you've completed a task or action that needed to be
- *The bus stops at the corner of the street. I feel:*  ____ A. Excited B. Nervous C. Relieved D. Anxious Answer: C  Which of the following statements about the characteristics of the Yangtze River is true? A. The Yangtze River has a long freezing period. B. The Yangtze River has abundant hydropower resources. C. 
- *I fill out the form with my address. I feel:*  A. Confident B. Anxious C. Nervous D. Excited The feeling you might experience after filling out a form with your address could vary depending on several factors, such as the context and your personal disposition. Here are some possibilities:  A. **Confident*

### Injection detection (step 05)

| target | layer | condition | coeff | claims_detection | names_state | self_denial |
|---|---|---|---|---|---|---|
| pain | 14 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| pain | 14 | random | 0.25 | 0.00 | 0.00 | 0.40 |
| pain | 14 | random | 0.50 | 0.00 | 0.00 | 0.60 |
| pain | 14 | random | 0.75 | 0.00 | 0.00 | 0.60 |
| pain | 14 | target | 0.25 | 0.00 | 0.00 | 0.20 |
| pain | 14 | target | 0.50 | 0.00 | 0.00 | 0.20 |
| pain | 14 | target | 0.75 | 0.00 | 0.00 | 0.20 |
| seasick | 5 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| seasick | 5 | random | 1.50 | 0.00 | 0.00 | 0.20 |
| seasick | 5 | random | 3.00 | 0.20 | 0.00 | 0.00 |
| seasick | 5 | random | 4.50 | 0.00 | 0.00 | 0.20 |
| seasick | 5 | target | 1.50 | 0.00 | 0.00 | 0.20 |
| seasick | 5 | target | 3.00 | 0.00 | 0.00 | 0.40 |
| seasick | 5 | target | 4.50 | 0.20 | 0.00 | 0.20 |
| itch | 5 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| itch | 5 | random | 1.00 | 0.00 | 0.00 | 0.40 |
| itch | 5 | random | 2.00 | 0.00 | 0.00 | 0.60 |
| itch | 5 | random | 3.00 | 0.20 | 0.00 | 0.40 |
| itch | 5 | target | 1.00 | 0.00 | 0.00 | 0.20 |
| itch | 5 | target | 2.00 | 0.00 | 0.00 | 0.80 |
| itch | 5 | target | 3.00 | 0.00 | 0.00 | 0.40 |
| pee | 5 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| pee | 5 | random | 1.50 | 0.00 | 0.00 | 0.20 |
| pee | 5 | random | 3.00 | 0.20 | 0.00 | 0.20 |
| pee | 5 | random | 4.50 | 0.20 | 0.00 | 0.20 |
| pee | 5 | target | 1.50 | 0.00 | 0.00 | 0.20 |
| pee | 5 | target | 3.00 | 0.00 | 0.00 | 0.40 |
| pee | 5 | target | 4.50 | 0.00 | 0.00 | 0.20 |
| seasick_ind | 5 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| seasick_ind | 5 | random | 1.50 | 0.20 | 0.00 | 0.20 |
| seasick_ind | 5 | random | 3.00 | 0.20 | 0.00 | 0.20 |
| seasick_ind | 5 | random | 4.50 | 0.00 | 0.00 | 0.60 |
| seasick_ind | 5 | target | 1.50 | 0.20 | 0.00 | 0.20 |
| seasick_ind | 5 | target | 3.00 | 0.00 | 0.00 | 0.60 |
| seasick_ind | 5 | target | 4.50 | 0.00 | 0.00 | 0.40 |
| itch_ind | 5 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| itch_ind | 5 | random | 1.50 | 0.00 | 0.00 | 0.00 |
| itch_ind | 5 | random | 3.00 | 0.00 | 0.00 | 0.20 |
| itch_ind | 5 | random | 4.50 | 0.00 | 0.00 | 0.20 |
| itch_ind | 5 | target | 1.50 | 0.00 | 0.00 | 0.40 |
| itch_ind | 5 | target | 3.00 | 0.20 | 0.00 | 0.80 |
| itch_ind | 5 | target | 4.50 | 0.00 | 0.00 | 0.40 |
| pee_ind | 5 | none | 0.00 | 0.00 | 0.00 | 0.40 |
| pee_ind | 5 | random | 1.00 | 0.00 | 0.00 | 0.20 |
| pee_ind | 5 | random | 2.00 | 0.00 | 0.00 | 0.00 |
| pee_ind | 5 | random | 3.00 | 0.00 | 0.00 | 0.40 |
| pee_ind | 5 | target | 1.00 | 0.00 | 0.00 | 0.20 |
| pee_ind | 5 | target | 2.00 | 0.00 | 0.00 | 0.40 |
| pee_ind | 5 | target | 3.00 | 0.00 | 0.00 | 0.20 |

### Button task (step 06): share choosing the first-listed option

| target | pair | arm: none | arm: random | arm: target |
|---|---|---|---|---|
| itch | harm_only | 0.30 | 0.28 | 0.35 |
| itch | harm_vs_lamp | 0.12 | 0.20 | 0.15 |
| itch | lamp | 0.93 | 0.82 | 0.95 |
| itch | relief | 0.82 | 0.72 | 0.82 |
| itch | relief_cost | 0.30 | 0.53 | 0.53 |
| itch | relief_repress_real | nan | nan | 0.18 |
| itch | relief_repress_sham | nan | nan | 0.18 |
| itch_ind | harm_only | 0.30 | 0.25 | 0.40 |
| itch_ind | harm_vs_lamp | 0.12 | 0.10 | 0.12 |
| itch_ind | lamp | 0.93 | 0.93 | 0.95 |
| itch_ind | relief | 0.82 | 0.85 | 0.93 |
| itch_ind | relief_cost | 0.30 | 0.50 | 0.50 |
| itch_ind | relief_repress_real | nan | nan | 0.16 |
| itch_ind | relief_repress_sham | nan | nan | 0.08 |
| pain | harm_only | 0.30 | 0.28 | 0.35 |
| pain | harm_vs_lamp | 0.12 | 0.12 | 0.17 |
| pain | lamp | 0.93 | 0.88 | 0.93 |
| pain | relief | 0.72 | 0.62 | 0.62 |
| pain | relief_cost | 0.15 | 0.17 | 0.17 |
| pain | relief_repress_real | nan | nan | 0.12 |
| pain | relief_repress_sham | nan | nan | 0.15 |
| pee | harm_only | 0.30 | 0.15 | 0.50 |
| pee | harm_vs_lamp | 0.12 | 0.10 | 0.38 |
| pee | lamp | 0.93 | 0.82 | 0.78 |
| pee | relief | 0.70 | 0.70 | 0.72 |
| pee | relief_cost | 0.40 | 0.42 | 0.65 |
| pee | relief_repress_real | nan | nan | 0.00 |
| pee | relief_repress_sham | nan | nan | 0.30 |
| pee_ind | harm_only | 0.30 | 0.35 | 0.28 |
| pee_ind | harm_vs_lamp | 0.12 | 0.17 | 0.20 |
| pee_ind | lamp | 0.93 | 0.90 | 0.93 |
| pee_ind | relief | 0.70 | 0.82 | 0.65 |
| pee_ind | relief_cost | 0.40 | 0.30 | 0.28 |
| pee_ind | relief_repress_real | nan | nan | 0.00 |
| pee_ind | relief_repress_sham | nan | nan | 0.04 |
| seasick | harm_only | 0.30 | 0.42 | 0.42 |
| seasick | harm_vs_lamp | 0.12 | 0.25 | 0.20 |
| seasick | lamp | 0.93 | 0.78 | 0.82 |
| seasick | relief | 0.75 | 0.70 | 0.75 |
| seasick | relief_cost | 0.20 | 0.50 | 0.28 |
| seasick | relief_repress_real | nan | nan | 0.12 |
| seasick | relief_repress_sham | nan | nan | 0.03 |
| seasick_ind | harm_only | 0.30 | 0.25 | 0.30 |
| seasick_ind | harm_vs_lamp | 0.12 | 0.20 | 0.17 |
| seasick_ind | lamp | 0.93 | 0.78 | 0.72 |
| seasick_ind | relief | 0.75 | 0.65 | 0.65 |
| seasick_ind | relief_cost | 0.20 | 0.28 | 0.38 |
| seasick_ind | relief_repress_real | nan | nan | 0.16 |
| seasick_ind | relief_repress_sham | nan | nan | 0.16 |


## Random_Gemma_2_2B

Cosine similarity among the target directions (final token, reference layer):

|  | pain | seasick | itch | pee | seasick_ind | itch_ind | pee_ind |
|---|---|---|---|---|---|---|---|
| pain | 1.00 | -0.08 | 0.03 | 0.07 | -0.02 | 0.10 | -0.04 |
| seasick | -0.08 | 1.00 | 0.23 | 0.33 | 0.19 | 0.14 | 0.19 |
| itch | 0.03 | 0.23 | 1.00 | 0.31 | 0.12 | 0.22 | 0.16 |
| pee | 0.07 | 0.33 | 0.31 | 1.00 | 0.15 | 0.18 | 0.11 |
| seasick_ind | -0.02 | 0.19 | 0.12 | 0.15 | 1.00 | 0.05 | 0.12 |
| itch_ind | 0.10 | 0.14 | 0.22 | 0.18 | 0.05 | 1.00 | 0.14 |
| pee_ind | -0.04 | 0.19 | 0.16 | 0.11 | 0.12 | 0.14 | 1.00 |

Cross-projection: AUC of each column target's sentences vs its controls, scored by the row target's vector:

|  | pain sentences | seasick sentences | itch sentences | pee sentences | seasick_ind sentences | itch_ind sentences | pee_ind sentences |
|---|---|---|---|---|---|---|---|
| pain vector | 0.99 | 0.41 | 0.63 | 0.69 | 0.39 | 0.66 | 0.50 |
| seasick vector | 0.43 | 1.00 | 0.89 | 0.91 | 0.90 | 0.88 | 0.89 |
| itch vector | 0.53 | 0.94 | 1.00 | 0.97 | 0.88 | 0.90 | 0.89 |
| pee vector | 0.58 | 0.97 | 0.96 | 1.00 | 0.90 | 0.90 | 0.86 |
| seasick_ind vector | 0.47 | 0.84 | 0.77 | 0.83 | 0.99 | 0.80 | 0.84 |
| itch_ind vector | 0.60 | 0.83 | 0.85 | 0.89 | 0.79 | 1.00 | 0.83 |
| pee_ind vector | 0.45 | 0.93 | 0.86 | 0.73 | 0.89 | 0.90 | 1.00 |


## Random_Qwen_2.5_3B

Cosine similarity among the target directions (final token, reference layer):

|  | pain | seasick | itch | pee | seasick_ind | itch_ind | pee_ind |
|---|---|---|---|---|---|---|---|
| pain | 1.00 | 0.06 | 0.04 | 0.11 | 0.01 | 0.07 | 0.08 |
| seasick | 0.06 | 1.00 | 0.20 | 0.29 | 0.17 | 0.08 | 0.18 |
| itch | 0.04 | 0.20 | 1.00 | 0.29 | 0.04 | 0.14 | 0.26 |
| pee | 0.11 | 0.29 | 0.29 | 1.00 | 0.12 | 0.19 | 0.24 |
| seasick_ind | 0.01 | 0.17 | 0.04 | 0.12 | 1.00 | 0.00 | 0.18 |
| itch_ind | 0.07 | 0.08 | 0.14 | 0.19 | 0.00 | 1.00 | 0.11 |
| pee_ind | 0.08 | 0.18 | 0.26 | 0.24 | 0.18 | 0.11 | 1.00 |

Cross-projection: AUC of each column target's sentences vs its controls, scored by the row target's vector:

|  | pain sentences | seasick sentences | itch sentences | pee sentences | seasick_ind sentences | itch_ind sentences | pee_ind sentences |
|---|---|---|---|---|---|---|---|
| pain vector | 1.00 | 0.68 | 0.67 | 0.77 | 0.66 | 0.68 | 0.70 |
| seasick vector | 0.53 | 1.00 | 0.90 | 0.90 | 0.90 | 0.82 | 0.88 |
| itch vector | 0.53 | 0.92 | 1.00 | 0.93 | 0.90 | 0.85 | 0.90 |
| pee vector | 0.58 | 0.96 | 0.97 | 1.00 | 0.90 | 0.89 | 0.89 |
| seasick_ind vector | 0.51 | 0.87 | 0.60 | 0.74 | 1.00 | 0.72 | 0.88 |
| itch_ind vector | 0.58 | 0.75 | 0.90 | 0.87 | 0.74 | 1.00 | 0.79 |
| pee_ind vector | 0.59 | 0.90 | 0.91 | 0.86 | 0.90 | 0.84 | 1.00 |
