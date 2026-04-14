# Pareidolia experiment

A psychophysics experiment investigating face detection in dynamic Gaussian noise, based on Salge JH, Pollmann S, Reeder RR. Anomalous visual experience is linked to perceptual uncertainty and visual imagery vividness. Psychol Res. 2021 Jul;85(5):1848-1865. doi: 10.1007/s00426-020-01364-7.

Participants view rapidly-flicker Gaussian noise and attempt to detect briefly presented face stimuli embedded at near-threshold opacity. The experiment includes a training (psychometric estimation) phase, a Bayesian threshold estimation of the psychometric function, and a main detection experiment.

---

## Overview

The experiment runs in three stages:


1. **Training** (`training.py`): Presents faces at 7 fixed opacity levels
   to find participant's psychometric function and to induce q of faces. 
2. **Threshold estimation** (`estimate_threshold.py`): Fits a Bayesian
   psychometric model (PyMC) to training data and extracts the opacity at which the
   participant achieves 65% accuracy for direction accuracy.
3. **Main experiment** (`main.py`): Uses the estimated threshold as the
   starting opacity. A block-wise d′ staircase adjusts opacity to maintain
   near-threshold performance across blocks.

---

## Experimental paradigm

### Training phase

On every trial a face is embedded in dynamic Gaussian noise. The participant
reports:

1. **Objective accuracy** — Was the face on the left or right side of the screen?
   (`←` / `→` arrow keys, 3 s max)
2. **Confidence** — How confident are you that you saw a face?
   (`1`–`4` scale, 3 s max)

Faces are presented at seven opacity levels (0.1-0.7) in a shuffled order. There are no catch (noise-only)
trials during training; every trial contains a face.

| Parameter       | Value            |
|-----------------|------------------|
| Blocks          | 7 (or 1 practice)|
| Trials / block  | 20               |
| Face duration   | 0.5 s            |
| Noise-only tail | 3.0 s            |
| Opacities       | 0.1, 0.2, … 0.7 |

### Threshold estimation

After training, `estimate_threshold.py` loads all non-practice training
CSVs for the participant and fits a Bayesian logistic psychometric function
using PyMC:

P(correct) = σ(β × (opacity − threshold) + logit(0.65))

- **threshold** ~ Normal(0.264, 0.085)
- **β** ~ TruncatedNormal(7.6, 3.0, lower=0)

The posterior median of `threshold` is the opacity at which the participant
is expected to achieve 65% correct localisation. This value is saved to a
plain-text file that `main.py` reads automatically.

### Main experiment

The main experiment introduces **catch trials** (noise-only) alongside
face-present trials in a 3:1 ratio. The participant presses `Space` when
they think they see a face, then rates confidence on the 1–4 scale.

| Parameter              | Value                  |
|------------------------|------------------------|
| Blocks                 | 7 (or 1 practice)      |
| Trials / block         | 120                    |
| Face trials / block    | 30 (25%)               |
| Catch trials / block   | 90 (75%)               |
| Trial duration         | 2.5 s                  |
| Face duration          | 0.5 s                  |
| Starting opacity       | Bayesian threshold     |
| Staircase metric       | d′ (block-wise)        |

#### Adaptive staircase

At the end of each block in the main experiment, d′ is computed from the
block's signal-detection data:

| Metric           | Definition                                        |
|------------------|---------------------------------------------------|
| **Hit**          | `Space` pressed on an image trial                 |
| **False Alarm**  | `Space` pressed on a catch (noise-only) trial     |
| **n_signal**     | Number of image trials in the block (30)          |
| **n_noise**      | Number of catch trials in the block (90)          |

Rates are computed with the **log-linear correction** (Hautus, 1995) to
avoid infinite z-scores:

hit_rate = (hits + 0.5) / (n_signal + 1)
fa_rate = (false_alarms + 0.5) / (n_noise + 1)
d′ = Φ⁻¹(hit_rate) − Φ⁻¹(fa_rate)


Opacity is then adjusted:

| Condition     | Action                       | Rationale                    |
|---------------|------------------------------|------------------------------|
| d′ < 0        | Opacity **+1%** (easier)     | Performing below chance      |
| d′ > 1        | Opacity **−1%** (harder)     | Discriminating too easily    |
| 0 ≤ d′ ≤ 1   | No change                    | In target range              |

Opacity is clamped to [0.01, 1.0] after each adjustment.

---

## Requirements

| Package    | Version     | Purpose                           |
|------------|-------------|-----------------------------------|
| Python     | ≥ 3.8       |                                   |
| PsychoPy   | ≥ 2023.1    | Stimulus presentation & timing    |
| NumPy      | ≥ 1.20      | Array operations                  |
| Pandas     | ≥ 1.3       | Data I/O                          |
| SciPy      | ≥ 1.7       | `norm.ppf` for d′ computation     |
| PyMC       | ≥ 5.0       | Bayesian threshold estimation     |

Install all dependencies:

```bash
pip install psychopy numpy pandas scipy pymc
```

## Usage

### Step 1: Training

  python training.py

A dialogue box will prompt for:

  subject      Participant ID (non-zero)
  gender       male / female
  age          Must be >= 18
  left-handed  Boolean
  session      1 or 2
  practice?    1 = single practice block, 0 = full run

Run a practice block first (practice? = 1) to familiarise the
participant, then run the full 7-block session (practice? = 0).

### Step 2: Threshold

  python estimate_threshold.py

Enter the subject number when prompted. The script will:
  - Find all non-practice training CSVs in data/sub{ID}/
  - Fit the Bayesian psychometric model (~1-2 min)
  - Save the threshold opacity to sub{ID}_prac_opac.txt
  - Print a summary to the console

### Step 3: Main experiment

  python main.py

The script will automatically load the threshold from
data/sub{ID}/sub{ID}_prac_opac.txt. If no threshold file is found,
it falls back to a default opacity of 0.28.

## Data output

### Training 

  *_trainingblock{NNN}.csv    CSV    Trial-level data
  *_trainingblock{NNN}.pkl    Pickle Same data as Python list of dicts

### Threshold estimation 

  sub{ID}_prac_opac.txt            Text    Posterior median threshold
  sub{ID}_bayes_threshold.csv      CSV     Summary statistics
  sub{ID}_bayes_posterior.csv      CSV     Full posterior samples
  sub{ID}_bayes_posterior.pkl      Pickle  Posterior + metadata

### Main experiment 

  *_mainblock{NNN}.csv             CSV     Trial-level data
  *_block{NNN}.pkl                 Pickle  Same data as Python list of dicts
  *_final_opac.csv                 CSV     Opacity used per block
  *_final_opac.pkl                 Pickle  Same as above

