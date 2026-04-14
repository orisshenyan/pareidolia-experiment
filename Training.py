# ============================================
# Imports
# ============================================
import pickle
import itertools
import os
import random
from datetime import datetime

import numpy as np
import numpy.random as rnd
import pandas as pd
from psychopy import core, event, visual, gui, monitors, logging

# Suppress DEPRECATED warnings
logging.setDefaultClock(core.Clock())
logging.console.setLevel(logging.ERROR)


# ============================================
# Dialogue Box — Subject Info
# ============================================
exp_info = {
    'subject': 'test',
    'gender': ('male', 'female'),
    'age': 18,
    'left-handed': False,
    'session': ('1', '2'),
    'practice?': ('0', '1'),
}
dlg = gui.DlgFromDict(dictionary=exp_info)

if not dlg.OK:
    core.quit()
if exp_info['subject'] == 0:
    raise Exception('Zero is not a valid subject number!')
if exp_info['age'] < 18:
    raise Exception(f"{exp_info['age']} year olds cannot give consent!")

exp_info['datetime'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ============================================
# File & Directory Setup
# ============================================
subject_dir = f"data/sub{exp_info['subject']}/"
os.makedirs(subject_dir, exist_ok=True)

practice_tag = "_practice" if exp_info['practice?'] == '1' else ""

data_fname = (
    f"{subject_dir}sub{exp_info['subject']}"
    f"_{exp_info['gender']}_{exp_info['age']}"
    f"_session{exp_info['session']}{practice_tag}_{exp_info['datetime']}"
)


# ============================================
# Random Seed
# ============================================
seed = random.randint(1, 1_000_000)
random.seed(seed)
np.random.seed(seed)

seed_filename = (
    f"{subject_dir}testingseed_{exp_info['subject']}"
    f"_{exp_info['gender']}_{exp_info['age']}_{exp_info['datetime']}.txt"
)
with open(seed_filename, "a") as log_file:
    log_file.write(
        f"Random seed for subject {exp_info['subject']} "
        f"({exp_info['gender']}, {exp_info['age']}): {seed}\n"
    )
print(f"[SEED] {seed}")


# ============================================
# Experiment Constants
# ============================================
IMPATH = 'pictures/'
N_TRIALS = 20
N_BLOCKS = 7 if exp_info['practice?'] == '0' else 1
FRAME_DUR = 0.066          # ~60 Hz minus 3 ms headroom
FACE_DUR = 0.5             # seconds face is embedded in noise
NOISE_ONLY_DUR = 3.0       # seconds of noise after face offset
DIRECTION_WAIT = 3.0       # max wait for direction response
CONFIDENCE_WAIT = 3.0      # max wait for confidence response
FRAMES_PER_NOISE = 4
MONITOR_HZ = 60

POSITIONS = [
    (-5, -5), (5, 5), (5, -5), (-5, 5),
    (7, 7), (-7, -7), (7, -7), (-7, 7),
]
OPACITIES = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
FACE_SIZES = [6, 7, 8, 9]
N_FACE_IMAGES = 10         # faces named f1.png … f10.png
N_NOISE_IMAGES = 431       # noise images named im0.png … im430.png

DIRECTION_LABELS = {0: 'None', 1: 'Left', 2: 'Right'}


# ============================================
# Monitor & Window
# ============================================
mon = monitors.Monitor('Iiyama', width=60.96, distance=60)
mon.setSizePix((1920, 1080))

win = visual.Window(fullscr=True, units='deg', monitor=mon, screen=0)
event.Mouse(visible=False)


# ============================================
# Stimuli
# ============================================
bitmap = visual.ImageStim(win, contrast=0.19)

wn_stim = visual.ImageStim(win, units='pix', size=(1920, 1080))

confi_msg = visual.TextStim(
    win,
    text=(
        '1 = I did not see the face\n'
        '2 = I probably did not see the face\n'
        '3 = I probably saw the face\n'
        '4 = I saw the face'
    ),
    color='black', height=1,
)

left_right_msg = visual.TextStim(
    win, text='Was the face on the left or right?',
    color='black', height=1, pos=(0, 0),
)

start_message = visual.TextStim(
    win, text='Press any key to begin',
    color='black', height=1, pos=(0, 3),
)


# ============================================
# Data Storage
# ============================================
face_resp     = np.zeros((N_BLOCKS, N_TRIALS))
confi_rate    = np.zeros((N_BLOCKS, N_TRIALS))
direction_arr = np.zeros((N_BLOCKS, N_TRIALS))   # 0=none, 1=left, 2=right
face_pos      = np.zeros((N_BLOCKS, N_TRIALS))
face_size_arr = np.zeros((N_BLOCKS, N_TRIALS))
opac_arr      = np.zeros((N_BLOCKS, N_TRIALS))
wn_number     = np.zeros((N_BLOCKS, N_TRIALS))
onset_time    = np.zeros((N_BLOCKS, N_TRIALS))


# ============================================
# Helper Functions
# ============================================
def check_for_escape():
    """Consume only 'escape' from the key buffer; quit if found."""
    if 'escape' in event.getKeys(keyList=['escape']):
        print("\n[QUIT] Escape pressed — aborting experiment.")
        win.close()
        core.quit()


def quit_if_escape(keys):
    """Quit if an 'escape' response was returned by waitKeys."""
    if keys and 'escape' in keys:
        print("\n[QUIT] Escape pressed — aborting experiment.")
        win.close()
        core.quit()


def parse_direction(keys):
    """Return 1 (left), 2 (right), or 0 (no / invalid response)."""
    if not keys:
        return 0
    if 'left' in keys:
        return 1
    if 'right' in keys:
        return 2
    return 0


def parse_confidence(keys):
    """Return 1-4 or 0 (no / invalid response)."""
    if keys and keys[0] in ('1', '2', '3', '4'):
        return int(keys[0])
    return 0


# ============================================
# Main Experiment Loop
# ============================================
print(f"\n{'=' * 55}")
print(f"  EXPERIMENT START  |  {N_BLOCKS} block(s) × {N_TRIALS} trials")
print(f"  Subject: {exp_info['subject']}  |  Session: {exp_info['session']}")
print(f"{'=' * 55}")

for block in range(N_BLOCKS):
    # Shuffle opacities for this block
    shuffled_opacities = OPACITIES.copy()
    np.random.shuffle(shuffled_opacities)

    print(f"\n{'─' * 55}")
    print(f"  BLOCK {block + 1}/{N_BLOCKS}")
    print(f"  Opacity cycle: {[f'{o:.1f}' for o in shuffled_opacities]}")
    print(f"{'─' * 55}")

    # Wait for keypress to start block
    start_message.draw()
    win.flip()
    event.waitKeys()
    check_for_escape()

    # Block clock
    rt_clock = core.Clock()
    globalClock = core.Clock()
    logging.setDefaultClock(globalClock)

    for trial in range(N_TRIALS):
        # ---- Trial parameters ----
        trial_opacity  = shuffled_opacities[trial % len(shuffled_opacities)]
        face_img_idx   = int(rnd.choice(N_FACE_IMAGES) + 1)
        trial_position = POSITIONS[rnd.randint(len(POSITIONS))]
        trial_size     = FACE_SIZES[rnd.randint(len(FACE_SIZES))]

        bitmap.setOpacity(trial_opacity)
        bitmap.setImage(f"{IMPATH}f{face_img_idx}.png")
        bitmap.setPos(trial_position)
        bitmap.setSize(trial_size)

        # Store
        opac_arr[block][trial]      = trial_opacity
        face_pos[block][trial]      = trial_position[0]
        face_size_arr[block][trial] = trial_size
        onset_time[block][trial]    = rt_clock.getTime()

        print(
            f"  Trial {trial + 1:2d}/{N_TRIALS}  |  "
            f"opacity={trial_opacity:.2f}  pos={trial_position}  "
            f"size={trial_size}  face=f{face_img_idx}.png"
        )

        # ---- Face + noise phase (FACE_DUR s) ----
        n_face_frames = int(FACE_DUR * MONITOR_HZ)

        for frame in range(n_face_frames):
            check_for_escape()
            if frame % FRAMES_PER_NOISE == 0:
                noise_idx = np.random.randint(N_NOISE_IMAGES)
                wn_number[block][trial] = noise_idx
                wn_stim.setImage(f'new_wn_pics/im{noise_idx}.png')
            wn_stim.draw()
            bitmap.draw()
            win.flip()

        # ---- Noise-only phase (NOISE_ONLY_DUR s) ----
        n_noise_frames = int(NOISE_ONLY_DUR * MONITOR_HZ)

        for frame in range(n_noise_frames):
            check_for_escape()
            if frame % FRAMES_PER_NOISE == 0:
                noise_idx = np.random.randint(N_NOISE_IMAGES)
                wn_number[block][trial] = noise_idx
                wn_stim.setImage(f'new_wn_pics/im{noise_idx}.png')
            wn_stim.draw()
            win.flip()

        # ---- Direction question ----
        left_right_msg.draw()
        win.flip()
        dir_keys = event.waitKeys(
            maxWait=DIRECTION_WAIT, keyList=['left', 'right', 'escape']
        )
        quit_if_escape(dir_keys)
        trial_direction = parse_direction(dir_keys)

        # ---- Confidence question ----
        confi_msg.draw()
        win.flip()
        confi_keys = event.waitKeys(
            maxWait=CONFIDENCE_WAIT, keyList=['1', '2', '3', '4', 'escape']
        )
        quit_if_escape(confi_keys)
        trial_confidence = parse_confidence(confi_keys)

        # ---- Store responses ----
        face_resp[block][trial]     = 1
        confi_rate[block][trial]    = trial_confidence
        direction_arr[block][trial] = trial_direction

        print(
            f"             → direction={DIRECTION_LABELS[trial_direction]}  "
            f"confidence={trial_confidence}"
        )

    # ---- Block summary ----
    block_opacities = opac_arr[block]
    block_conf      = confi_rate[block]
    print(f"\n  Block {block + 1} summary:")
    print(f"    Mean confidence : {block_conf.mean():.2f}")
    print(f"    Direction counts: "
          f"Left={int((direction_arr[block] == 1).sum())}  "
          f"Right={int((direction_arr[block] == 2).sum())}  "
          f"None={int((direction_arr[block] == 0).sum())}")

    # ---- Save block data ----
    block_data = []
    for t in range(N_TRIALS):
        block_data.append({
            'Trial':            t + 1,
            'Face_Response':    face_resp[block][t],
            'Face_Confidence':  confi_rate[block][t],
            'Face_Onset_Time':  onset_time[block][t],
            'Face_Position_X':  face_pos[block][t],
            'Noise_Number':     wn_number[block][t],
            'Face_Size':        face_size_arr[block][t],
            'Opacity':          opac_arr[block][t],
            'Direction_Report': direction_arr[block][t],
        })

    pkl_path = f"{data_fname}_trainingblock{block:03d}.pkl"
    csv_path = f"{data_fname}_trainingblock{block:03d}.csv"

    with open(pkl_path, 'wb') as f:
        pickle.dump(block_data, f)

    pd.DataFrame(block_data).to_csv(csv_path, index=False)
    print(f"    Saved → {csv_path}")


# ============================================
# Cleanup
# ============================================
win.close()
core.quit()


# ============================================
# Cleanup
# ============================================
print(f"\n{'=' * 55}")
print("  EXPERIMENT COMPLETE")
print(f"{'=' * 55}\n")

win.close()
core.quit()