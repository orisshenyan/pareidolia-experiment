# ==============================================================================
# Imports
# ==============================================================================
import os
import pickle
import random
import itertools
from datetime import datetime
from scipy.stats import norm 
import numpy as np
import numpy.random as rnd
import pandas as pd
from psychopy import core, event, visual, gui, data, monitors, logging

# ==============================================================================
# Patch deprecated Clock.add method
# ==============================================================================
from psychopy.core import Clock
Clock.add = lambda self, time: self.addTime(time)

# Suppress DEPRECATED warnings
logging.setDefaultClock(core.Clock())
logging.console.setLevel(logging.CRITICAL)

# ==============================================================================
# Constants
# ==============================================================================
NUM_NOISE_IMAGES = 431
FACE_IMAGE_COUNT = 10
FRAME_RATE = 0.066
TRIAL_LENGTH = 2.5
FACE_DISPLAY_DURATION = 0.5
CONFIDENCE_MAX_WAIT = 2.5
DEFAULT_OPACITY = 0.28
OPACITY_STEP = 0.01
DETECTION_HIGH_THRESHOLD = 0.90
DETECTION_LOW_THRESHOLD = 0.10
HALLUCINATION_WARNING_THRESHOLD = 30

CODE_HALL_TRIAL = 80
CODE_IMG_TRIAL = 90

POSITIONS = [
    (-5, -5), (5, 5), (5, -5), (-5, 5),
    (7, 7), (-7, -7), (7, -7), (-7, 7),
]
FACE_SIZES = [6, 7, 8, 9]

IMAGE_PATH = 'pictures/'
NOISE_PATH = 'new_wn_pics/'

# ==============================================================================
# Dialogue Box
# ==============================================================================
exp_info = {
    'subject': 0,
    'gender': ('male', 'female'),
    'age': 0,
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

# ==============================================================================
# File / Directory Setup
# ==============================================================================
subject_dir = f"data/sub{exp_info['subject']}/"
os.makedirs(subject_dir, exist_ok=True)

practice_tag = "_practice" if exp_info['practice?'] == '1' else ""

data_fname = (
    f"{subject_dir}sub{exp_info['subject']}"
    f"_{exp_info['gender']}_{exp_info['age']}"
    f"_session{exp_info['session']}{practice_tag}_{exp_info['datetime']}"
)

# ==============================================================================
# Random Seed
# ==============================================================================
seed = random.randint(1, 1_000_000)
random.seed(seed)
np.random.seed(seed)
print(f"[SEED] {seed}")

seed_filename = (
    f"{subject_dir}mainexpseed_{exp_info['subject']}"
    f"_{exp_info['gender']}_{exp_info['age']}_{exp_info['datetime']}.txt"
)
with open(seed_filename, "a") as log_file:
    log_file.write(
        f"Random seed for subject {exp_info['subject']} "
        f"({exp_info['gender']}, {exp_info['age']}): {seed}\n"
    )

# ==============================================================================
# Experiment Settings
# ==============================================================================
n_trials = 120
n_faces = n_trials // 4

if exp_info['practice?'] == '0':
    n_blocks = 7
elif exp_info['practice?'] == '1':
    n_blocks = 1
else:
    raise ValueError("Invalid value for 'practice?'. Must be '0' or '1'.")

# ==============================================================================
# Load Bayesian threshold from training
# ==============================================================================
opac_txt_path = os.path.join(
    subject_dir, f"sub{exp_info['subject']}_prac_opac.txt"
)
opac_txt_session = os.path.join(
    subject_dir,
    f"sub{exp_info['subject']}_session{exp_info['session']}_prac_opac.txt"
)

start_opac = None

for path in [opac_txt_session, opac_txt_path]:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                start_opac = float(f.read().strip())
            print(f"[INFO] Loaded Bayesian threshold: {start_opac:.4f}")
            print(f"       from: {path}")
            break
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

if start_opac is None or start_opac <= 0 or start_opac > 1:
    print(f"[WARN] No valid threshold found. Using default: {DEFAULT_OPACITY}")
    start_opac = DEFAULT_OPACITY

print(f"[INFO] Starting opacity: {start_opac:.4f}")

# ==============================================================================
# Helper Functions
# ==============================================================================
def pattern_match(pattern, sequence):
    """Count the number of times `pattern` occurs in `sequence`."""
    pattern = tuple(pattern)
    k = len(pattern)
    iterators = itertools.tee(sequence, k)
    for j in range(k):
        for _ in range(j):
            next(iterators[j])
    return sum(1 for q in zip(*iterators) if q == pattern)


def show_noise_frame(wn_stim, win):
    """Draw a random noise image and flip."""
    img_idx = np.random.randint(NUM_NOISE_IMAGES)
    wn_stim.setImage(f'{NOISE_PATH}im{img_idx}.png')
    wn_stim.draw()
    win.flip()
    return img_idx


def show_noise_with_face(wn_stim, bitmap, win):
    """Draw a random noise image with a face overlay and flip."""
    img_idx = np.random.randint(NUM_NOISE_IMAGES)
    wn_stim.setImage(f'{NOISE_PATH}im{img_idx}.png')
    wn_stim.draw()
    bitmap.draw()
    win.flip()
    return img_idx


def wait_for_frames(wait_time):
    """Busy-wait until the next frame is due."""
    while wait_time.getTime() < 0:
        pass
    wait_time.add(FRAME_RATE)


def collect_confidence(win, confi_msg, rt_clock, confions, block, trial):
    """Display confidence screen and collect response."""
    confi_msg.draw()
    win.flip()
    confions[block][trial] = rt_clock.getTime()

    confi_keys = event.waitKeys(
        maxWait=CONFIDENCE_MAX_WAIT,
        keyList=['1', '2', '3', '4', 'escape']
    )

    if not confi_keys:
        return 'nan'
    if 'escape' in confi_keys:
        win.close()
        core.quit()

    key_map = {'1': 1, '2': 2, '3': 3, '4': 4}
    for key, val in key_map.items():
        if key in confi_keys:
            return val
    return 'nan'


def fill_remaining_time(remaining, wn_stim, win):
    """Fill the rest of a trial with noise frames."""
    if remaining <= 0:
        return
    timer = core.CountdownTimer()
    wait_time = core.Clock()
    timer.reset()
    timer.add(remaining)
    wait_time.reset()
    wait_time.add(FRAME_RATE)
    while timer.getTime() > 0:
        show_noise_frame(wn_stim, win)
        wait_for_frames(wait_time)


def compute_dprime(hits, n_signal, false_alarms, n_noise):
    """
    Compute d' with log-linear correction to avoid infinite values.

    The log-linear rule adds 0.5 to both hit and FA counts and adds 1
    to both trial counts, which avoids rates of exactly 0 or 1 and the
    resulting ±inf z-scores.  This is the recommended correction from:
    Hautus, M. J. (1995). Corrections for extreme proportions and their
    biasing effects on estimated values of d'. Behavior Research Methods,
    Instruments, & Computers, 27, 46–51.
    """
    hit_rate = (hits + 0.5) / (n_signal + 1)
    fa_rate  = (false_alarms + 0.5) / (n_noise + 1)

    dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    return dprime, hit_rate, fa_rate


def update_opacity(current_opacity, hits, n_signal, false_alarms, n_noise):
    """
    Adjust opacity based on d':
      d' < 0  →  increase opacity (make face easier to see)
      d' > 1  →  decrease opacity (make face harder to see)
      0 ≤ d' ≤ 1  →  no change
    """
    dprime, hit_rate, fa_rate = compute_dprime(
        hits, n_signal, false_alarms, n_noise
    )

    print(f"  Hit rate:  {hit_rate:.3f}  ({hits}/{n_signal})")
    print(f"  FA rate:   {fa_rate:.3f}  ({false_alarms}/{n_noise})")
    print(f"  d':        {dprime:.3f}")

    if dprime < 0:
        new_opacity = current_opacity + OPACITY_STEP   # easier
        print(f"  d' < 0 → opacity UP   {current_opacity:.4f} → {new_opacity:.4f}")
    elif dprime > 1:
        new_opacity = current_opacity - OPACITY_STEP   # harder
        print(f"  d' > 1 → opacity DOWN {current_opacity:.4f} → {new_opacity:.4f}")
    else:
        new_opacity = current_opacity                   # no change
        print(f"  0 ≤ d' ≤ 1 → opacity UNCHANGED at {new_opacity:.4f}")

    # Clamp to valid range
    new_opacity = np.clip(new_opacity, 0.01, 1.0)
    return new_opacity


# ==============================================================================
# Monitor & Window Setup
# ==============================================================================
mon = monitors.Monitor('Iiyama', width=60.96, distance=60)
mon.setSizePix((1920, 1080))

win = visual.Window(
    fullscr=True, color='grey', monitor=mon, units='deg', screen=0
)
event.Mouse(visible=False)

# ==============================================================================
# Stimuli
# ==============================================================================
bitmap = visual.ImageStim(win, contrast=0.19)
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
wn_stim = visual.ImageStim(win, units='pix', size=(1920, 1080))
start_message = visual.TextStim(
    win, text='Press any key to begin', color='black', height=1, pos=(0, 3),
)
interblock_msg = visual.TextStim(win, text='', color='black', height=1)

# ==============================================================================
# Data Arrays
# ==============================================================================
updated_opac = np.zeros(n_blocks)
sub_opac = np.zeros(n_blocks)

detection_resp = np.zeros((n_blocks, n_trials))
detection_confi = np.zeros((n_blocks, n_trials))
hall_resp = np.zeros((n_blocks, n_trials))
hall_confi = np.zeros((n_blocks, n_trials))
wn_number = np.zeros((n_blocks, n_trials))

fixons = np.zeros((n_blocks, n_trials))
confions = np.zeros((n_blocks, n_trials))
bitons = np.zeros((n_blocks, n_trials))
hall_resp_timestamp = np.zeros((n_blocks, n_trials))
face_resp_timestamp = np.zeros((n_blocks, n_trials))
face_pos = np.zeros((n_blocks, n_trials))
face_size = np.zeros((n_blocks, n_trials))

# ==============================================================================
# Main Experiment Loop
# ==============================================================================
face_count = 0

for block in range(n_blocks):

    # --- Set opacity for this block ---
    if block == 0:
        sub_opac[block] = start_opac
    else:
        sub_opac[block] = updated_opac[block - 1]

    # --- Randomize trial order (3 fixation : 1 image) ---
    trial_type = ['f', 'f', 'f', 'i'] * (n_trials // 4)
    np.random.shuffle(trial_type)

    # --- Show start / inter-block message ---
    if block == 0:
        start_message.draw()
    else:
        interblock_msg.text = (
            f"Block completed!\n\n"
            f"You have completed block {block} out of {n_blocks}.\n"
            f"You found {int((face_count / n_faces) * 100)}% of faces.\n\n"
            f"Press any key to begin the next block."
        )
        interblock_msg.draw()
    win.flip()

    keys = event.waitKeys(timeStamped=True)
    if 'escape' in [k[0] for k in keys]:
        win.close()
        core.quit()

    # --- Block clock ---
    rt_clock = core.Clock()
    rt_clock.reset()
    globalClock = core.Clock()
    logging.setDefaultClock(globalClock)
    begin = rt_clock.getTime()

    # ------------------------------------------------------------------
    # Trial Loop
    # ------------------------------------------------------------------
    for trial in range(n_trials):

        trial_onset = rt_clock.getTime()
        timer = core.CountdownTimer()
        wait_time = core.Clock()

        resp = 0
        confi = 'nan'
        hall = 0
        hallconfi = 'nan'

        # ==============================================================
        # FIXATION TRIAL (noise only)
        # ==============================================================
        if trial_type[trial] == 'f':

            fixons[block][trial] = rt_clock.getTime()
            timer.reset()
            timer.add(TRIAL_LENGTH)
            wait_time.reset()
            wait_time.add(FRAME_RATE)

            keys = []
            while timer.getTime() > 0:
                keys = event.getKeys(keyList=['space', 'escape'])
                img_idx = show_noise_frame(wn_stim, win)
                wn_number[block][trial] = img_idx
                wait_for_frames(wait_time)
                if keys:
                    break

            if not keys:
                hall = 0
                hallconfi = 'nan'
            else:
                hall_resp_timestamp[block][trial] = rt_clock.getTime()
                break_time = hall_resp_timestamp[block][trial] - fixons[block][trial]

                if 'escape' in keys:
                    win.close()
                    core.quit()

                if 'space' in keys:
                    hall = 1
                    hallconfi = collect_confidence(
                        win, confi_msg, rt_clock, confions, block, trial
                    )
                    print(
                        f"Hallucination detected: Confidence = {hallconfi}, "
                        f"Timestamp = {hall_resp_timestamp[block][trial]}"
                    )

                confi_time = rt_clock.getTime() - confions[block][trial]
                makeup_time = TRIAL_LENGTH - (confi_time + break_time)
                fill_remaining_time(makeup_time, wn_stim, win)

            resp = CODE_HALL_TRIAL
            confi = CODE_HALL_TRIAL

            print(f"Fixation trial duration: {rt_clock.getTime() - trial_onset:.3f}s")

        # ==============================================================
        # IMAGE TRIAL (face + noise)
        # ==============================================================
        elif trial_type[trial] == 'i':

            face_id = int(rnd.choice(FACE_IMAGE_COUNT) + 1)
            bitmap.setImage(f'{IMAGE_PATH}f{face_id}.png')
            bitmap.setPos(POSITIONS[rnd.randint(len(POSITIONS))])
            bitmap.setSize(FACE_SIZES[rnd.randint(len(FACE_SIZES))])
            bitmap.setOpacity(sub_opac[block])
            bitons[block][trial] = rt_clock.getTime()
            face_pos[block][trial] = bitmap.pos[0]
            face_size[block][trial] = bitmap.size[0]

            # --- Phase 1: face + noise ---
            timer.reset()
            timer.add(FACE_DISPLAY_DURATION)
            wait_time.reset()
            wait_time.add(FRAME_RATE)

            keys = []
            while timer.getTime() > 0:
                keys = event.getKeys(keyList=['space', 'escape'])
                img_idx = show_noise_with_face(wn_stim, bitmap, win)
                wn_number[block][trial] = img_idx
                wait_for_frames(wait_time)
                if keys:
                    break

            # --- Phase 2: noise only ---
            if not keys:
                timer.reset()
                timer.add(TRIAL_LENGTH - FACE_DISPLAY_DURATION)
                wait_time.reset()
                wait_time.add(FRAME_RATE)

                while timer.getTime() > 0:
                    keys = event.getKeys(keyList=['space', 'escape'])
                    img_idx = show_noise_frame(wn_stim, win)
                    wn_number[block][trial] = img_idx
                    wait_for_frames(wait_time)
                    if keys:
                        break

            # --- Process response ---
            if not keys:
                resp = 0
                confi = 'nan'
            else:
                face_resp_timestamp[block][trial] = rt_clock.getTime()
                break_time = face_resp_timestamp[block][trial] - bitons[block][trial]

                if 'escape' in keys:
                    win.close()
                    core.quit()

                if 'space' in keys:
                    resp = 1
                    confi = collect_confidence(
                        win, confi_msg, rt_clock, confions, block, trial
                    )
                    print(
                        f"Face perceived: Confidence = {confi}, "
                        f"Timestamp = {face_resp_timestamp[block][trial]}"
                    )

                confi_time = rt_clock.getTime() - confions[block][trial]
                makeup_time = TRIAL_LENGTH - (confi_time + break_time)
                fill_remaining_time(max(0, makeup_time), wn_stim, win)

            hall = CODE_IMG_TRIAL
            hallconfi = CODE_IMG_TRIAL

            print(f"Image trial duration: {rt_clock.getTime() - trial_onset:.3f}s")
            print(
                f"Face position: {bitmap.pos}, "
                f"size: {bitmap.size}, opacity: {bitmap.opacity}"
            )

        # --- Store trial data ---
        detection_resp[block][trial] = resp
        detection_confi[block][trial] = confi if confi != 'nan' else np.nan
        hall_resp[block][trial] = hall
        hall_confi[block][trial] = hallconfi if hallconfi != 'nan' else np.nan

    # ------------------------------------------------------------------
    # End-of-Block Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"BLOCK {block + 1}/{n_blocks} SUMMARY")
    print(f"{'='*50}")
    print(f"  Duration:  {end - begin:.2f}s")
    print(f"  Opacity:   {sub_opac[block]:.4f}")

    face_count = int(np.sum(detection_resp[block] == 1))
    hall_count = int(np.sum(hall_resp[block] == 1))

    n_signal_trials = trial_type.count('i')
    n_noise_trials  = trial_type.count('f')

    updated_opac[block] = update_opacity(
        sub_opac[block],
        hits=face_count,
        n_signal=n_signal_trials,
        false_alarms=hall_count,
        n_noise=n_noise_trials,
    )

    if hall_count > HALLUCINATION_WARNING_THRESHOLD:
        print(f"  ⚠ WARNING: {hall_count} hallucinations — pressing too much?")
    print(f"{'='*50}\n")

    # --- Save block data ---
    block_data = []
    for (a, b, c, d, e, f, g, h, i, j, k, l) in zip(
        detection_resp[block], detection_confi[block],
        bitons[block], face_resp_timestamp[block],
        hall_resp[block], hall_confi[block],
        fixons[block], hall_resp_timestamp[block],
        confions[block], face_pos[block],
        wn_number[block], face_size[block],
    ):
        block_data.append({
            'Realface_response': a,
            'Realface_confidence': b,
            'Realface_Image_Onset_Time': c,
            'RealFace_Image_Response_Time': d,
            'Hallucination_response': e,
            'Hallucination_confidence': f,
            'Noise_Onset_Time': g,
            'Hallucination_Response_Time': h,
            'ConfidenceScreen_Onset': i,
            'Face_Position': j,
            'Noise_Number': k,
            'Face_Size': l,
        })

    with open(f'{data_fname}_block{block:03d}.pkl', 'wb') as output:
        pickle.dump(block_data, output)

    pd.DataFrame(block_data).to_csv(
        f'{data_fname}_mainblock{block:03d}.csv', index=False
    )

# ==============================================================================
# Save Final Opacity & Cleanup
# ==============================================================================
with open(f'{data_fname}_final_opac.pkl', 'wb') as opac_output:
    pickle.dump(sub_opac, opac_output)

pd.DataFrame(sub_opac).to_csv(f'{data_fname}_final_opac.csv', index=False)

win.close()
core.quit()