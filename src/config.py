"""
config.py - Hybrid static + LSTM gesture recognition system
Static model  →  A-Z, SPACE, SPEAK, DELETE  (29 classes)
Motion model  →  J, Z kept for future motion recognition  ( 2 classes, preserved)
"""

import os

# --- Paths -------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

# --- All gesture classes -----------------------------------------------------
GESTURE_CLASSES = [
    "A","B","C","D","DELETE","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z","SPACE","SPEAK"
]

# --- Motion vs Static split --------------------------------------------------
# J and Z are kept as motion letters for future motion model development.
# For now they are ALSO included in static training so they work immediately.
MOTION_LETTERS  = ["J", "Z"]
MOTION_CLASSES  = MOTION_LETTERS                                      # alias
STATIC_LETTERS  = GESTURE_CLASSES                                     # all 29 — J, Z, DELETE included
STATIC_CLASSES  = STATIC_LETTERS                                      # alias

NUM_CLASSES         = len(GESTURE_CLASSES)    # 29
NUM_STATIC_CLASSES  = len(STATIC_LETTERS)     # 29
NUM_MOTION_CLASSES  = len(MOTION_LETTERS)     #  2

# --- MediaPipe ---------------------------------------------------------------
NUM_LANDMARKS       = 21
LANDMARK_FEATURES   = NUM_LANDMARKS * 3       # 63
MP_DETECTION_CONF   = 0.7
MP_TRACKING_CONF    = 0.6

# --- Static dataset / model --------------------------------------------------
CSV_FILE            = os.path.join(PROCESSED_DIR, "landmarks_static.csv")
STATIC_MODEL_PATH   = os.path.join(MODELS_DIR,    "gesture_static.keras")
LABEL_ENCODER_PATH  = os.path.join(MODELS_DIR,    "label_encoder_static.npy")

# --- Motion (LSTM) dataset / model -------------------------------------------
MOTION_CSV_FILE     = os.path.join(PROCESSED_DIR, "landmarks_motion.csv")
MOTION_SEQ_FILE     = os.path.join(PROCESSED_DIR, "sequences_motion.npz")
MOTION_MODEL_PATH   = os.path.join(MODELS_DIR,    "gesture_motion.keras")
MOTION_ENCODER_PATH = os.path.join(MODELS_DIR,    "label_encoder_motion.npy")

MOTION_SEQUENCE_LENGTH   = 60    # frames per J/Z clip (~2 seconds at 30fps)
MOTION_SAMPLES_PER_CLASS = 150   # clips per letter

# --- Number (digit) dataset / model ------------------------------------------
NUMBER_CLASSES      = ["0","1","2","3","4","5","6","7","8","9"]
NUM_NUMBER_CLASSES  = len(NUMBER_CLASSES)     # 10

NUMBER_CSV_FILE      = os.path.join(PROCESSED_DIR, "landmarks_number.csv")
NUMBER_MODEL_PATH    = os.path.join(MODELS_DIR,    "gesture_number.keras")
NUMBER_ENCODER_PATH  = os.path.join(MODELS_DIR,    "label_encoder_number.npy")

# --- Training ----------------------------------------------------------------
TEST_SPLIT      = 0.2
VAL_SPLIT       = 0.1
RANDOM_STATE    = 42
BATCH_SIZE      = 32
EPOCHS          = 150
LEARNING_RATE   = 0.001
DROPOUT_RATE    = 0.4
PATIENCE        = 20

# --- Augmentation ------------------------------------------------------------
AUGMENTATION_FACTOR = 3
NOISE_SCALE         = 0.005
ROTATION_RANGE      = 15
SCALE_RANGE         = (0.85, 1.15)

# --- Inference ---------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80
MOTION_THRESHOLD     = 0.70
SMOOTHING_WINDOW     = 10
UNKNOWN_LABEL        = "Unknown"
SPEAK_LABEL          = "SPEAK"
DELETE_LABEL         = "DELETE"
SPEAK_HOLD_SECONDS   = 1.5    # hold thumbs-up for this long to trigger TTS

# --- Display -----------------------------------------------------------------
WINDOW_WIDTH    = 640
WINDOW_HEIGHT   = 480
FPS_TARGET      = 30

# --- Legacy aliases ----------------------------------------------------------
SEQUENCE_LENGTH       = MOTION_SEQUENCE_LENGTH
LSTM_MODEL_PATH       = MOTION_MODEL_PATH
SEQUENCE_FILE         = MOTION_SEQ_FILE
MIN_SAMPLES_PER_CLASS = 200