"""
Game constants for the corrected bluff game implementation.
"""

NUM_FACES = 2       # Number of faces on each die
NUM_DICES = 1       # Number of dice per player  
NUM_PLAYERS = 2     # Number of players

# Maximum possible claim: (total_dice * face_value)
MAX_CLAIM_QUANTITY = NUM_PLAYERS * NUM_DICES
MAX_CLAIM_FACE = NUM_FACES

# Total number of possible claims (quantity, face) pairs
TOTAL_CLAIMS = MAX_CLAIM_QUANTITY * NUM_FACES

EPSILON = 1e-8      # Small value for numerical stability
