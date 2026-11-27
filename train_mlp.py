import numpy as np
from collections import deque
from common import normalize

WINDOW = 30
MAX_NAN_RATIO = 0.2


class SequenceBuffer:
    """Fenêtre glissante de 30 images normalisées (33,2) -> (30,66)."""
    def __init__(self):
        self.buf = deque(maxlen=WINDOW)
        self.missing = 0

    def push(self, kp):
        """kp : (33,2) ou (33,3), ou None si personne. Renvoie (30,66) quand la fenêtre est prête, sinon None."""
        if kp is None:
            self.missing += 1
            if self.missing > 5:
                self.buf.clear()
            return None
        self.missing = 0
        self.buf.append(normalize(np.asarray(kp, np.float32)).reshape(-1))
        if len(self.buf) < WINDOW:
            return None
        seq = np.stack(self.buf)
        if np.isnan(seq).mean() > MAX_NAN_RATIO:
            return None
        return np.nan_to_num(seq)  # remplacer par interpolation dans la version finale


def predict_movement(model, seq):
    """seq (30, 66) -> (classe, probabilité). Squelette : valeur bidon."""
    return 0, 1.0