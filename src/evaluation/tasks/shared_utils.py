import ast
import numpy as np


def convert_to_2d_array(preds):
    """
    Convert predictions to 2D numpy array.
    Only converts if the array contains string representations of lists.
    If already numeric, returns as-is.
    """
    if preds.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
        if len(preds) > 0 and isinstance(preds[0], str) and preds[0].strip().startswith('['):
            print("Detected string format - converting to numeric array")
            return np.array([ast.literal_eval(pred_str) for pred_str in preds])

    return np.stack(preds)

