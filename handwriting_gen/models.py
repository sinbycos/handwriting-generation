from handwriting_gen.data_manager import get_model_dir
from handwriting_gen.unconditional_stroke_model import (
    UnconditionalStrokeModel,
    decode as unconditional_decode
)
from handwriting_gen.conditional_stroke_model import (
    ConditionalStrokeModel,
    decode as conditional_decode
)

MODEL_DIR = get_model_dir()


def generate_unconditionally(random_seed=1, **kwargs):
    """
    Input:
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    """
    model = UnconditionalStrokeModel.load(
        str(MODEL_DIR / 'unconditional-stroke-model'),
        batch_size=1, rnn_steps=1, is_train=False)
    return unconditional_decode(model, seed=random_seed, **kwargs)


def generate_conditionally(text='welcome', random_seed=1, **kwargs):
    """
    Input:
      text - str
      random_seed - integer

    Output:
      stroke - numpy 2D-array (T x 3)
    """
    model = ConditionalStrokeModel.load(
        str(MODEL_DIR / 'conditional-stroke-model'),
        batch_size=1, rnn_steps=1, is_train=False, char_seq_len=len(text) + 1)
    return conditional_decode(model, seed=random_seed, text=text, **kwargs)


# def recognize_stroke(stroke):
#     # Input:
#     #   stroke - numpy 2D-array (T x 3)

#     # Output:
#     #   text - str
#     return 'welcome'
