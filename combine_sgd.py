import json
import random
import sys
from typing import Dict
import collections
import copy
import json
import six
import tensorflow as tf
import torch
from tqdm.auto import tqdm

persona = json.load(open(sys.argv[1], "r"))
intent_description: Dict[str, str] = {
    "LookupSong": "search for a song",
    "PlaySong": "play the selected song on the device",
    "LookupMusic": "search for a song based on the name and optionally other attributes",
    "FindMovies": "find movies by genre and optionally director",
    "GetTimesForMovie": "get show times for a movie at a location on a given date",
    "FindAttractions": "browse attractions in a given city",
}
output = open("combine_sgd.json", "w")
transition_questions: Dict[str, str] = {
    k: f"Do you want to {v}?" for (k, v) in intent_description.items()
}
device = "cuda" if torch.cuda.is_available() else "cpu"
intent = {}
intents = {}
data = []
random.seed(26)

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    # Tensor.name is not supported in Eager mode.
    if tf.executing_eagerly():
      name = "get_shape_list"
    else:
      name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

for k in intent_description.keys():
    with open(f"sgd_intent_dialog/{k}_delex.json", "r") as f:
        intents[k] = json.load(f)
        random.shuffle(intents[k])

for d in tqdm(persona):
    intent_appear = False
    context = []
    for i, turn in enumerate(d):
        context.append(turn["text"])
        if len(turn["intent"]) != 0:
            last_chit_chat = d[i + 1]["text"] if (i + 1) < len(d) else ""
            intent_appear = True
            intent = {"type": turn["intent"], "position": i}
            whole_transition = (
                last_chit_chat + " " + transition_questions[turn["intent"][0]]
            )
            context.append(whole_transition)
            break

    if intent_appear and len(intents[intent["type"][0]]) != 0:
        sample = intents[intent["type"][0]][0]
        intents[intent["type"][0]] = intents[intent["type"][0]][1:]
        dialog = sample["dialogue"][sample["intent_pos"] :]
        context += dialog
        data.append(
            {"id": f"merge_{len(data):04d}", "dialog": context, "intent": intent}
        )
json.dump(data, output, indent=4)
