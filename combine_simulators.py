import json
import re
import sys
from typing import Dict

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import attention
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import tensorflow as tf


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def causal_depthwise_conv(x, context, kernel_size=3):
  """Causal depthwise convolution."""

  def scale_var(shift_distance):
    return mtf.get_variable(
        context.mesh,
        "conv_%s" % shift_distance,
        mtf.Shape(context.model.ensemble_dims + x.shape.dims[-1:]),
        initializer=tf.constant_initializer(0.5 if shift_distance ==
                                            0 else 0.5 / kernel_size),
        dtype=context.variable_dtype)

  ret = x * scale_var(0)
  for shift_distance in range(1, kernel_size):
    x = mtf.shift(x, 1, context.length_dim, wrap=False)
    ret += x * scale_var(shift_distance)
  return ret


def primer_norm(x, dim, epsilon=1e-6, name="layer_prepostprocess"):
  """Primer normalization over dimension `dim`.

  Args:
    x: a mtf.Tensor whose shape contains `dim`.
    dim: a mtf.Dimension.
    epsilon: a floating point number.
    name: a string used for tf.variable_scope.

  Returns:
    a mtf.Tensor with same shape as x.
  """
  with tf.variable_scope(name + "/primer_norm"):
    scale = mtf.get_variable(
        x.mesh,
        "primer_norm_scale",
        mtf.Shape([dim]),
        initializer=tf.ones_initializer(),
        activation_dtype=x.dtype)
    bias = mtf.get_variable(
        x.mesh,
        "primer_norm_bias",
        mtf.Shape([dim]),
        initializer=tf.zeros_initializer(),
        activation_dtype=x.dtype)
    reduced_shape = x.shape - dim
    mean = mtf.reduce_mean(x, output_shape=reduced_shape)
    mean_centered_x = x - mean
    pseudo_variance = mtf.reduce_mean(
        x * mean_centered_x, output_shape=reduced_shape)
    norm_x = mean_centered_x * mtf.rsqrt(pseudo_variance + epsilon)
    return norm_x * scale + bias

def sublayer_prime_norm(x,
                        layer_stack,
                        context,
                        epsilon=1e-6,
                        name="primer_norm"):
  """Sublayer wrapper around Primer norm.

  Args:
    x: an input mtf.Tensor.
    layer_stack: a LayerStack.
    context: a Context.
    epsilon: a float.
    name: a string.

  Returns:
    a mtf.Tensor.
  """
  del layer_stack
  model_dim = context.model.model_dim
  with tf.variable_scope(name):
    return primer_norm(x, model_dim, epsilon)

class DoubleHeadsAttentionLayer(transformer.TransformerLayer):
  """Attention with twice as many heads for Evolved Transformer."""

  def __init__(self, base_num_heads, key_value_size, dropout_rate):
    self._self_attention = transformer_layers.SelfAttention(
        num_heads=int(2 * base_num_heads),
        key_value_size=int(key_value_size / 2),
        dropout_rate=dropout_rate)

  def call(self, context, x, losses=None):
    """Call the layer."""
    with tf.variable_scope("double_heads_attention"):
      return self._self_attention.call(context, x, losses)


class MDHAParams(attention.AttentionParams):
  """Multi-DConv-Head Attention parameters."""

  def __init__(self,
               create_k_weights,
               **kwargs):
    self._create_k_weights = create_k_weights
    super().__init__(**kwargs)

  def init_weights(self):
    """Initialize projection matrices."""
    if mtf.layers.unit_scaling_convention():
      init = tf.random_normal_initializer(stddev=1.0)
      q_init = init
      kv_init = init
      o_init = init
    else:
      stddev = self.query_input_dim.size ** -0.5
      if self.fold_scaling_into_initializer:
        stddev *= self.key_dim.size ** -0.5
      q_init = tf.random_normal_initializer(stddev=stddev)
      kv_init = tf.random_normal_initializer(
          stddev=self.memory_input_dim.size ** -0.5)
      o_init = tf.random_normal_initializer(
          stddev=mtf.Shape(self.query_heads_dims + [self.value_dim]).size**-0.5)

    if self.make_attention_vars:
      if not self.no_query:
        self.wq = mtf.get_variable(
            self.mesh,
            "q",
            self.q_shape,
            initializer=q_init,
            dtype=self.variable_dtype)
      if self.shared_kv:
        self.wkv = mtf.get_variable(
            self.mesh,
            "kv",
            self.k_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
      else:
        if self._create_k_weights:
          self.wk = mtf.get_variable(
              self.mesh,
              "k",
              self.k_shape,
              initializer=kv_init,
              dtype=self.variable_dtype)
        self.wv = mtf.get_variable(
            self.mesh,
            "v",
            self.v_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
    self.wo = mtf.get_variable(
        self.mesh,
        "o",
        self.o_shape,
        initializer=o_init,
        dtype=self.variable_dtype)

  def mdha_q(self, query_antecedent, context):
    """MDHA Q projection."""
    ret = mtf.layers.us_einsum([query_antecedent, self.wq],
                               reduced_dims=[self.query_input_dim])
    with tf.variable_scope("q_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.q_dims)
    if not self.fold_scaling_into_initializer:
      ret *= self.key_dim.size**-0.5
    return ret

  def mdha_k(self, memory_antecedent, context):
    """MDHA K projection."""
    ret = mtf.layers.us_einsum([memory_antecedent, self.wk],
                               reduced_dims=[self.memory_input_dim])
    with tf.variable_scope("k_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.k_dims)
    return ret

  def mdha_v(self, memory_antecedent, context):
    """MDHA V projection."""
    ret = mtf.layers.us_einsum([memory_antecedent, self.wv],
                               reduced_dims=[self.memory_input_dim])
    with tf.variable_scope("v_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.v_dims)
    return ret

  def mdha_shared_qk(self, query_antecedent, context):
    """MDHA QK shared projection."""
    ret = mtf.layers.us_einsum([query_antecedent, self.wq],
                               reduced_dims=[self.query_input_dim])
    with tf.variable_scope("qk_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim

    q = mtf.layers.dense(
        ret,
        ret.shape.dims[-1:],
        use_bias=False,
        activation=None,
        variable_dtype=context.variable_dtype,
        reduced_dims=ret.shape.dims[-1:],
        name="q_solo_project",
        expert_dims=context.model.ensemble_dims)

    k = ret

    if self.combine_dims:
      q = mtf.replace_dimensions(q, q.shape.dims[-1], self.q_dims)
      k = mtf.replace_dimensions(k, k.shape.dims[-1], self.k_dims)
    if not self.fold_scaling_into_initializer:
      q *= self.key_dim.size**-0.5

    return q, k


persona = json.load(open(sys.argv[1], "r"))
intent_description: Dict[str, str] = {
    "LookupSong": "search for a song",
    "PlaySong": "play the selected song on the device",
    "LookupMusic": "search for a song based on the name and optionally other attributes",
    "FindMovies": "find movies by genre and optionally director",
    "GetTimesForMovie": "get show times for a movie at a location on a given date",
    "FindAttractions": "browse attractions in a given city",
}
output = open("combine_simulators.json", "w")
transition_questions: Dict[str, str] = {
    k: f"Do you want to {v}?" for (k, v) in intent_description.items()
}
device = "cuda" if torch.cuda.is_available() else "cpu"
end_keywords = ["goodbye", "bye"]
end_sentences = [
    "have a great day",
    "have a nice day",
    "have a good day",
    "have a wonderful day",
    "enjoy your day",
    "have a good one",
    "have a good time",
    "enjoy the rest of your day",
    "have a fantastic day",
    "i am glad i could help have a nice day",
]
intent = {}
data = []

for d in tqdm(persona):
    intent_appear = False
    history = []
    context = []
    for i, turn in enumerate(d):
        history.append(turn["text"])
        context.append(turn["text"])
        if len(turn["intent"]) != 0:
            last_chit_chat = d[i + 1]["text"] if (i + 1) < len(d) else ""
            intent_appear = True
            intent = {"type": turn["intent"], "position": i}
            whole_transition = (
                last_chit_chat + " " + transition_questions[turn["intent"][0]]
            )
            history.append(whole_transition)
            context.append(whole_transition)
            history = history[-3:]
            break

    if intent_appear:
        for _ in range(4):
            user_checkpoint = "stanleychu2/user_400M"
            user_tokenizer = AutoTokenizer.from_pretrained(
                user_checkpoint, use_fast=False
            )
            user = AutoModelForSeq2SeqLM.from_pretrained(user_checkpoint).to(device)
            user.eval()
            noise_lambda = 0.05
            for name, para in user.named_parameters():
                user.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * noise_lambda * torch.std(para)

            prefix = "user: "
            inputs = user_tokenizer(
                " ".join(history), max_length=128, truncation=True, return_tensors="pt"
            ).to(device)
            outputs = user.generate(
                **inputs,
                do_sample=True,
                top_k=120,
                no_repeat_ngram_size=2,
                min_length=1,
                max_length=64,
            ).squeeze(0)
            # 8010 = __END__
            if 8010 in outputs:
                print("__END__")
                break
            utterance = user_tokenizer.decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            history.append(utterance)
            context.append(utterance)
            history = history[-2:]

            system_checkpoint = "stanleychu2/system_400M"
            prefix = "sys: "
            sys_tokenizer = AutoTokenizer.from_pretrained(
                system_checkpoint, use_fast=False
            )
            system = AutoModelForSeq2SeqLM.from_pretrained(system_checkpoint).to(device)
            system.eval()

            noise_lambda = 0.05
            for name, para in system.named_parameters():
                system.state_dict()[name][:] += (torch.rand(para.size())-0.5) * noise_lambda * torch.std(para)

            inputs = sys_tokenizer(
                " ".join(history), max_length=128, truncation=True, return_tensors="pt"
            ).to(device)
            outputs = system.generate(
                **inputs,
                do_sample=True,
                num_beams=5,
                no_repeat_ngram_size=3,
                num_return_sequences=5,
                early_stopping=True,
                max_length=128,
            ).squeeze(0)
            utterance = user_tokenizer.decode(
                outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            processed_utterance = re.sub(r"[^\w\s]", "", utterance.lower())
            processed_last_utterance = re.sub(r"[^\w\s]", "", history[-2].lower())
            if (
                jaccard_similarity(
                    sys_tokenizer.tokenize(processed_last_utterance),
                    sys_tokenizer.tokenize(processed_utterance),
                )
                > 0.4
            ):
                print("REPEAT:", utterance)
                print("REPEAT:", history[-2])
                break
            history.append(utterance)
            context.append(utterance)
            history = history[-2:]
            if any([(k in utterance) for k in end_keywords]) or any(
                [
                    jaccard_similarity(
                        sys_tokenizer.tokenize(processed_utterance),
                        sys_tokenizer.tokenize(s),
                    )
                    > 0.2
                    for s in end_sentences
                ]
            ):
                print("RULE:", utterance)
                break

        print(context)
        data.append(
            {"id": f"simulateTOD_{len(data):04d}", "dialog": context, "intent": intent}
        )

json.dump(data, output, indent=4)
