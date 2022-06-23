import json
from argparse import ArgumentParser
from operator import itemgetter
from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline
import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import attention
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import tensorflow as tf

intent_questions: Dict[str, List[str]] = {
    "LookupSong": [
        "Is the intent asking about looking up songs ?",
        "Is the user asking about looking up songs ?",
        "Are there any websites which advertise or advertise songs for free?",
        "Is the users question about looking up songs?",
        "Is there a way to ask for help with the search of songs?",
        "How much time does someone waste searching for songs?",
        "Is the user asked about searching up song?",
        "Is a user ask about searching up songs?",
        "Does the user consider to look up songs?",
    ],
    "PlaySong": [
        "Is the intent asking about playing songs ?",
        "Is the user asking about playing songs ?",
        "Is the user asking about playing songs?",
        "Is your user asking about playing songs?",
        "Is the user asking about playing music?",
        "Why does the user ask about playing a song?",
        "Is a user asking about playing songs?",
        "Does my iPhone asks about playing songs?",
        "Does the user ask about playing songs?",
        "Is the user planning to playing songs ?",
    ],
    "LookupMusic": [
        "Is the intent asking about looking up music ?",
        "Is the user asking about looking up music ?",
        "Are you asking people to look up music?",
        "Is the user asking about looking up music?",
        "Is the user asking about searching for music?",
        "Why does it seem that people are obsessed with looking up music?",
        "Is the user asking about searching music?",
        "How s/he asked about searching up music?",
        "Will the user ask about finding other music?",
        "Is it helpful when I ask for help about searching for music on a website?",
        "Is it the user asking about looking up songs (or saying songs)?",
        "Why is the user so interested in looking up music?",
        "Does the user want to look up music ?",
    ],
    "FindMovies": [
        "Is the intent asking about finding movies ?",
        "Is the user asking about finding movies ?",
        "Does someone want to find a movie?",
        "Does the user ask about finding movies?",
        "Why does user ask to find movies?",
        "Is the user asking about finding movies?",
        "Is the user about looking movies and trawl?",
        "Is the user asking about finding movies. Is it true that it is the same question of no different people?",
        "When did you start a game and you start asking about movies?",
        "What are the users complaints about getting movies?",
        "Does the user hope to find movies ?",
    ],
    "GetTimesForMovie": [
        "Is the intent asking about getting the time for movies ?",
        "Is the user asking about getting the time for movies ?",
        "What's your question about getting the time for movies?",
        "Is my mom asking about getting time for movies?",
        "How can I get the time for movies?",
        "Is the user asking about getting the time for movies?",
        "Can you fix my time problem for movies?",
        "What is the thing the user is asking about getting a time in movie or TV watching?",
        "How do you determine if you have enough time to watch movies?",
        "Is the user asking about getting time for movies?",
        "If you are a movie watcher, would you like to give you a good amount of time for your filmmaking needs?",
        "Is getting the time for movies the purpose of the user?",
    ],
    "FindAttractions": [
        "Is the intent asking about finding attractions ?",
        "Is the user asking about finding attractions ?",
        "Is the user asking about finding attractions?",
        "Is the user asking about how to find attractions?",
        "How can I find an attraction?",
        "What are some of the common questions asked by a visitor about how to find an attraction?",
        "Is it the user asking about finding attractions?",
        "Is the User Asking about Theme parks?",
        "Does the user have trouble finding attractions ?",
    ],
}

sgd_intents: Dict[str, str] = {
    f"{intent}-{q}": q
    for intent, questions in intent_questions.items()
    for q in questions
}

def mdha_params(context,
                kv_dim,
                num_heads,
                num_memory_heads=0,
                shared_kv=False,
                no_query=False,
                combine_dims=True,
                keep_query_heads_dims=False,
                fold_scaling_into_initializer=True,
                create_k_weights=True):
  """Multi-DConv Head Attention parameters."""
  if num_heads == 1:
    query_heads_dims = None
    memory_heads_dims = None
  elif num_memory_heads == 0:
    query_heads_dims = [mtf.Dimension("heads", num_heads)]
    memory_heads_dims = query_heads_dims
  elif num_memory_heads == 1:
    query_heads_dims = [mtf.Dimension("heads", num_heads)]
    memory_heads_dims = None
  else:
    if num_heads % num_memory_heads != 0:
      raise ValueError("num_memory_heads must divide num_heads")
    memory_heads_dims = [mtf.Dimension("heads", num_memory_heads)]
    query_heads_dims = memory_heads_dims + [
        mtf.Dimension("query_heads", num_heads // num_memory_heads)]

class PrePostNormLayerStack(transformer.LayerStack):
  """Alternating pre and post normalization."""

  def call(self, context, x):
    """Call the layer stack."""
    x = self._call_sublayers(self._sublayers_initial, x, context, 0)
    context.layer_outputs.append(x)
    for lnum, layer in enumerate(self._layers):
      with tf.variable_scope(layer.name or ""):
        if self._recompute_grads:

          def fn(x, l=layer, c=context, lnum_arg=lnum):
            return self._layer_fn(x, l, c, lnum_arg)

          x = mtf.recompute_grad(fn, [x])
        else:
          x = self._layer_fn(x, layer, context, lnum)
      if lnum != len(self._layers) - 1:
        context.layer_outputs.append(x)
      context.layer_index += 1
    x = self._call_sublayers(self._sublayers_final, x, context, 0)
    x = transformer.sublayer_mask_padding(x, self, context)
    context.layer_outputs.append(x)
    return x

  # Pre and post norm.
  def _call_sublayers(self, sublayers, x, context, lnum):
    if lnum % 2 == 0:
      for s in sublayers:
        x = s(x, self, context)
    else:
      for s in [1, 2, 0, 3, 4]:
        x = sublayers[s](x, self, context)
    return x

  def _layer_fn(self, x, layer, context, lnum):
    context.current_layer = layer
    context.current_layer_input = x
    y = self._call_sublayers(self._sublayers_per_layer, x, context, lnum)
    if y.shape != x.shape:
      raise ValueError("Layer %s returned misshaped output x=%s y=%s" %
                       (layer.__class__.__name__, x, y))
    return y


@gin.configurable
class MDHA(transformer_layers.SelfAttention):
  """Multi-DConv-Head Attention."""

  def __init__(self,
               num_heads=8,
               num_memory_heads=0,
               key_value_size=128,
               shared_kv=False,
               dropout_rate=0.0,
               attention_kwargs=None,
               relative_attention_type=None,
               relative_attention_num_buckets=32,
               attention_func=None,
               combine_dims=True,
               keep_query_heads_dims=False,
               fold_scaling_into_initializer=True,
               z_loss_coeff=None,
               share_qk_rep=False):
    super().__init__(
        num_heads=num_heads,
        num_memory_heads=num_memory_heads,
        key_value_size=key_value_size,
        shared_kv=shared_kv,
        dropout_rate=dropout_rate,
        attention_kwargs=attention_kwargs,
        relative_attention_type=relative_attention_type,
        relative_attention_num_buckets=relative_attention_num_buckets,
        attention_func=attention_func,
        combine_dims=combine_dims,
        keep_query_heads_dims=keep_query_heads_dims,
        fold_scaling_into_initializer=fold_scaling_into_initializer,
        z_loss_coeff=z_loss_coeff)
    self.share_qk_rep = share_qk_rep

  def make_params(self, context):
    return mdha_params(
        context=context,
        kv_dim=self.kv_dim,
        num_heads=self.num_heads,
        num_memory_heads=self.num_memory_heads,
        shared_kv=self.shared_kv,
        combine_dims=self.combine_dims,
        keep_query_heads_dims=self.keep_query_heads_dims,
        fold_scaling_into_initializer=self.fold_scaling_into_initializer,
        create_k_weights=not self.share_qk_rep)

  @gin.configurable
  def call(self, context, x, losses=None):
    """Call the layer."""
    params = self.make_params(context)
    if self.share_qk_rep:
      q, k = params.mdha_shared_qk(x, context)
    else:
      q = params.mdha_q(x, context)
    memory_length = self.memory_length(context)
    if context.mode == "incremental":
      m = x
    else:
      if self.share_qk_rep:
        k = mtf.replace_dimensions(k, context.length_dim, memory_length)
      m = mtf.replace_dimensions(x, context.length_dim, memory_length)
    if self.shared_kv:
      kv = params.compute_kv(m)
    else:
      if not self.share_qk_rep:
        k = params.mdha_k(m, context)
      v = params.mdha_v(m, context)
    if context.mode == "incremental":
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      if self.shared_kv:
        old_kv = context.get_states(1)
        kv = old_kv * inv_one_hot + kv * one_hot
      else:
        old_k, old_v = context.get_states(2)
        k = old_k * inv_one_hot + k * one_hot
        v = old_v * inv_one_hot + v * one_hot
      memory_position = mtf.range(context.mesh, memory_length, tf.int32)
    else:
      memory_position = self.rename_length_to_memory_length(
          context.position, context)
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([kv] if self.shared_kv else [k, v])
    if self.shared_kv:
      k = kv
      v = kv
    o = self.attention_fn(
        q,
        k,
        v,
        context=context,
        memory_length_dim=memory_length,
        key_dim=self.kv_dim,
        value_dim=self.kv_dim,
        bias=self.compute_bias(context, memory_position, x,
                               params.query_heads_dims, q),
        **self.attention_kwargs_from_context(context))
    attention_output_shape = self.expected_attention_output_shape(x, params)
    attention_output = params.compute_output(
        o, output_shape=attention_output_shape)
    return self.layer_output_from_attention_output(context, attention_output,
                                                   losses)


def classify_intent(example: Dict) -> Dict:

    instances = [
        (idx, intent, f"yes. no. {turn}", question)
        for idx, turn in enumerate(example)
        for intent, question in sgd_intents.items()
    ]
    results = nlp(
        question=list(map(itemgetter(-1), instances)),
        context=list(map(itemgetter(-2), instances)),
    )
    mappings = {i[:2]: r["answer"] for i, r in zip(instances, results)}
    new_dialog = [
        {
            "id": idx,
            "text": turn,
            "intent": list(
                set(
                    [
                        intent.split("-")[0]
                        for intent in sgd_intents
                        if mappings.get((idx, intent), None) == "yes."
                    ]
                )
            ),
        }
        for idx, turn in enumerate(example)
    ]

    return new_dialog


parser = ArgumentParser()
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--data_file", type=str, default="blender.jsonl")
parser.add_argument("--output_file", type=str, default="intent_sample.json")
args = parser.parse_args()

MODEL_NAME = "adamlin/distilbert-base-cased-sgd_qa-step5000"
REVISION = "negative_sample-questions"
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, revision=REVISION)

noise_lambda = 0.05

for name, para in model.named_parameters():
    model.state_dict()[name][:] += (torch.rand(para.size())-0.5) * noise_lambda * torch.std(para)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
nlp = QuestionAnsweringPipeline(model, tokenizer, device=args.device)

samples = [json.loads(i) for i in open(args.data_file, "r")]
utterances = []
for s in samples:
    tempt = []
    for d in s["dialog"]:
        p1, p2 = d[0]["text"], d[1]["text"]
        tempt.append(p1)
        tempt.append(p2)
    utterances.append(tempt)
intent_samples = []
for e in tqdm(utterances):
    intent_samples.append(classify_intent(e))

json.dump(intent_samples, open(args.output_file, "w"))
