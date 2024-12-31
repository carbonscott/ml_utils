import jax
import jax.numpy as jnp
import jax.random as jr

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

import einops

'''
Write a multi layer perceptron (MLP) that has 4 layers with the layer size laid
out below.

layer_sizes = [784, 8192, 8192, 8192, 10]
batch_size = 8192

Please perform a loop of forward/backward/update for 30 times.
'''

def init_params(key, fan_in, fan_out):
    k1, k2 = jr.split(key, 2) # Each param group requires its own key
    W = jr.normal(k1, (fan_in, fan_out)) / jnp.sqrt(fan_in) # Simulate Xavir init
    b = jr.normal(k2, (fan_out,))
    return W, b

def compute(params, inputs):
    W, b = params
    return einops.einsum(
        W, inputs,
        'fan_in fan_out, fan_in -> fan_out'
    ) + b

def calc_loss(params, inputs, targets):
    '''
    params is a tuple of (W, b) pairs.
    '''
    layer_inputs = inputs
    compute_jit = jax.jit(jax.vmap(compute, in_axes=(None,0), out_axes=0))
    for param in params:
        logits = compute_jit(param, layer_inputs)
        activation = jax.nn.relu(logits)
        layer_inputs = activation
    loss = jnp.mean(jnp.sum((logits-targets)**2, axis=-1))
    return loss

# Set up a key
seed = 0
key = jr.PRNGKey(seed)

# Init params
layer_sizes = [784, 8192, 8192, 8192, 10]
key, *keys = jr.split(key, len(layer_sizes))
params = []
for k, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
    W, b = init_params(k, fan_in, fan_out)
    params.append((W, b))

# Define dummy data
key, *keys = jr.split(key, 3)
batch_size = 8192
input_dim  = layer_sizes[0]
output_dim = layer_sizes[-1]
inputs = jr.normal(keys[0], (batch_size, input_dim))
targets = jr.normal(keys[1], (batch_size, output_dim))

# Define training loop
lr = 1e-3
total_interations = 30
value_and_grad= jax.jit(jax.value_and_grad(calc_loss))
for iteration in range(total_interations):
    loss, grads = value_and_grad(params, inputs, targets)
    for enum_idx, ((W, b), (dW, db)) in enumerate(zip(params, grads)):
        W = W - lr * dW
        b = b - lr * db
        params[enum_idx] = (W, b)
    print(f"{iteration=} | loss={loss.item()} | dW={jnp.linalg.norm(dW).item()} | db={jnp.linalg.norm(db).item()} | dW/W={jnp.linalg.norm(dW)/jnp.linalg.norm(W)}")
