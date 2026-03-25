# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Semantic convention attribute name constants for the NeMo ecosystem.

Follows OTel semconv naming: ``<namespace>.<entity>.<attribute>``.
"""


# ------------------------------------------------------------------ #
# Distributed learning (dl.*)
# ------------------------------------------------------------------ #

DL_RANK = "dl.rank"
DL_WORLD_SIZE = "dl.world_size"
DL_LOCAL_RANK = "dl.local_rank"
DL_DATA_PARALLEL_RANK = "dl.data_parallel.rank"
DL_DATA_PARALLEL_SIZE = "dl.data_parallel.size"
DL_TENSOR_PARALLEL_RANK = "dl.tensor_parallel.rank"
DL_TENSOR_PARALLEL_SIZE = "dl.tensor_parallel.size"
DL_PIPELINE_PARALLEL_RANK = "dl.pipeline_parallel.rank"
DL_PIPELINE_PARALLEL_SIZE = "dl.pipeline_parallel.size"
DL_ITERATION = "dl.iteration"
DL_LOSS = "dl.loss"
DL_GRAD_NORM = "dl.grad_norm"
DL_LEARNING_RATE = "dl.learning_rate"
DL_THROUGHPUT_TFLOPS = "dl.throughput_tflops"
DL_THROUGHPUT_TOKENS_PER_SEC = "dl.throughput_tokens_per_sec"
DL_BATCH_SIZE = "dl.batch_size"
DL_SEQUENCE_LENGTH = "dl.sequence_length"
DL_MICROBATCH_ID = "dl.microbatch_id"

# ------------------------------------------------------------------ #
# GenAI semconv (gen_ai.*) — standard OTel
# ------------------------------------------------------------------ #

GENAI_OPERATION_NAME = "gen_ai.operation.name"
GENAI_PROVIDER_NAME = "gen_ai.provider.name"
GENAI_REQUEST_MODEL = "gen_ai.request.model"
GENAI_TOKEN_TYPE = "gen_ai.token.type"

# ------------------------------------------------------------------ #
# Reinforcement learning (rl.*)
# ------------------------------------------------------------------ #

RL_ALGORITHM = "rl.algorithm"
RL_REWARD = "rl.reward"
RL_REWARD_MEAN = "rl.reward.mean"
RL_KL_DIVERGENCE = "rl.kl_divergence"
RL_POLICY_LOSS = "rl.policy_loss"
RL_VALUE_LOSS = "rl.value_loss"
RL_ENTROPY = "rl.entropy"
RL_GENERATION_BACKEND = "rl.generation.backend"
RL_NUM_ROLLOUTS = "rl.num_rollouts"
RL_RESPONSE_LENGTH_MEAN = "rl.response_length.mean"

# ------------------------------------------------------------------ #
# Gym (gym.*)
# ------------------------------------------------------------------ #

GYM_SERVER_NAME = "gym.server.name"
GYM_SERVER_TYPE = "gym.server.type"
GYM_NUM_SERVERS = "gym.num_servers"
GYM_ROLLOUT_BATCH_SIZE = "gym.rollout.batch_size"
GYM_VERIFY_SUCCESS_RATE = "gym.verify.success_rate"

# ------------------------------------------------------------------ #
# SLURM (slurm.*)
# ------------------------------------------------------------------ #

SLURM_JOB_ID = "slurm.job.id"
SLURM_JOB_NAME = "slurm.job.name"
SLURM_NODELIST = "slurm.nodelist"
SLURM_NNODES = "slurm.nnodes"
SLURM_NTASKS = "slurm.ntasks"
SLURM_PARTITION = "slurm.partition"
SLURM_CLUSTER_NAME = "slurm.cluster.name"

# ------------------------------------------------------------------ #
# Kubernetes (k8s.*)  — standard OTel semconv
# ------------------------------------------------------------------ #

K8S_NAMESPACE_NAME = "k8s.namespace.name"
K8S_POD_NAME = "k8s.pod.name"
K8S_POD_UID = "k8s.pod.uid"
K8S_NODE_NAME = "k8s.node.name"
K8S_CONTAINER_NAME = "k8s.container.name"
K8S_JOB_NAME = "k8s.job.name"
