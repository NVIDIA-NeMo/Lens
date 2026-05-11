# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic convention attribute name constants for the NeMo ecosystem.

Follows OTel semconv naming: ``<namespace>.<entity>.<attribute>``.
"""

# ------------------------------------------------------------------ #
# Version tracking
# ------------------------------------------------------------------ #
# Tracks which upstream OTel semconv version these constants are based
# on. Update when syncing with a new semconv release.
#
# Reference: https://github.com/open-telemetry/semantic-conventions
# ------------------------------------------------------------------ #

SEMCONV_VERSION = "1.29.0"
"""OTel semantic conventions version these constants are aligned with.

Standard namespaces (gen_ai.*, k8s.*) follow the upstream spec at this
version. Custom namespaces (dl.*, rl.*, gym.*, slurm.*, nemo.*, wandb.*)
are NeMo-specific extensions that do not exist upstream.
"""

# ------------------------------------------------------------------ #
# Stability markers
# ------------------------------------------------------------------ #
# gen_ai.*  — Experimental (upstream, stabilising in semconv 1.30+)
# k8s.*     — Stable (upstream)
# dl.*      — NeMo custom (stable within NeMo ecosystem)
# rl.*      — NeMo custom (stable within NeMo ecosystem)
# gym.*     — NeMo custom (stable within NeMo ecosystem)
# slurm.*   — NeMo custom (stable within NeMo ecosystem)
# nemo.*    — NeMo custom (stable within NeMo ecosystem)
# wandb.*   — NeMo custom (stable within NeMo ecosystem)

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
# Run identification (nemo.*)
# ------------------------------------------------------------------ #

NEMO_RUN_ID = "nemo.run.id"
NEMO_USER_ID = "nemo.user.id"

# ------------------------------------------------------------------ #
# W&B Weave (wandb.*)
# ------------------------------------------------------------------ #

WANDB_ENTITY = "wandb.entity"
WANDB_PROJECT = "wandb.project"

# ------------------------------------------------------------------ #
# Kubernetes (k8s.*)  — standard OTel semconv
# ------------------------------------------------------------------ #

K8S_NAMESPACE_NAME = "k8s.namespace.name"
K8S_POD_NAME = "k8s.pod.name"
K8S_POD_UID = "k8s.pod.uid"
K8S_NODE_NAME = "k8s.node.name"
K8S_CONTAINER_NAME = "k8s.container.name"
K8S_JOB_NAME = "k8s.job.name"
