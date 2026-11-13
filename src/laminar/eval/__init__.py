# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Standalone checkpoint evaluation entry point and orchestration."""

from laminar.eval.server import EvaluationQueueClient, resolve_queue_endpoint

__all__ = ["EvaluationQueueClient", "resolve_queue_endpoint"]
