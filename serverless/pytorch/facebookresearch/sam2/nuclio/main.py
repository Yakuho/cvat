# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os
import json

from utils import Profile
from nuclio_sdk import Context, Event
from model_handler import ModelHandler, SAMRedis
from cvat_sdk.api_client import Configuration, ApiClient

CVAT_API       = os.environ.get("CVAT_API")
CVAT_TOKEN     = os.environ.get("CVAT_TOKEN")
CVAT_USERNAME  = os.environ.get("CVAT_USERNAME")
CVAT_PASSWORD  = os.environ.get("CVAT_PASSWORD")

REDIS_HOST     = os.environ.get("REDIS_HOST", "")
REDIS_PORT     = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB       = int(os.environ.get("REDIS_DB", 0))

MODEL_CFG      = os.environ.get("MODEL_CFG", "sam2.1_hiera_l.yaml")
MODEL          = os.environ.get("MODEL",     "sam2.1_hiera_large.pt")


def init_context(context: Context):
    """Nuclio init function (run only once)"""
    context.logger.info("Init context...   0%")

    # Init CVAT Client
    context.logger.info("CVAT Client initialized ....")
    if not ((CVAT_USERNAME and CVAT_PASSWORD) or CVAT_TOKEN):
        raise EnvironmentError("CVAT Client Connect must set env (CVAT_USERNAME and CVAT_PASSWORD) or CVAT_TOKEN")
    config = Configuration(host=CVAT_API, username=CVAT_USERNAME, password=CVAT_PASSWORD, access_token=CVAT_TOKEN)
    client = ApiClient(config)
    client.users_api.retrieve_self()  # ping
    context.logger.info("CVAT Client initialized done")
    context.logger.info("Init context...  33%")

    # Init Redis Client
    context.logger.info("SAM2.1 redis cache initialized ....")
    redis = SAMRedis(REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD)
    redis.ping()  # ping
    context.logger.info("SAM2.1 redis cache initialized done")
    context.logger.info("Init context...  66%")

    # Init SAM2.1 Model object
    context.logger.info("SAM2.1 initialized ....")
    context.user_data.model = ModelHandler(MODEL, MODEL_CFG, client, redis)
    context.logger.info("SAM2.1 initialized done")
    context.logger.info("Init context... 100%")


def handler(context: Context, event: Event):
    """Nuclio call function"""
    context.logger.info("CVAT SAM2.1 AutoTrack called")

    jobId = event.body["jobId"]
    items = event.body["batch"]
    threshold = event.body.get("threshold")
    approx_threshold = event.body.get("approx_threshold")

    with Profile(device=context.user_data.model.device) as profile:
        results = list(context.user_data.model.handle(jobId, items, threshold, approx_threshold))
    context.logger.info("CVAT SAM2.1 AutoTrack called done, elapsed %d ms" % (profile.dt * 1e3))
    return context.Response(body=json.dumps(results), headers={}, content_type="application/json", status_code=200)
