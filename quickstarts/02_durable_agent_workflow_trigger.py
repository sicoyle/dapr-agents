#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Trigger script for 02_durable_agent_workflow.py.

Run 02_durable_agent_workflow.py first (in a separate terminal), then run this
script to schedule a task for the WeatherAgent via a Dapr workflow that calls
the WeatherAgent's workflow as a child workflow using call_child_workflow.

Usage:
    # Terminal 1
    uv run dapr run --app-id weather-agent --resources-path resources -- python 02_durable_agent_workflow.py

    # Terminal 2
    uv run dapr run --app-id workflow-trigger --dapr-http-port 3501 -- python 02_durable_agent_workflow_trigger.py
"""

import dapr.ext.workflow as wf

wfr = wf.WorkflowRuntime()

WEATHER_AGENT_APP_ID = "weather-agent"
WEATHER_AGENT_WORKFLOW = "dapr.agents.WeatherAgent.workflow"


@wfr.workflow
def trigger_workflow(ctx: wf.DaprWorkflowContext):
    """Calls the WeatherAgent's workflow as a child workflow in a separate app."""
    result = yield ctx.call_child_workflow(
        workflow=WEATHER_AGENT_WORKFLOW,
        input={"task": "What is the weather in London?"},
        app_id=WEATHER_AGENT_APP_ID,
    )
    return result


def main() -> None:
    wfr.start()

    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(workflow=trigger_workflow)
    print(f"Scheduled trigger workflow instance: {instance_id}")

    state = client.wait_for_workflow_completion(
        instance_id=instance_id,
        timeout_in_seconds=120,
        fetch_payloads=True,
    )
    print(f"Workflow status: {state.runtime_status.name}")
    if state.serialized_output:
        print(f"Result: {state.serialized_output}")

    wfr.shutdown()


if __name__ == "__main__":
    main()
