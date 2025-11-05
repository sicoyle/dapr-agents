from dapr_agents.llm.dapr import DaprChatClient
import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from dapr_agents.workflow.decorators import llm_activity
import logging
import time

load_dotenv()

runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


@runtime.workflow(name="analyze_topic")
def analyze_topic(ctx: DaprWorkflowContext, topic: str):
    # Each step is durable and can be retried
    outline = yield ctx.call_activity(create_outline, input=topic)
    if len(outline) > 0:
        print("Outline:", outline)
    blog_post = yield ctx.call_activity(write_blog, input=outline)
    if len(blog_post) > 0:
        print("Blog post:", blog_post)
    return blog_post


@runtime.activity(name="create_outline")
@llm_activity(
    prompt="Create a short outline about {topic}",
    llm=llm,
)
def create_outline(topic: str) -> str:
    pass


@runtime.activity(name="write_blog")
@llm_activity(
    prompt="Write a short blog post following this outline: {outline}",
    llm=llm,
)
def write_blog(outline: str) -> str:
    pass


if __name__ == "__main__":
    runtime.start()
    time.sleep(5)  # small grace period for runtime to be ready

    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(
        workflow=analyze_topic, input="AI Agents"
    )
    logging.info(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        logging.error("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        logging.info(f"\nResearch Report:\n{state.serialized_output}")
    else:
        logging.error(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            logging.error("Failure type: %s", fd.error_type)
            logging.error("Failure message: %s", fd.message)
            logging.error("Stack trace:\n%s", fd.stack_trace)
        else:
            logging.error("Custom status: %s", state.serialized_custom_status)

    runtime.shutdown()
