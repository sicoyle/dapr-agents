"""Shared fixtures for integration tests."""
import os
import subprocess
import pytest
import docker
import logging
import time
import requests
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def quickstarts_dir(project_root):
    """Get the quickstarts directory."""
    return project_root / "quickstarts"


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def elevenlabs_api_key():
    """Get ElevenLabs API key from environment."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        pytest.skip("ELEVENLABS_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def huggingface_api_key():
    """Get HuggingFace API key from environment."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        pytest.skip("HUGGINGFACE_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def nvidia_api_key():
    """Get NVIDIA API key from environment."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def docker_client():
    """Get Docker client."""
    client = docker.from_env()
    yield client
    client.close()


def _check_dapr_running(port: int = 3500) -> bool:
    """Check if Dapr sidecar is running on the given port."""
    return not _check_port_available(port)


def _ensure_dapr_initialized():
    """Ensure Dapr CLI is installed and Dapr is initialized (dapr init has been run)."""
    # Check if Dapr CLI is installed
    logger.info("Checking Dapr CLI installation...")
    result = subprocess.run(
        ["dapr", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        logger.error("Dapr CLI not found!")
        pytest.skip(
            "Dapr CLI not found. Please install Dapr CLI: https://docs.dapr.io/getting-started/install-dapr-cli/"
        )

    # Show Dapr version
    version_output = result.stdout.strip() or result.stderr.strip()
    logger.info(f"Dapr CLI found: {version_output}")

    # Check if Dapr is initialized by checking for dapr components directory
    # This is a lightweight check - if dapr init hasn't been run, components won't exist
    dapr_components_path = Path.home() / ".dapr" / "components"
    dapr_bin_path = Path.home() / ".dapr" / "bin"

    logger.info("Checking Dapr initialization status...")
    logger.info(f"  Dapr components path: {dapr_components_path}")
    logger.info(f"  Components exist: {dapr_components_path.exists()}")
    logger.info(f"  Dapr bin path: {dapr_bin_path}")
    logger.info(f"  Bin exists: {dapr_bin_path.exists()}")

    if not dapr_components_path.exists() or not dapr_bin_path.exists():
        logger.warning("Dapr not initialized. Running 'dapr init'...")
        logger.info("  This may take a few minutes (downloads Docker images)...")
        init_result = subprocess.run(
            ["dapr", "init"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for init
        )
        if init_result.returncode != 0:
            logger.error(f"Dapr initialization failed! Error: {init_result.stderr}")
            pytest.skip(
                f"Dapr initialization failed. Please run 'dapr init' manually.\n"
                f"Error: {init_result.stderr}"
            )
        logger.info("Dapr initialized successfully!")
    else:
        logger.info("Dapr already initialized.")


@pytest.fixture(scope="session")
def dapr_runtime(project_root):
    """
    Ensure Dapr is initialized. This fixture ensures Dapr CLI is available
    and Dapr is initialized. Tests that need a Dapr sidecar should use
    `use_dapr=True` in run_quickstart_script() which will start it via `dapr run`.
    """
    _ensure_dapr_initialized()
    yield


def _check_port_available(port: int) -> bool:
    """Check if a port is available (not in use)."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    return result != 0  # Port is available if connection fails


def _check_service_healthy(url: str, timeout: int = 2) -> bool:
    """Check if a service is healthy by making an HTTP request."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _get_or_create_container(
    client: docker.DockerClient,
    container_name: str,
    image: str,
    ports: dict,
    health_check_url: str,
    health_check_timeout: int = 30,
    environment: Optional[dict] = None,
) -> Optional[docker.models.containers.Container]:
    """
    Get existing container if it's running and healthy, otherwise create a new one.
    Returns None if port is already in use by a healthy service (we'll use that instead).

    Args:
        client: Docker client
        container_name: Name of the container
        image: Docker image to use
        ports: Port mapping dictionary
        health_check_url: URL to check for service health
        health_check_timeout: Maximum time to wait for service to be healthy
        environment: Optional environment variables

    Returns:
        Container object, or None if using existing service on port
    """
    # Check if container already exists
    try:
        existing_container = client.containers.get(container_name)

        # Check if container is running
        if existing_container.status == "running":
            # Check if service is healthy
            if _check_service_healthy(health_check_url, timeout=2):
                # Container exists, is running, and service is healthy - reuse it
                return existing_container
            else:
                # Container is running but service isn't healthy - restart it
                existing_container.restart()
                # Wait for service to be healthy
                for i in range(health_check_timeout):
                    if _check_service_healthy(health_check_url, timeout=2):
                        return existing_container
                    time.sleep(1)
                # If still not healthy, remove and create new
                existing_container.remove(force=True)
                time.sleep(1)
        else:
            # Container exists but isn't running - remove it
            existing_container.remove(force=True)
            time.sleep(1)
    except docker.errors.NotFound:
        # Container doesn't exist - that's fine, we'll create it
        pass

    # Check if port is already in use by something else
    # (not our container, but maybe another service)
    for host_port in ports.values():
        if not _check_port_available(host_port):
            # Port is in use - check if it's healthy
            if _check_service_healthy(health_check_url, timeout=2):
                # Something else is using the port and it's healthy - use it instead
                # Return None to indicate we're using existing service, not our container
                return None
            else:
                # Port is in use but service isn't healthy - try to create container anyway
                # (might fail, but that's okay - test will show the error)
                pass

    # Create new container
    container = client.containers.run(
        image,
        name=container_name,
        ports=ports,
        detach=True,
        remove=True,
        environment=environment,
    )

    # Wait for service to be ready
    for i in range(health_check_timeout):
        if _check_service_healthy(health_check_url, timeout=2):
            return container
        time.sleep(1)

    # Service didn't become healthy - clean up and return None
    # Test will still run, but will likely fail
    try:
        container.remove(force=True)
    except docker.errors.NotFound:
        pass

    return None


@pytest.fixture(scope="session")
def zipkin_service(docker_client):
    """Start Zipkin service for observability tests."""
    container = _get_or_create_container(
        client=docker_client,
        container_name="zipkin-test",
        image="openzipkin/zipkin:latest",
        ports={"9411/tcp": 9411},
        health_check_url="http://localhost:9411/health",
        health_check_timeout=30,
        environment=None,
    )

    # Verify service is healthy (whether from our container or existing service)
    if not _check_service_healthy("http://localhost:9411/health", timeout=2):
        pytest.skip("Zipkin service at http://localhost:9411 is not healthy")

    yield {
        "endpoint": "http://localhost:9411",
        "api_endpoint": "http://localhost:9411/api/v2/spans",
        "container": container,  # May be None if using existing service
    }

    # Only remove container if we created it (not if it was None/existing service)
    if container is not None:
        try:
            container.remove(force=True)
        except docker.errors.NotFound:
            pass


@pytest.fixture(scope="session")
def jaeger_service(docker_client):
    """Start Jaeger service for OTLP tests."""
    container = _get_or_create_container(
        client=docker_client,
        container_name="jaeger-test",
        image="jaegertracing/all-in-one:latest",
        ports={"4318/tcp": 4318, "16686/tcp": 16686},
        health_check_url="http://localhost:16686/",
        health_check_timeout=30,
        environment={"COLLECTOR_OTLP_ENABLED": "true"},
    )

    # Verify service is healthy (whether from our container or existing service)
    if not _check_service_healthy("http://localhost:16686/", timeout=2):
        pytest.skip("Jaeger service at http://localhost:16686 is not healthy")

    yield {
        "endpoint": "http://localhost:16686",
        "otlp_endpoint": "http://localhost:4318/v1/traces",
        "container": container,  # May be None if using existing service
    }

    # Only remove container if we created it (not if it was None/existing service)
    if container is not None:
        try:
            container.remove(force=True)
        except docker.errors.NotFound:
            pass


def run_quickstart_script(
    script_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: int = 60,
    use_dapr: bool = False,
    app_id: Optional[str] = None,
    resources_path: Optional[Path] = None,
    dapr_http_port: int = 3500,
) -> subprocess.CompletedProcess:
    """
    Run a quickstart script and return the result.

    Args:
        script_path: Path to the Python script to run
        cwd: Working directory (defaults to script's parent directory)
        env: Environment variables to set
        timeout: Timeout in seconds
        check: Whether to raise on non-zero return code
        use_dapr: Whether to run with `dapr run`
        app_id: Dapr app ID (required if use_dapr=True)
        resources_path: Path to Dapr resources/components (defaults to cwd/components)
        dapr_http_port: Dapr HTTP port (defaults to 3500)
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    cwd_path = cwd or script_path.parent

    if use_dapr:
        if not app_id:
            raise ValueError("app_id is required when use_dapr=True")
        if not resources_path:
            # Default to components directory in quickstart directory
            resources_path = cwd_path / "components"

        # Check if components directory exists and resolve env vars if needed
        # Get project root from the quickstarts_dir parent
        project_root_path = (
            cwd_path.parent.parent
            if "quickstarts" in str(cwd_path)
            else cwd_path.parent
        )
        resolve_script = project_root_path / "quickstarts" / "resolve_env_templates.py"
        if resolve_script.exists() and resources_path.exists():
            # Resolve environment variables in components
            resolve_result = subprocess.run(
                ["python", str(resolve_script), str(resources_path)],
                cwd=cwd_path,
                env=full_env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if resolve_result.returncode == 0:
                resolved_path = resolve_result.stdout.strip()
                if resolved_path and Path(resolved_path).exists():
                    resources_path = Path(resolved_path)

        # Build dapr run command
        cmd = [
            "dapr",
            "run",
            "--app-id",
            app_id,
            "--dapr-http-port",
            str(dapr_http_port),
            "--resources-path",
            str(resources_path),
            "--",
            "python",
            str(script_path),
        ]
    else:
        cmd = ["python", str(script_path)]

    result = subprocess.run(
        cmd,
        cwd=cwd_path,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script_path} failed with return code {result.returncode}.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    return result
