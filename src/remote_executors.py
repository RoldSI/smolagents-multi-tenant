#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import json
import pickle
import re
import time
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple

import PIL.Image
import requests

from .local_python_executor import PythonExecutor
from .monitoring import LogLevel
from .tools import Tool, get_tools_definition_code
from .utils import AgentError


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    def __init__(self, additional_imports: List[str], logger):
        self.additional_imports = additional_imports
        self.logger = logger
        self.logger.log("Initializing executor, hold on...")
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$", re.M)
        self.installed_packages = []

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        raise NotImplementedError

    def send_tools(self, tools: Dict[str, Tool]):
        tool_definition_code = get_tools_definition_code(tools)

        packages_to_install = set()
        for tool in tools.values():
            for package in tool.to_dict()["requirements"]:
                if package not in self.installed_packages:
                    packages_to_install.add(package)
                    self.installed_packages.append(package)

        execution = self.run_code_raise_errors(
            f"!pip install {' '.join(packages_to_install)}\n" + tool_definition_code
        )
        self.logger.log(execution[1])

    def send_variables(self, variables: dict):
        """
        Send variables to the kernel namespace using pickle.
        """
        pickled_vars = base64.b64encode(pickle.dumps(variables)).decode()
        code = f"""
import pickle, base64
vars_dict = pickle.loads(base64.b64decode('{pickled_vars}'))
locals().update(vars_dict)
"""
        self.run_code_raise_errors(code)

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """Check if code is a final answer and run it accordingly"""
        is_final_answer = bool(self.final_answer_pattern.search(code_action))
        output = self.run_code_raise_errors(code_action, return_final_answer=is_final_answer)
        return output[0], output[1], is_final_answer

    def install_packages(self, additional_imports: List[str]):
        additional_imports = additional_imports + ["smolagents"]
        _, execution_logs = self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
        self.logger.log(execution_logs)
        return additional_imports


class E2BExecutor(RemotePythonExecutor):
    """
    Executes Python code using E2B.

    Args:
        additional_imports (`list[str]`): Additional imports to install.
        logger (`Logger`): Logger to use.
        **kwargs: Additional arguments to pass to the E2B Sandbox.
    """

    def __init__(self, additional_imports: List[str], logger, **kwargs):
        super().__init__(additional_imports, logger)
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install 'smolagents[e2b]'`"""
            )
        self.sandbox = Sandbox(**kwargs)
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("E2B is running", level=LogLevel.INFO)

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        execution = self.sandbox.run_code(
            code,
        )
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs
            logs += "Executing code yielded an error:"
            logs += execution.error.name + "\n"
            logs += execution.error.value
            logs += execution.error.traceback
            raise AgentError(logs, self.logger)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        if not execution.results:
            return None, execution_logs
        else:
            for result in execution.results:
                if result.is_main_result:
                    for attribute_name in ["jpeg", "png"]:
                        if getattr(result, attribute_name) is not None:
                            image_output = getattr(result, attribute_name)
                            decoded_bytes = base64.b64decode(image_output.encode("utf-8"))
                            return PIL.Image.open(BytesIO(decoded_bytes)), execution_logs
                    for attribute_name in [
                        "chart",
                        "data",
                        "html",
                        "javascript",
                        "json",
                        "latex",
                        "markdown",
                        "pdf",
                        "svg",
                        "text",
                    ]:
                        if getattr(result, attribute_name) is not None:
                            return getattr(result, attribute_name), execution_logs
            if return_final_answer:
                raise AgentError("No main result returned by executor!", self.logger)
            return None, execution_logs


class DockerExecutor(RemotePythonExecutor):
    """
    Executes Python code using Jupyter Kernel Gateway in a Docker container.
    """

    def __init__(
        self,
        additional_imports: List[str],
        logger,
        host: str = "127.0.0.1",
        port: int = 8888,
    ):
        """
        Initialize the Docker-based Jupyter Kernel Gateway executor.
        """
        super().__init__(additional_imports, logger)
        try:
            import docker
            from websocket import create_connection
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'docker' extra to use DockerExecutor: `pip install 'smolagents[docker]'`"
            )
        self.host = host
        self.port = port

        # Initialize Docker
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.") from e

        # Build and start container
        try:
            self.logger.log("Building Docker image...", level=LogLevel.INFO)
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            if not dockerfile_path.exists():
                with open(dockerfile_path, "w") as f:
                    f.write("""FROM python:3.12-slim

RUN pip install jupyter_kernel_gateway requests numpy pandas
RUN pip install jupyter_client notebook

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
""")
            _, build_logs = self.client.images.build(
                path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag="jupyter-kernel"
            )
            self.logger.log(build_logs, level=LogLevel.DEBUG)

            self.logger.log(f"Starting container on {host}:{port}...", level=LogLevel.INFO)
            self.container = self.client.containers.run(
                "jupyter-kernel", ports={"8888/tcp": (host, port)}, detach=True
            )

            retries = 0
            while self.container.status != "running" and retries < 5:
                self.logger.log(f"Container status: {self.container.status}, waiting...", level=LogLevel.INFO)
                time.sleep(1)
                self.container.reload()
                retries += 1

            self.base_url = f"http://{host}:{port}"

            # Create new kernel via HTTP
            r = requests.post(f"{self.base_url}/api/kernels")
            if r.status_code != 201:
                error_details = {
                    "status_code": r.status_code,
                    "headers": dict(r.headers),
                    "url": r.url,
                    "body": r.text,
                    "request_method": r.request.method,
                    "request_headers": dict(r.request.headers),
                    "request_body": r.request.body,
                }
                self.logger.log_error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
                raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None

            self.kernel_id = r.json()["id"]

            ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"
            self.ws = create_connection(ws_url)

            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log(
                f"Container {self.container.short_id} is running with kernel {self.kernel_id}", level=LogLevel.INFO
            )

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def run_code_raise_errors(self, code_action: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        """
        Execute code and return result based on whether it's a final answer.
        """
        try:
            if return_final_answer:
                match = self.final_answer_pattern.search(code_action)
                if match:
                    pre_final_answer_code = self.final_answer_pattern.sub("", code_action)
                    result_expr = match.group(1)
                    wrapped_code = pre_final_answer_code + dedent(f"""
                        import pickle, base64
                        _result = {result_expr}
                        print("RESULT_PICKLE:" + base64.b64encode(pickle.dumps(_result)).decode())
                        """)
            else:
                wrapped_code = code_action

            # Send execute request
            msg_id = self._send_execute_request(wrapped_code)

            # Collect output and results
            outputs = []
            result = None
            waiting_for_idle = False

            while True:
                msg = json.loads(self.ws.recv())
                msg_type = msg.get("msg_type", "")
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")

                # Only process messages related to our execute request
                if parent_msg_id != msg_id:
                    continue

                if msg_type == "stream":
                    text = msg["content"]["text"]
                    if return_final_answer and text.startswith("RESULT_PICKLE:"):
                        pickle_data = text[len("RESULT_PICKLE:") :].strip()
                        result = pickle.loads(base64.b64decode(pickle_data))
                        waiting_for_idle = True
                    else:
                        outputs.append(text)
                elif msg_type == "error":
                    traceback = msg["content"].get("traceback", [])
                    raise AgentError("\n".join(traceback), self.logger)
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    if not return_final_answer or waiting_for_idle:
                        break

            return result, "".join(outputs)

        except Exception as e:
            self.logger.log_error(f"Code execution failed: {e}")
            raise

    def _send_execute_request(self, code: str) -> str:
        """Send code execution request to kernel."""
        import uuid

        # Generate a unique message ID
        msg_id = str(uuid.uuid4())

        # Create execute request
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        self.ws.send(json.dumps(execute_request))
        return msg_id

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "container"):
                self.logger.log(f"Stopping and removing container {self.container.short_id}...", level=LogLevel.INFO)
                self.container.stop()
                self.container.remove()
                self.logger.log("Container cleanup completed", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


class FirecrackerExecutor(RemotePythonExecutor):
    """
    Executes Python code using Firecracker microVMs.
    
    This executor creates and manages Firecracker microVMs to execute code in an isolated environment.
    Each profile can have its own Firecracker image with specific tools and dependencies pre-installed.
    
    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger instance to use for logging.
        profile_id (`str`, optional): Profile ID to use a specific pre-built Firecracker image.
        base_image_path (`str`, optional): Path to base Firecracker image if not using a profile.
        vm_memory_mb (`int`, optional): Amount of memory to allocate to the VM in MB. Defaults to 2048.
        vm_cpu_count (`int`, optional): Number of CPUs to allocate to the VM. Defaults to 2.
    """
    
    def __init__(
        self,
        additional_imports: List[str],
        logger,
        profile_id: str = None,
        base_image_path: str = None,
        vm_memory_mb: int = 2048,
        vm_cpu_count: int = 2,
    ):
        super().__init__(additional_imports, logger)
        
        try:
            import subprocess
            import tempfile
            import os
            import uuid
            import signal
            import atexit
            from pathlib import Path
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Failed to import required standard libraries")
        
        self.subprocess = subprocess
        self.tempfile = tempfile
        self.os = os
        self.uuid = uuid
        self.signal = signal
        self.atexit = atexit
        self.Path = Path
        
        # Check Firecracker availability
        try:
            self.subprocess.run(["which", "firecracker"], check=True, capture_output=True)
        except (self.subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Firecracker not found. Please install Firecracker: "
                "https://github.com/firecracker-microvm/firecracker/releases"
            )
        
        # Configuration
        self.profile_id = profile_id
        self.base_image_path = base_image_path
        self.vm_memory_mb = vm_memory_mb
        self.vm_cpu_count = vm_cpu_count
        self.vm_id = str(self.uuid.uuid4())
        
        # Paths
        self.temp_dir = self.tempfile.mkdtemp(prefix=f"firecracker-{self.vm_id}-")
        self.socket_path = f"{self.temp_dir}/firecracker.socket"
        self.log_path = f"{self.temp_dir}/firecracker.log"
        self.metrics_path = f"{self.temp_dir}/metrics.fifo"
        
        # VM state
        self.firecracker_process = None
        self.api_endpoint = f"http://localhost:8000/vm/{self.vm_id}"
        
        # Setup VM
        self._setup_vm()
        
        # Register cleanup handler
        self.atexit.register(self.cleanup)
        
        # Install packages
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("Firecracker VM is running", level=LogLevel.INFO)
    
    def _setup_vm(self):
        """Set up the Firecracker VM."""
        
        # Prepare VM image
        if self.profile_id:
            # Get profile image path from multi.py
            from multi import Multi
            multi = Multi()
            profile = multi.get_profile(self.profile_id)
            image_path = f"profiles/{self.profile_id}/rootfs.ext4"
            if not self.Path(image_path).exists():
                raise FileNotFoundError(f"Profile image not found: {image_path}")
        elif self.base_image_path:
            image_path = self.base_image_path
            if not self.Path(image_path).exists():
                raise FileNotFoundError(f"Base image not found: {image_path}")
        else:
            # Use default image
            image_path = "profiles/default/rootfs.ext4"
            if not self.Path(image_path).exists():
                self._create_default_image()
                image_path = "profiles/default/rootfs.ext4"
        
        # Create a copy of the image for this VM
        vm_image_path = f"{self.temp_dir}/rootfs.ext4"
        self.subprocess.run(["cp", image_path, vm_image_path], check=True)
        
        # Create API socket
        self.subprocess.run(["rm", "-f", self.socket_path], check=True, stderr=self.subprocess.DEVNULL)
        
        # Create metrics fifo
        self.subprocess.run(["mkfifo", self.metrics_path], check=True)
        
        # Start Firecracker process
        self.firecracker_process = self.subprocess.Popen(
            ["firecracker", "--api-sock", self.socket_path, "--log-path", self.log_path],
            stdout=self.subprocess.PIPE,
            stderr=self.subprocess.PIPE,
            text=True,
        )
        
        # Wait for socket to be available
        for _ in range(30):  # 3 seconds timeout
            if self.Path(self.socket_path).exists():
                break
            time.sleep(0.1)
        else:
            self.cleanup()
            raise RuntimeError("Firecracker failed to start")
        
        # Configure VM
        self._configure_vm(vm_image_path)
        
        # Start VM helper service
        self._start_helper_service()
    
    def _create_default_image(self):
        """Create default Firecracker image with Python and common packages."""
        self.logger.log("Creating default Firecracker image...", level=LogLevel.INFO)
        
        # Create profiles directory if it doesn't exist
        self.os.makedirs("profiles/default", exist_ok=True)
        
        # Create a basic ext4 image with Python and common packages
        with self.tempfile.TemporaryDirectory() as temp_dir:
            # Create rootfs directory
            rootfs_dir = f"{temp_dir}/rootfs"
            self.os.makedirs(rootfs_dir)
            
            # Use debootstrap to create a minimal Debian system
            self.subprocess.run(
                ["debootstrap", "--variant=minbase", "bullseye", rootfs_dir],
                check=True,
            )
            
            # Install Python and common packages
            self.subprocess.run(
                ["chroot", rootfs_dir, "apt-get", "update"],
                check=True,
            )
            self.subprocess.run(
                [
                    "chroot", rootfs_dir, "apt-get", "install", "-y",
                    "python3", "python3-pip", "python3-venv", "ca-certificates",
                    "curl", "wget", "git", "vim", "nano"
                ],
                check=True,
            )
            
            # Install common Python packages
            self.subprocess.run(
                [
                    "chroot", rootfs_dir, "pip3", "install", 
                    "numpy", "pandas", "matplotlib", "requests", "flask", "fastapi",
                    "uvicorn", "gunicorn", "pydantic", "websockets"
                ],
                check=True,
            )
            
            # Create helper service
            helper_service_path = f"{rootfs_dir}/usr/local/bin/vm_helper.py"
            with open(helper_service_path, "w") as f:
                f.write(self._get_helper_service_code())
            self.subprocess.run(["chmod", "+x", helper_service_path], check=True)
            
            # Create systemd service for helper
            service_dir = f"{rootfs_dir}/etc/systemd/system"
            self.os.makedirs(service_dir, exist_ok=True)
            with open(f"{service_dir}/vm-helper.service", "w") as f:
                f.write(dedent("""
                [Unit]
                Description=VM Helper Service
                After=network.target
                
                [Service]
                ExecStart=/usr/bin/python3 /usr/local/bin/vm_helper.py
                Restart=always
                User=root
                
                [Install]
                WantedBy=multi-user.target
                """))
            
            # Enable service
            self.subprocess.run(
                ["chroot", rootfs_dir, "systemctl", "enable", "vm-helper.service"],
                check=True,
            )
            
            # Create ext4 image
            image_size_mb = 4096  # 4GB image
            image_path = "profiles/default/rootfs.ext4"
            self.subprocess.run(
                ["dd", "if=/dev/zero", f"of={image_path}", "bs=1M", f"count={image_size_mb}"],
                check=True,
            )
            self.subprocess.run(
                ["mkfs.ext4", image_path],
                check=True,
            )
            
            # Mount and copy files
            with self.tempfile.TemporaryDirectory() as mount_dir:
                self.subprocess.run(
                    ["mount", image_path, mount_dir],
                    check=True,
                )
                try:
                    self.subprocess.run(
                        ["cp", "-a", f"{rootfs_dir}/.", mount_dir],
                        check=True,
                    )
                finally:
                    self.subprocess.run(
                        ["umount", mount_dir],
                        check=True,
                    )
        
        self.logger.log("Default Firecracker image created", level=LogLevel.INFO)
    
    def _get_helper_service_code(self) -> str:
        """Get the code for the VM helper service."""
        return dedent("""
        #!/usr/bin/env python3
        import base64
        import json
        import pickle
        import traceback
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class VMHelperHandler(BaseHTTPRequestHandler):
            def _send_response(self, status_code, content):
                self.send_response(status_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(content).encode())
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length).decode()
                request = json.loads(body)
                
                if self.path == '/execute':
                    try:
                        code = request.get('code', '')
                        return_final_answer = request.get('return_final_answer', False)
                        
                        # Create a new locals dict for this execution
                        local_vars = {}
                        
                        # Execute the code and capture stdout/stderr
                        from io import StringIO
                        import sys
                        
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        stdout = StringIO()
                        stderr = StringIO()
                        
                        sys.stdout = stdout
                        sys.stderr = stderr
                        
                        try:
                            exec(code, globals(), local_vars)
                            if return_final_answer and 'final_answer' in local_vars:
                                result = local_vars['final_answer']
                            else:
                                result = None
                            
                            # Try to get the last expression's value
                            last_value = None
                            try:
                                import ast
                                tree = ast.parse(code)
                                if tree.body and isinstance(tree.body[-1], ast.Expr):
                                    last_expr = code.strip().split('\\n')[-1]
                                    last_value = eval(last_expr, globals(), local_vars)
                            except:
                                pass
                            
                            if result is None and last_value is not None:
                                result = last_value
                            
                            success = True
                            error = None
                        except Exception as e:
                            success = False
                            error = {
                                'name': type(e).__name__,
                                'value': str(e),
                                'traceback': traceback.format_exc()
                            }
                            result = None
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                        
                        # Try to pickle the result
                        if result is not None:
                            try:
                                result_pickled = base64.b64encode(pickle.dumps(result)).decode()
                            except:
                                result_pickled = None
                        else:
                            result_pickled = None
                        
                        response = {
                            'success': success,
                            'stdout': stdout.getvalue(),
                            'stderr': stderr.getvalue(),
                            'result': result_pickled,
                            'error': error
                        }
                        
                        self._send_response(200, response)
                    except Exception as e:
                        self._send_response(500, {
                            'success': False,
                            'error': {
                                'name': type(e).__name__,
                                'value': str(e),
                                'traceback': traceback.format_exc()
                            }
                        })
                else:
                    self._send_response(404, {'error': 'Not found'})
        
        if __name__ == '__main__':
            server = HTTPServer(('0.0.0.0', 8000), VMHelperHandler)
            print('VM Helper Service started on port 8000')
            server.serve_forever()
        """)
    
    def _configure_vm(self, image_path: str):
        """Configure the VM using the Firecracker API."""
        import json
        import requests
        
        # Configure machine
        machine_config = {
            "vcpu_count": self.vm_cpu_count,
            "mem_size_mib": self.vm_memory_mb,
            "track_dirty_pages": True
        }
        
        response = requests.put(
            f"http://localhost/machine-config",
            data=json.dumps(machine_config),
            headers={"Content-Type": "application/json"},
            unix_socket_path=self.socket_path
        )
        if response.status_code != 204:
            self.cleanup()
            raise RuntimeError(f"Failed to configure VM: {response.text}")
        
        # Configure boot source
        boot_config = {
            "kernel_image_path": "/boot/vmlinux",
            "boot_args": "console=ttyS0 reboot=k panic=1 pci=off"
        }
        
        response = requests.put(
            f"http://localhost/boot-source",
            data=json.dumps(boot_config),
            headers={"Content-Type": "application/json"},
            unix_socket_path=self.socket_path
        )
        if response.status_code != 204:
            self.cleanup()
            raise RuntimeError(f"Failed to configure boot source: {response.text}")
        
        # Configure root filesystem
        rootfs_config = {
            "drive_id": "rootfs",
            "path_on_host": image_path,
            "is_root_device": True,
            "is_read_only": False
        }
        
        response = requests.put(
            f"http://localhost/drives/rootfs",
            data=json.dumps(rootfs_config),
            headers={"Content-Type": "application/json"},
            unix_socket_path=self.socket_path
        )
        if response.status_code != 204:
            self.cleanup()
            raise RuntimeError(f"Failed to configure root filesystem: {response.text}")
        
        # Configure network
        network_config = {
            "iface_id": "eth0",
            "host_dev_name": "tap0",
            "guest_mac": "AA:FC:00:00:00:01"
        }
        
        response = requests.put(
            f"http://localhost/network-interfaces/eth0",
            data=json.dumps(network_config),
            headers={"Content-Type": "application/json"},
            unix_socket_path=self.socket_path
        )
        if response.status_code != 204:
            self.cleanup()
            raise RuntimeError(f"Failed to configure network: {response.text}")
        
        # Start VM
        response = requests.put(
            f"http://localhost/actions",
            data=json.dumps({"action_type": "InstanceStart"}),
            headers={"Content-Type": "application/json"},
            unix_socket_path=self.socket_path
        )
        if response.status_code != 204:
            self.cleanup()
            raise RuntimeError(f"Failed to start VM: {response.text}")
    
    def _start_helper_service(self):
        """Start the helper service inside the VM."""
        # The helper service should start automatically as a systemd service
        # Wait for it to be ready
        import requests
        import time
        
        for _ in range(60):  # 30 seconds timeout
            try:
                response = requests.get(f"{self.api_endpoint}/ping", timeout=1)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(0.5)
        else:
            self.cleanup()
            raise RuntimeError("VM helper service failed to start")
    
    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        """
        Run code in the Firecracker VM and return the result and execution logs.
        
        Args:
            code (`str`): The Python code to execute.
            return_final_answer (`bool`, optional): Whether to return the final answer. Defaults to False.
            
        Returns:
            `Tuple[Any, str]`: The result of the execution and execution logs.
        """
        import requests
        import json
        import base64
        import pickle
        
        try:
            # Send code to VM
            response = requests.post(
                f"{self.api_endpoint}/execute",
                json={"code": code, "return_final_answer": return_final_answer},
                timeout=60  # Longer timeout for complex operations
            )
            
            if response.status_code != 200:
                raise AgentError(f"Failed to execute code: {response.text}", self.logger)
            
            response_data = response.json()
            
            if not response_data["success"]:
                error = response_data["error"]
                error_message = (
                    f"Execution failed: {error['name']}\n"
                    f"{error['value']}\n{error['traceback']}"
                )
                raise AgentError(error_message, self.logger)
            
            # Get execution logs
            execution_logs = response_data["stdout"]
            if response_data["stderr"]:
                execution_logs += "\n" + response_data["stderr"]
            
            # Get result
            result = None
            if response_data["result"]:
                try:
                    result = pickle.loads(base64.b64decode(response_data["result"]))
                except Exception as e:
                    self.logger.log(f"Failed to unpickle result: {e}", level=LogLevel.ERROR)
            
            return result, execution_logs
            
        except requests.RequestException as e:
            raise AgentError(f"Failed to communicate with VM: {str(e)}", self.logger)
    
    def cleanup(self):
        """Clean up resources used by the VM."""
        if self.firecracker_process and self.firecracker_process.poll() is None:
            try:
                # Try to stop VM gracefully
                import requests
                import json
                
                requests.put(
                    f"http://localhost/actions",
                    data=json.dumps({"action_type": "SendCtrlAltDel"}),
                    headers={"Content-Type": "application/json"},
                    unix_socket_path=self.socket_path,
                    timeout=1
                )
                
                # Wait for process to terminate
                for _ in range(10):  # 5 seconds
                    if self.firecracker_process.poll() is not None:
                        break
                    time.sleep(0.5)
                
                # Force terminate if still running
                if self.firecracker_process.poll() is None:
                    self.firecracker_process.terminate()
                    self.firecracker_process.wait(timeout=5)
            except:
                # If anything fails, force kill
                if self.firecracker_process.poll() is None:
                    self.firecracker_process.kill()
        
        # Clean up temporary directory
        try:
            self.subprocess.run(["rm", "-rf", self.temp_dir], check=False)
        except:
            pass
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


__all__ = ["E2BExecutor", "DockerExecutor", "FirecrackerExecutor"]
