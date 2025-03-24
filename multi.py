import os
import json
import uuid
import sqlite3
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from textwrap import dedent

class Multi:
    def __init__(self, db_path="storage.db"):
        """Initialize the Multi class with a LightSQL database."""
        self.db_path = db_path
        self._setup_database()
    
    def _setup_database(self):
        """Set up the database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create profiles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            tools TEXT NOT NULL
        )
        ''')
        
        # Create llms table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS llms (
            llm_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            api_key TEXT NOT NULL,
            model_name TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_profile(self, profile_name: str, profile_description: str, tools: List[Tuple[str, str, str, List[str]]]) -> str:
        """
        Add a new profile to the system.
        
        Args:
            profile_name: Name of the profile
            profile_description: Description of the profile
            tools: List of tuples containing (tool_name, tool_description, tool_code, tool_dependencies)
            
        Returns:
            profile_id: Unique identifier for the profile
        """
        profile_id = str(uuid.uuid4())
        
        # Store tools as JSON
        tools_json = json.dumps(tools)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO profiles (profile_id, name, description, tools) VALUES (?, ?, ?, ?)",
            (profile_id, profile_name, profile_description, tools_json)
        )
        
        conn.commit()
        conn.close()
        
        # Build and store a firecracker base system for this profile
        self._build_firecracker_image(profile_id, tools)
        
        return profile_id
    
    def _build_firecracker_image(self, profile_id: str, tools: List[Tuple[str, str, str, List[str]]]):
        """
        Build a Firecracker image for the profile with the specified tools.
        
        Args:
            profile_id: ID of the profile
            tools: List of tools to install in the image
        """
        # Create profile directory
        profile_dir = f"profiles/{profile_id}"
        os.makedirs(profile_dir, exist_ok=True)
        
        # Check if default image exists, if not create it
        default_image_path = "profiles/default/rootfs.ext4"
        if not os.path.exists(default_image_path):
            self._create_default_image()
        
        # Copy default image as a base
        profile_image_path = f"{profile_dir}/rootfs.ext4"
        shutil.copy(default_image_path, profile_image_path)
        
        # Mount the image
        with tempfile.TemporaryDirectory() as mount_dir:
            try:
                # Mount the image
                subprocess.run(
                    ["mount", "-o", "loop", profile_image_path, mount_dir],
                    check=True
                )
                
                # Install tool dependencies
                all_dependencies = []
                for _, _, _, dependencies in tools:
                    all_dependencies.extend(dependencies)
                
                if all_dependencies:
                    try:
                        # Install pip dependencies
                        pip_cmd = f"chroot {mount_dir} pip3 install {' '.join(all_dependencies)}"
                        subprocess.run(pip_cmd, shell=True, check=True)
                    except subprocess.CalledProcessError:
                        print(f"Warning: Some dependencies could not be installed: {all_dependencies}")
                
                # Save tools to the image
                tools_dir = f"{mount_dir}/opt/tools"
                os.makedirs(tools_dir, exist_ok=True)
                
                # Write each tool to a file
                for tool_name, tool_description, tool_code, _ in tools:
                    tool_filename = f"{tools_dir}/{tool_name.replace(' ', '_').lower()}.py"
                    with open(tool_filename, "w") as f:
                        f.write(f"# {tool_description}\n\n")
                        f.write(tool_code)
                
                # Create tool loader script
                loader_path = f"{mount_dir}/opt/load_tools.py"
                with open(loader_path, "w") as f:
                    f.write(dedent("""
                    import os
                    import sys
                    import importlib.util
                    
                    def load_tools():
                        tools = {}
                        tools_dir = "/opt/tools"
                        
                        for filename in os.listdir(tools_dir):
                            if filename.endswith(".py"):
                                tool_name = os.path.splitext(filename)[0]
                                tool_path = os.path.join(tools_dir, filename)
                                
                                spec = importlib.util.spec_from_file_location(tool_name, tool_path)
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                
                                # Add module to global namespace
                                sys.modules[tool_name] = module
                                globals()[tool_name] = module
                                
                                # Add to tools dict
                                tools[tool_name] = module
                        
                        return tools
                    
                    if __name__ == "__main__":
                        tools = load_tools()
                        print(f"Loaded {len(tools)} tools: {', '.join(tools.keys())}")
                    """))
                
            finally:
                # Unmount the image
                subprocess.run(["umount", mount_dir], check=True)
    
    def _create_default_image(self):
        """Create default Firecracker image with Python and common packages."""
        # Create profiles directory if it doesn't exist
        os.makedirs("profiles/default", exist_ok=True)
        
        # Create a basic ext4 image with Python and common packages
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create rootfs directory
            rootfs_dir = f"{temp_dir}/rootfs"
            os.makedirs(rootfs_dir)
            
            # Use debootstrap to create a minimal Debian system
            try:
                subprocess.run(
                    ["debootstrap", "--variant=minbase", "bullseye", rootfs_dir],
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(
                    "Failed to create base image. Make sure debootstrap is installed: "
                    "sudo apt-get install debootstrap"
                )
            
            # Install Python and common packages
            subprocess.run(
                ["chroot", rootfs_dir, "apt-get", "update"],
                check=True,
            )
            subprocess.run(
                [
                    "chroot", rootfs_dir, "apt-get", "install", "-y",
                    "python3", "python3-pip", "python3-venv", "ca-certificates",
                    "curl", "wget", "git", "vim", "nano"
                ],
                check=True,
            )
            
            # Install common Python packages
            subprocess.run(
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
            subprocess.run(["chmod", "+x", helper_service_path], check=True)
            
            # Create systemd service for helper
            service_dir = f"{rootfs_dir}/etc/systemd/system"
            os.makedirs(service_dir, exist_ok=True)
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
            subprocess.run(
                ["chroot", rootfs_dir, "systemctl", "enable", "vm-helper.service"],
                check=True,
            )
            
            # Create ext4 image
            # Use a more reasonable size based on actual needs
            image_size_mb = 2048  # 2GB image - sufficient for base system and common packages
            image_path = "profiles/default/rootfs.ext4"
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            subprocess.run(
                ["dd", "if=/dev/zero", f"of={image_path}", "bs=1M", f"count={image_size_mb}"],
                check=True,
            )
            subprocess.run(
                ["mkfs.ext4", image_path],
                check=True,
            )
            
            # Mount and copy files
            with tempfile.TemporaryDirectory() as mount_dir:
                subprocess.run(
                    ["mount", image_path, mount_dir],
                    check=True,
                )
                try:
                    subprocess.run(
                        ["cp", "-a", f"{rootfs_dir}/.", mount_dir],
                        check=True,
                    )
                finally:
                    subprocess.run(
                        ["umount", mount_dir],
                        check=True,
                    )
    
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
            
            def do_GET(self):
                if self.path == '/ping':
                    self._send_response(200, {'status': 'ok'})
                else:
                    self._send_response(404, {'error': 'Not found'})
            
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
    
    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """
        Get information about a specific profile.
        
        Args:
            profile_id: The ID of the profile to retrieve
            
        Returns:
            Dictionary containing all profile information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM profiles WHERE profile_id = ?", (profile_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            profile_id, name, description, tools_json = result
            return {
                "profile_id": profile_id,
                "name": name,
                "description": description,
                "tools": json.loads(tools_json)
            }
        else:
            raise ValueError(f"Profile with ID {profile_id} not found")
    
    def get_profiles(self) -> List[Dict[str, str]]:
        """
        Get a list of all profiles.
        
        Returns:
            List of dictionaries containing profile_id and profile_description
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT profile_id, description FROM profiles")
        results = cursor.fetchall()
        
        conn.close()
        
        return [{"profile_id": pid, "profile_description": desc} for pid, desc in results]
    
    def add_llm(self, base_url: str, api_key: str, model_name: str, llm_name: str) -> str:
        """
        Add a new LLM to the system.
        
        Args:
            base_url: Base URL for the LLM API
            api_key: API key for authentication
            model_name: Name of the model to use
            llm_name: Friendly name for this LLM configuration
            
        Returns:
            llm_id: Unique identifier for the LLM
        """
        llm_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO llms (llm_id, name, base_url, api_key, model_name) VALUES (?, ?, ?, ?, ?)",
            (llm_id, llm_name, base_url, api_key, model_name)
        )
        
        conn.commit()
        conn.close()
        
        return llm_id
    
    def get_llm(self, llm_id: str) -> Dict[str, str]:
        """
        Get information about a specific LLM.
        
        Args:
            llm_id: The ID of the LLM to retrieve
            
        Returns:
            Dictionary containing llm_name, base_url, and model_name
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, base_url, model_name FROM llms WHERE llm_id = ?", (llm_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            llm_name, base_url, model_name = result
            return {
                "llm_name": llm_name,
                "base_url": base_url,
                "model_name": model_name
            }
        else:
            raise ValueError(f"LLM with ID {llm_id} not found")
    
    def get_llms(self) -> List[Dict[str, str]]:
        """
        Get a list of all LLMs.
        
        Returns:
            List of dictionaries containing llm_id and llm_name
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT llm_id, name FROM llms")
        results = cursor.fetchall()
        
        conn.close()
        
        return [{"llm_id": lid, "llm_name": name} for lid, name in results]
    
    def run_query(self, tool_ids: List[str], llm_id: str) -> str:
        """
        Run a query using the specified tools and LLM.
        
        Args:
            tool_ids: List of tool IDs to use for the query
            llm_id: ID of the LLM to use for the query
            
        Returns:
            answer: The response from the LLM
        """
        # Get LLM details
        llm_details = self.get_llm(llm_id)
        
        # Create FirecrackerExecutor to run the code
        try:
            from src.remote_executors import FirecrackerExecutor
            from src.monitoring import Logger, LogLevel
        except ImportError:
            raise ImportError("Could not import FirecrackerExecutor. Make sure smolagents is installed.")
        
        # Find the profile IDs for the tools
        tools_info = []
        profile_ids = set()
        
        for tool_id in tool_ids:
            # Extract profile_id from tool_id (assuming format: profile_id:tool_name)
            if ":" in tool_id:
                profile_id, tool_name = tool_id.split(":", 1)
                profile_ids.add(profile_id)
                tools_info.append((profile_id, tool_name))
            else:
                # If no profile ID specified, search all profiles
                profiles = self.get_profiles()
                for profile in profiles:
                    try:
                        profile_data = self.get_profile(profile["profile_id"])
                        for tool in json.loads(profile_data["tools"]):
                            if tool[0] == tool_id:
                                profile_ids.add(profile["profile_id"])
                                tools_info.append((profile["profile_id"], tool_id))
                                break
                    except:
                        continue
        
        if not profile_ids:
            raise ValueError("No valid profiles found for the specified tools")
        
        # For simplicity, use the first profile's image if multiple are found
        profile_id = list(profile_ids)[0]
        
        # Setup logger
        logger = Logger()
        
        # Create executor with the specified profile
        executor = FirecrackerExecutor(
            additional_imports=["requests", "pandas", "numpy"],
            logger=logger,
            profile_id=profile_id
        )
        
        try:
            # Load tools
            executor.run_code_raise_errors("import sys\nsys.path.append('/opt')\nfrom load_tools import load_tools\ntools = load_tools()")
            
            # This is a placeholder for a more complete implementation
            # A full implementation would:
            # 1. Use the LLM API to generate code based on the tools
            # 2. Execute that code in the Firecracker VM
            # 3. Process the results
            
            logger.log("Firecracker executor is working properly with the specified tools", level=LogLevel.INFO)
            return "This is a placeholder response. The query execution is not yet fully implemented."
        finally:
            executor.cleanup()
