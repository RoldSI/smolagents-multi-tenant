# smolagents-multi-tenant

## Overview
A multi-tenant system for running isolated LLM agents with Firecracker microVMs.

## Features
- Profiles
  - Store APIs (url, key)
  - Store specialized tools with isolated execution environments
  - Firecracker microVM isolation for each profile
- Tool templates for SQL
- Base tools: visual web, text web search, code execution
- LLM switching (capable vs. simple)

## Architecture
- Each profile is stored with its own Firecracker image
- Tools are isolated in separate microVMs for security
- LLMs can be configured and switched easily

## Components
### Multi Interface (`multi.py`)
Provides a Python interface for:
- Managing profiles with tools
- Managing LLM configurations
- Running queries with isolated tool execution

### Firecracker Executor
- Creates microVMs for isolated code execution
- Based on lightweight Debian images
- Each profile gets its own VM with pre-installed tools

## Usage
```python
from multi import Multi

# Initialize the Multi interface
multi = Multi()

# Add a profile with tools
profile_id = multi.add_profile(
    profile_name="My Profile",
    profile_description="A profile with data analysis tools",
    tools=[("tool_name", "tool_description", "tool_code", ["dependency1", "dependency2"])]
)

# Add an LLM configuration
llm_id = multi.add_llm(
    base_url="https://api.openai.com/v1",
    api_key="YOUR_API_KEY",
    model_name="gpt-4",
    llm_name="GPT-4"
)

# Run a query using the profile's tools and LLM
answer = multi.run_query(
    tool_ids=[f"{profile_id}:tool_name"],
    llm_id=llm_id
)
```

## Requirements
- Firecracker installed (`apt-get install firecracker`)
- Python 3.8+
- debootstrap for creating VM images (`apt-get install debootstrap`)
- Root privileges for creating and mounting VM images

## Security
- Each profile runs in an isolated Firecracker microVM
- Tools cannot access the host system or other profiles
- Automatic cleanup of VM resources after use
