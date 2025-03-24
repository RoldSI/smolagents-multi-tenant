#!/usr/bin/env python3
"""
Example script demonstrating how to use the Multi interface with Firecracker.
"""
import sys
import json
from multi import Multi

def main():
    """Run the example."""
    # Initialize Multi
    multi = Multi()
    
    # Example 1: Add a profile with a simple calculator tool
    print("Creating a profile with a simple calculator tool...")
    
    calculator_name = "Calculator"
    calculator_desc = "A simple calculator tool for basic math operations"
    calculator_code = """
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b

def subtract(a, b):
    \"\"\"Subtract b from a.\"\"\"
    return a - b

def multiply(a, b):
    \"\"\"Multiply two numbers.\"\"\"
    return a * b

def divide(a, b):
    \"\"\"Divide a by b.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
    calculator_deps = []
    
    # Add a profile with the calculator tool
    profile_id = multi.add_profile(
        profile_name="Basic Math Tools",
        profile_description="Profile containing basic math tools",
        tools=[(calculator_name, calculator_desc, calculator_code, calculator_deps)]
    )
    
    print(f"Created profile with ID: {profile_id}")
    
    # Example 2: Add a more complex tool with dependencies
    print("\nCreating a profile with a data processing tool...")
    
    data_tool_name = "DataProcessor"
    data_tool_desc = "A tool for processing and analyzing data"
    data_tool_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_sample_data(n=100):
    \"\"\"Generate sample data for testing.\"\"\"
    data = {
        'x': np.random.normal(0, 1, n),
        'y': np.random.normal(0, 1, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    }
    return pd.DataFrame(data)

def analyze_data(data):
    \"\"\"Perform basic analysis on a dataframe.\"\"\"
    if not isinstance(data, pd.DataFrame):
        data = generate_sample_data()
    
    result = {
        'summary': data.describe().to_dict(),
        'correlation': data.corr().to_dict(),
        'head': data.head().to_dict()
    }
    return result

def plot_data(data, x='x', y='y', color='category'):
    \"\"\"Create a plot from data and return as base64 encoded string.\"\"\"
    if not isinstance(data, pd.DataFrame):
        data = generate_sample_data()
    
    plt.figure(figsize=(10, 6))
    for cat, group in data.groupby(color):
        plt.scatter(group[x], group[y], label=cat)
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y} by {color}')
    plt.legend()
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_str}"
"""
    data_tool_deps = ["pandas", "numpy", "matplotlib"]
    
    # Add a profile with the data processing tool
    data_profile_id = multi.add_profile(
        profile_name="Data Analysis Tools",
        profile_description="Profile containing data analysis tools",
        tools=[(data_tool_name, data_tool_desc, data_tool_code, data_tool_deps)]
    )
    
    print(f"Created profile with ID: {data_profile_id}")
    
    # Example 3: List all profiles
    print("\nListing all profiles:")
    profiles = multi.get_profiles()
    for profile in profiles:
        print(f"  - {profile['profile_id']}: {profile['profile_description']}")
    
    # Example 4: Add an LLM
    print("\nAdding an LLM...")
    
    llm_id = multi.add_llm(
        base_url="https://api.openai.com/v1",
        api_key="YOUR_API_KEY_HERE",
        model_name="gpt-4",
        llm_name="GPT-4"
    )
    
    print(f"Added LLM with ID: {llm_id}")
    
    # Example 5: Retrieve LLM details
    print("\nRetrieving LLM details:")
    llm_details = multi.get_llm(llm_id)
    print(f"  - Name: {llm_details['llm_name']}")
    print(f"  - Model: {llm_details['model_name']}")
    print(f"  - Base URL: {llm_details['base_url']}")
    
    # Example 6: Run a query (Note: this is a simplified example)
    print("\nRun a query (Note: this is just a demonstration):")
    try:
        answer = multi.run_query(tool_ids=[f"{profile_id}:Calculator"], llm_id=llm_id)
        print(f"Query answer: {answer}")
    except Exception as e:
        print(f"Error running query: {e}")
        print("Note: To run an actual query, you need to have Firecracker installed and configured properly.")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main() 