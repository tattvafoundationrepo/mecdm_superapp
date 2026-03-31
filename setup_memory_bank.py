"""
One-time setup script to create a Vertex AI Agent Engine instance for Memory Bank.

Usage:
    uv run python setup_memory_bank.py

Prerequisites:
    - GOOGLE_CLOUD_PROJECT must be set in .env or environment
    - gcloud auth application-default login must have been run
    - Vertex AI API must be enabled on the project

After running, add the printed AGENT_ENGINE_ID to your .env file as:
    MEMORY_BANK_AGENT_ENGINE_ID=<id>
"""

import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "superapp-488009")
LOCATION = os.environ.get("MEMORY_BANK_LOCATION", "asia-south1")


def main():
    import vertexai

    print(f"Creating Agent Engine instance in {PROJECT_ID}/{LOCATION}...")
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

    agent_engine = client.agent_engines.create()

    resource_name = agent_engine.api_resource.name
    agent_engine_id = resource_name.split("/")[-1]

    print()
    print("=" * 60)
    print("Agent Engine created successfully!")
    print(f"  Resource name: {resource_name}")
    print(f"  Agent Engine ID: {agent_engine_id}")
    print()
    print("Add this to your .env file:")
    print(f"  MEMORY_BANK_AGENT_ENGINE_ID={agent_engine_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
