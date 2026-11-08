"""Runs tests/run_single_file_subset.py with the OpenAI key blanked INSIDE the process.

The system prefers OpenAI whenever OPENAI_API_KEY is non-empty, and backend/.env holds that key. python-dotenv does not
override a variable that already exists, so an empty value keeps the system on the Google client. The blank value cannot
be set from PowerShell (assigning "" there deletes the variable, and .env would then be read again), hence this wrapper.
The model comes from GOOGLE_GENAI_MODEL, set by launch_gemini.ps1. Arguments are passed on unchanged.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["OPENAI_API_KEY"] = ""
if not os.environ.get("GOOGLE_GENAI_MODEL", "").startswith("gemini-"):
    sys.exit("GOOGLE_GENAI_MODEL is not a gemini model; start this through launch_gemini.ps1")
script = os.path.join(HERE, "..", "tests", "run_single_file_subset.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
