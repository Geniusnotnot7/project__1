
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "pydantic",
#     "uvicorn",
#     "aiofiles",
#     "pytesseract",
#     "numpy",
#     "sentence_transformers",
#     "markdown",
#     "re",
#     "os",
#     "json",
#     "sqlite3",
#     "glob",
#     "subprocess",
#     "datetime",
#     "pathlib",
#     "requests",
#     "scikit-learn",
#     "dateutil",
#     "typing",
#     "logging",
#     "httpx",
#     "pyterract",
#     "faker",
#     "pillow"
# ]
# ///





from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import os
import json
import sqlite3
import glob
import subprocess
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image
import markdown
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from dateutil import parser
from typing import Tuple, List, Dict, Optional
import logging
import httpx
import pytesseract




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DataWorks Task Executor API")

class TaskExecutor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.aiproxy_token = os.environ.get('AIPROXY_TOKEN')
        if not self.aiproxy_token:
            raise ValueError("AIPROXY_TOKEN environment variable is required")

    async def _call_llm(self, prompt: str) -> str:
        """Helper method to call GPT-4-mini through AI Proxy"""
        headers = {
            'Authorization': f'Bearer {self.aiproxy_token}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'gpt-4o-mini',  # Ensure the model parameter is provided
            'messages': [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
            ],
            
            'max_tokens': 150
        }
        url="https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        async with httpx.AsyncClient() as session:
            response = await session.post(url, json=data, headers=headers)
            result =response.json()
            
            if "error" in result:
                raise ValueError(f"AI Proxy Error: {result['error']['message']}")

            # ✅ Extracting the correct key
            return result["choices"][0]["message"]["content"].strip()

            #return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            

    async def _identify_task_type(self, task_description: str) -> str:
        """Use LLM to identify the type of task from the description"""
        prompt = f"""
        Identify the type of task from this description: "{task_description}"
        Possible types:
        - install_uv: Install uv and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py script with this email as argument
        - format_markdown: Format markdown with prettier
        - count_weekday: Count occurrences of a specific weekday
        - sort_contacts: Sort contacts by name
        - recent_logs: Get recent log file contents
        - create_index: Create index of markdown files
        - extract_email: Extract email from text
        - extract_card: Extract credit card number
        - find_similar: Find similar text items
        - sql_query: Run SQL query
        
        Return only the task type.
        """
        
        task_type = (await self._call_llm(prompt)).strip().lower()
        return task_type

    async def _extract_paths(self, task_description: str) -> Tuple[str, str]:
        """Use LLM to extract input and output file paths from task description"""
        prompt = f"""
        Extract input and output file paths from this task: "{task_description}"
        Format: input_path|||output_path
        Example: /data/input.txt|||/data/output.txt
        If multiple input files, use the first one.
        """
        
        paths = (await self._call_llm(prompt)).strip()
        input_path, output_path = paths.split('|||')
        return input_path.strip(), output_path.strip()

    async def _extract_weekday(self, task_description: str) -> int:
        """Use LLM to identify which weekday to count"""
        prompt = f"""
        Which weekday should be counted in this task: "{task_description}"
        Return only the weekday name in lowercase.
        """
        
        weekday = (await self._call_llm(prompt)).strip().lower()
        weekday_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        return weekday_map.get(weekday, 2)  # Default to Wednesday if unclear

    async def execute_task(self, task_description: str) -> Tuple[bool, str]:
        """Main method to parse and execute tasks based on description"""
        try:
            # Safety checks
            file_paths = [
                path for path in re.findall(r'/\S+', task_description) 
                if not path.startswith(('http://', 'https://')) and  # Exclude URLs
                not path.startswith('/data/')  # Exclude valid data paths
            ]
        
            
            
            # Identify task type using LLM
            task_type = await self._identify_task_type(task_description)
            
            # Execute identified task
            task_handlers = {
                "install_uv": self._handle_uv_installation,
                "format_markdown": self._handle_markdown_formatting,
                "count_weekday": self._handle_weekday_counting,
                "sort_contacts": self._handle_contact_sorting,
                "recent_logs": self._handle_recent_logs,
                "create_index": self._handle_markdown_index,
                "extract_email": self._handle_email_extraction,
                "extract_card": self._handle_card_extraction,
                "find_similar": self._handle_similar_comments ,
                "sql_query": self._handle_sql_query
            }
            
            handler = task_handlers.get(task_type)
            if handler:
                return await handler(task_description)
            else:
                return False, f"Unrecognized task type: {task_type}"
                
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            return False, f"Error executing task: {str(e)}"

    async def _handle_uv_installation(self, task_description: str) -> Tuple[bool, str]:
        try:
            # Extract email from task description using LLM
            prompt = f"""Extract the email address from: "{task_description}"
            If no email found, return "default@example.com"."""
            email = (await self._call_llm(prompt)).strip()
            
            # Check if uv is installed
            try:
                subprocess.run(['uv', '--version'], check=True, capture_output=True)
            except:
                subprocess.run(['curl', '-sSf', 'https://static.uv.dev/install.sh', '|', 'sh'], shell=True, check=True)
            
            # Run datagen script
            response = requests.get('https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py')
            with open('datagen.py', 'w') as f:
                f.write(response.text)
            
            subprocess.run(['python', '-m', 'uv', 'run', 'datagen.py', email], check=True)
            return True, f"Successfully installed uv and ran datagen script with email: {email}"
        except Exception as e:
            logger.error(f"Error in uv installation: {str(e)}")
            return False, f"Error in uv installation: {str(e)}"
    async def _handle_markdown_formatting(self, task_description: str) -> Tuple[bool, str]:
        """
        Formats a Markdown file using Prettier.
        """
        try:
            input_path, _ = await self._extract_paths(task_description)

            # Ensure input file exists
            if not os.path.exists(input_path):
                return False, f"Error: File not found - {input_path}"
            if subprocess.run(["which", "npx"], capture_output=True).returncode != 0:
                return False, "Error: npx (Node.js) is not installed."


            # Run Prettier to format Markdown
            subprocess.run(["npx", "prettier", "--write", input_path], check=True)

            return True, f"Successfully formatted Markdown file: {input_path}"
        except FileNotFoundError as e:
            print("Command not found:", e)
        except Exception as e:
            return False, f"Error formatting Markdown: {str(e)}"
    async def _handle_weekday_counting(self, task_description: str) -> Tuple[bool, str]:
        """
        Counts occurrences of a specific weekday in a file.
        """
        try:
            input_path, output_path = await self._extract_paths(task_description)
            weekday_num = await self._extract_weekday(task_description)

            # Read file and count matching weekdays
            with open(input_path, 'r') as f:
                dates = [line.strip() for line in f.readlines()]
        
            count = sum(1 for date in dates if parser.parse(date).weekday() == weekday_num)

            with open(output_path, 'w') as f:
                f.write(str(count))

            return True, f"Successfully counted {count} Wednesdays in {input_path}"
    
        except Exception as e:
            return False, f"Error counting weekdays: {str(e)}"
    async def _handle_contact_sorting(self, task_description: str) -> Tuple[bool, str]:
        """
        Sorts contacts JSON file by last name and first name.
        """
        try:
            input_path, output_path = await self._extract_paths(task_description)

            with open(input_path, 'r') as f:
                contacts = json.load(f)
        
            contacts.sort(key=lambda c: (c.get("last_name", "").lower(), c.get("first_name", "").lower()))

            with open(output_path, 'w') as f:
                json.dump(contacts, f, indent=4)

            return True, f"Successfully sorted contacts in {output_path}"
    
        except Exception as e:
            return False, f"Error sorting contacts: {str(e)}"
    
    async def _handle_recent_logs(self, task_description: str) -> Tuple[bool, str]:
        """
        Reads the first line of the 10 most recent .log files.
        """
        try:
            log_files = sorted(glob.glob("/data/logs/*.log"), key=os.path.getmtime, reverse=True)[:10]

            with open("/data/logs-recent.txt", "w") as f:
                for file in log_files:
                    with open(file, 'r') as log_f:
                        f.write(log_f.readline().strip() + '\n')

            return True, "Successfully extracted recent log entries."
    
        except Exception as e:
            return False, f"Error extracting logs: {str(e)}"

    async def _handle_markdown_index(self, task_description: str) -> Tuple[bool, str]:
        """
        Creates an index of markdown files mapping filename to the first H1.
        """
        try:
            index = {}

            for md_file in glob.glob("/data/docs/*.md"):
                with open(md_file, "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[os.path.basename(md_file)] = line[2:].strip()
                            break

            with open("/data/docs/index.json", "w") as f:
                json.dump(index, f, indent=4)

            return True, "Successfully created Markdown index."
    
        except Exception as e:
            return False, f"Error creating Markdown index: {str(e)}"

    async def _handle_email_extraction(self, task_description: str) -> Tuple[bool, str]:
        """
        Extracts sender's email address from a file.
        """
        try:
            input_path, output_path = await self._extract_paths(task_description)

            with open(input_path, "r") as f:
                email_content = f.read()

            email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", email_content)

            if email:
                with open(output_path, "w") as f:
                    f.write(email.group())

                return True, f"Extracted email: {email.group()}"
            else:
                return False, "No email found."
    
        except Exception as e:
            return False, f"Error extracting email: {str(e)}"

    async def _handle_card_extraction(self, task_description: str) -> Tuple[bool, str]:
        """
        Extracts credit card number from an image.
        """
        try:
            input_path, output_path = await self._extract_paths(task_description)

            # Use OCR to extract text from image
            image = Image.open(input_path)
            text = pytesseract.image_to_string(image)

            # Extract only digits (credit card numbers are numeric)
            card_number = re.sub(r"\D", "", text)  

            with open(output_path, "w") as f:
                f.write(card_number)

            return True, f"Extracted credit card number: {card_number}"
    
        except Exception as e:
            return False, f"Error extracting card number: {str(e)}"

    async def _handle_similar_comments(self, task_description: str) -> Tuple[bool, str]:
        """
        Finds the most similar pair of comments using embeddings.
        """
        try:
            input_path, output_path = await self._extract_paths(task_description)

            with open(input_path, 'r') as f:
                comments = [line.strip() for line in f.readlines()]

            embeddings = self.model.encode(comments)
            similarity_matrix = cosine_similarity(embeddings)

            np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
            most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

            with open(output_path, "w") as f:
                f.write(comments[most_similar_indices[0]] + "\n")
                f.write(comments[most_similar_indices[1]])

            return True, "Successfully found the most similar comments."
    
        except Exception as e:
            return False, f"Error finding similar comments: {str(e)}"


    # ... [Other handler methods remain similar, just add async/await] ...

    async def _handle_sql_query(self, task_description: str) -> Tuple[bool, str]:
        try:
            _, output_path = await self._extract_paths(task_description)
            
            # Use LLM to extract query details
            prompt = f"""Extract the SQL query details from: "{task_description}"
            What type of tickets should be counted? Return only the ticket type."""
            
            ticket_type = (await self._call_llm(prompt)).strip()
            
            conn = sqlite3.connect('/data/ticket-sales.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT SUM(units * price)
                FROM tickets
                WHERE type = ?
            """, (ticket_type,))
            
            total_sales = cursor.fetchone()[0]
            
            with open(output_path, 'w') as f:
                f.write(str(total_sales))
            
            conn.close()
            return True, f"Successfully calculated ticket sales for {ticket_type} tickets"
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            return False, f"Error querying database: {str(e)}"

# Initialize task executor
executor = TaskExecutor()

@app.post("/run")
async def run_task(task: str) -> PlainTextResponse:
    """
    Execute a task based on the provided description.
    
    Args:
        task: Task description in plain English
        
    Returns:
        PlainTextResponse with execution result
    
    Raises:
        HTTPException: If task execution fails
    """
    if not task:
        raise HTTPException(status_code=400, detail="No task provided")
    
    try:
        success, message = await executor.execute_task(task)
    
        if success:
            return PlainTextResponse(message, status_code=200)
        elif "Access denied" in message:
            raise HTTPException(status_code=400, detail=message)
        else:
            raise HTTPException(status_code=500, detail=message)

    except Exception as e:
        logger.error(f"Task execution error: {e}", exc_info=True)  # Logs full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/read")
async def read_file(path: str) -> PlainTextResponse:
    """
    Read contents of a file from the specified path.
    
    Args:
        path: Path to the file to read
        
    Returns:
        PlainTextResponse with file contents
    
    Raises:
        HTTPException: If file cannot be read or access is denied
    """
    if not path:
        raise HTTPException(status_code=400, detail="No path provided")
    
    if '../' in path or not path.startswith('/data/'):
        raise HTTPException(status_code=400, detail="Access denied: Can only read files within /data directory")
    
    try:
        with open(path, 'r') as f:
            content = f.read()
        return PlainTextResponse(content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
