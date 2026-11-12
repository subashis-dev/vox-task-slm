"""
Production-grade Local Web Server for the Voice-to-Do SLM.
Multi-threaded socket server with non-blocking concurrency,
persistent task storage, and GPU acceleration.
Zero external pip dependencies (built on standard library http.server + PyTorch).
"""

import os
import re
import json
import time
import mimetypes
from urllib.parse import urlparse
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from model import VoiceTaskSLM
from tokenizer import VoiceTaskTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEB_DIR = os.path.join(BASE_DIR, "web")
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoint_best.pt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Multi-Threaded HTTP Server for Concurrent Non-Blocking Requests
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = VoiceTaskTokenizer()
model = None

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading custom SLM weights from: {CHECKPOINT_PATH} on {device}...")
    cp = torch.load(CHECKPOINT_PATH, map_location=device)
    cfg = cp["config"]
    model = VoiceTaskSLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_seq_len=cfg["max_seq_len"],
        dropout=0.0
    ).to(device)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    print(f"SLM Model ready! Parameters: {model.count_parameters():,} on {device}")
else:
    print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Please train first.")


def normalize_voice_prompt(text: str) -> str:
    """Normalizes conversational preambles and speech recognition quirks before SLM tokenization."""
    s = text.strip()
    # Correct common speech recognition mis-transcriptions
    s = re.sub(r'\bwirte\b', 'write', s, flags=re.IGNORECASE)
    s = re.sub(r'\btommorrow\b', 'tomorrow', s, flags=re.IGNORECASE)
    s = re.sub(r'\batomrow\b', 'tomorrow', s, flags=re.IGNORECASE)
    s = re.sub(r'\basop\b', 'a sop', s, flags=re.IGNORECASE)

    wrapper_patterns = [
        r'^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?write\s+a\s+(?:to-?do|to\s+do)\s+list\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?make\s+a\s+(?:to-?do|to\s+do)\s+list\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?create\s+a\s+(?:to-?do|to\s+do)\s+list\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?write\s+(?:down\s+)?a\s+(?:to-?do|to\s+do)\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?better\s+to\s+do\s+this\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?better\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:it(?:\x27s|\s+is)\s+)?better\s+to\s+',
        r'^(?:please\s+|hey\s+)?(?:i\x27d\s+better|we\x27d\s+better)\s+',
    ]
    for pat in wrapper_patterns:
        m = re.match(pat, s, flags=re.IGNORECASE)
        if m:
            rest = s[m.end():].strip()
            has_time = bool(re.search(r'\b(at\s+\d+|tomorrow|tonight|today|am|pm|in\s+\d+)\b', rest, flags=re.IGNORECASE))
            if has_time:
                return f"remind me to {rest}"
            else:
                return f"add a todo to {rest}"
    return s


def clean_display_title(raw_title: str) -> str:
    """Strips conversational leading fillers, trailing prepositions, and formats acronyms."""
    s = raw_title.strip()
    # Strip trailing prepositions
    s = re.sub(r'\s+(?:at|on|for|in|by|to)$', '', s, flags=re.IGNORECASE)

    leading_fillers = [
        r'^(?:better\s+to\s+do\s+this\s+to\s+)',
        r'^(?:better\s+to\s+do\s+this\s+)',
        r'^(?:better\s+to\s+)',
        r'^(?:do\s+this\s+to\s+write\s+)',
        r'^(?:do\s+this\s+to\s+)',
        r'^(?:do\s+this\s+)',
        r'^(?:list\s+to\s+write\s+)',
        r'^(?:wrist\s+to\s+write\s+)',
        r'^(?:make\s+sure\s+to\s+)',
        r'^(?:remember\s+to\s+)',
        r'^(?:need\s+to\s+)',
        r'^(?:have\s+to\s+)',
        r'^(?:got\s+to\s+)',
        r'^(?:i\s+need\s+to\s+)',
        r'^(?:i\s+have\s+to\s+)',
        r'^(?:try\s+to\s+)',
        r'^(?:about\s+)',
        r'^(?:to\s+)',
    ]
    for pattern in leading_fillers:
        s = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()

    s = re.sub(r'\basop\b', 'a SOP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bsop\b', 'SOP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bphd\b', 'PhD', s, flags=re.IGNORECASE)
    s = re.sub(r'\bkent\b', 'Kent', s, flags=re.IGNORECASE)

    if s:
        s = s[0].upper() + s[1:]
    return s if s else "Untitled Task"


def clean_reply_text(reply_str: str, clean_task: str, due_time: str = "") -> str:
    """Ensures spoken assistant reply contains clean task title and accurate time."""
    if not reply_str:
        if due_time:
            return f"I've set a reminder for {clean_task} {due_time}."
        return f"Added {clean_task} to your list."

    s = reply_str
    s = re.sub(r'\basop\b', 'a SOP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bsop\b', 'SOP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bphd\b', 'PhD', s, flags=re.IGNORECASE)
    s = re.sub(r'\bkent\b', 'Kent', s, flags=re.IGNORECASE)

    # If the reply mentions the task, ensure it uses clean_task
    if clean_task and ("reminder for" in s.lower() or "added" in s.lower()):
        if due_time:
            return f"I've set a reminder for {clean_task} {due_time}."
        else:
            return f"I've saved {clean_task} to your list."
    return s.strip()


def predict_action(voice_text: str):
    """Feeds voice prompt into the custom SLM and generates structured JSON action."""
    if model is None:
        return None, "Model weights not loaded", 0.0

    normalized_text = normalize_voice_prompt(voice_text)
    voice_ids = tokenizer.encode(normalized_text.lower().strip())
    # Format: <bos> {voice} <sep>
    input_ids = [tokenizer.bos_id] + voice_ids + [tokenizer.sep_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_tensor,
            max_new_tokens=150,
            temperature=0.1,  # Deterministic decoding for structured JSON
            eos_id=tokenizer.eos_id
        )
    inference_ms = round((time.time() - start_time) * 1000, 2)

    prompt_len = len(input_ids)
    output_tokens = generated_ids[0][prompt_len:].tolist()
    raw_output = tokenizer.decode(output_tokens).strip()

    try:
        parsed_json = json.loads(raw_output)
        if isinstance(parsed_json, dict):
            if "task" in parsed_json:
                parsed_json["task"] = clean_display_title(parsed_json["task"])
            due_time = parsed_json.get("time", "")
            if due_time and due_time.lower() != "unspecified":
                parsed_json["action"] = "REMIND"
            parsed_json["reply"] = clean_reply_text(
                parsed_json.get("reply", ""),
                parsed_json.get("task", ""),
                due_time if due_time and due_time.lower() != "unspecified" else ""
            )
        return parsed_json, raw_output, inference_ms
    except json.JSONDecodeError:
        return None, raw_output, inference_ms


# ---------------------------------------------------------------------------
# Task Storage Management
# ---------------------------------------------------------------------------
def load_tasks():
    if not os.path.exists(TASKS_FILE):
        initial = {
            "tasks": [
                {
                    "id": "t-1",
                    "type": "REMIND",
                    "title": "Welcome to your Voice SLM Task Manager",
                    "time": "Today",
                    "category": "general",
                    "completed": False,
                    "created_at": time.strftime("%Y-%m-%d %H:%M")
                }
            ]
        }
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2)
        return initial

    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tasks": []}


def save_tasks(data):
    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving tasks: {e}")


def apply_action_to_store(action_dict):
    """Executes the action predicted by the custom SLM on the persistent tasks store."""
    data = load_tasks()
    tasks = data.get("tasks", [])
    act = action_dict.get("action")
    spoken_reply = action_dict.get("reply", "")

    due_time = action_dict.get("time", "")
    has_valid_time = bool(due_time and due_time.lower() != "unspecified" and due_time.strip())
    raw_task = action_dict.get("task", "Untitled Task")
    title = clean_display_title(raw_task)

    # Any task with an explicit time deadline is saved as a scheduled REMINDER
    if act == "REMIND" or has_valid_time:
        new_item = {
            "id": f"t-{int(time.time() * 1000)}",
            "type": "REMIND",
            "title": title,
            "time": due_time if has_valid_time else "No time specified",
            "category": action_dict.get("category", "reminder"),
            "completed": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M")
        }
        tasks.insert(0, new_item)
        if not spoken_reply or "Do this to write" in spoken_reply:
            if has_valid_time:
                spoken_reply = f"Scheduled reminder: {title} for {due_time}."
            else:
                spoken_reply = f"Scheduled reminder: {title}."

    elif act == "TODO":
        cat = action_dict.get("category", "general")
        new_item = {
            "id": f"t-{int(time.time() * 1000)}",
            "type": "TODO",
            "title": title,
            "time": "",
            "category": cat,
            "completed": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M")
        }
        tasks.insert(0, new_item)
        if not spoken_reply:
            spoken_reply = f"Added {title} to your {cat} list."

    elif act == "DONE":
        target = action_dict.get("task", "").lower()
        matched = False
        for t in tasks:
            if target in t["title"].lower() or t["title"].lower() in target:
                t["completed"] = True
                matched = True
                break
        if not spoken_reply:
            spoken_reply = f"Marked {target} as completed." if matched else f"Could not find a task matching '{target}'."

    elif act == "DELETE":
        target = action_dict.get("task", "").lower()
        initial_len = len(tasks)
        tasks = [t for t in tasks if target not in t["title"].lower() and t["title"].lower() not in target]
        if not spoken_reply:
            spoken_reply = f"Removed {target} from your list." if len(tasks) < initial_len else f"No task found matching '{target}'."

    elif act == "LIST":
        if not spoken_reply:
            pending_count = sum(1 for t in tasks if not t.get("completed"))
            spoken_reply = f"You currently have {pending_count} pending items on your list."

    elif act == "GREET":
        if not spoken_reply:
            spoken_reply = "Hello! I'm VoxTask, your private voice assistant. How can I help you today?"

    data["tasks"] = tasks
    save_tasks(data)
    return data, spoken_reply


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class SLMRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        try:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_OPTIONS(self):
        try:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. API: Get Tasks
        if path == "/api/tasks":
            return self._send_json(200, load_tasks())

        # 2. API: Health Check
        if path == "/api/health":
            return self._send_json(200, {
                "status": "healthy",
                "model_loaded": model is not None,
                "parameters": model.count_parameters() if model else 0,
                "device": str(device)
            })

        # 3. Static Web Frontend Files
        if path == "/" or path == "/index.html":
            target = os.path.join(WEB_DIR, "index.html")
        else:
            rel = path.lstrip("/")
            target = os.path.join(WEB_DIR, rel)

        if os.path.exists(target) and not os.path.isdir(target):
            mime_type, _ = mimetypes.guess_type(target)
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(target, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "Invalid JSON body"})

        # 1. API: Parse Voice Command via Custom SLM
        if path == "/api/parse":
            text = body.get("text", "").strip()
            if not text:
                return self._send_json(400, {"error": "No voice command text provided"})

            action, raw_str, ms = predict_action(text)
            if not action:
                return self._send_json(422, {
                    "error": "Failed to parse command into valid action",
                    "raw_output": raw_str,
                    "inference_ms": ms
                })

            updated_store, reply = apply_action_to_store(action)

            return self._send_json(200, {
                "action": action,
                "raw_output": raw_str,
                "inference_ms": ms,
                "spoken_reply": reply,
                "tasks": updated_store["tasks"]
            })

        # 2. API: Toggle Task Completion
        if path == "/api/tasks/toggle":
            task_id = body.get("id")
            data = load_tasks()
            for t in data.get("tasks", []):
                if t["id"] == task_id:
                    t["completed"] = not t.get("completed", False)
                    break
            save_tasks(data)
            return self._send_json(200, data)

        # 3. API: Delete Task
        if path == "/api/tasks/delete":
            task_id = body.get("id")
            data = load_tasks()
            data["tasks"] = [t for t in data.get("tasks", []) if t["id"] != task_id]
            save_tasks(data)
            return self._send_json(200, data)

        self._send_json(404, {"error": "Endpoint Not Found"})


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    server = ThreadedHTTPServer((host, port), SLMRequestHandler)
    print("")
    print("=" * 68)
    print("  🎙️ VOICE-TO-DO SMALL LANGUAGE MODEL (SLM) SERVER IS RUNNING!")
    print(f"  🌐 Web Dashboard: http://localhost:{port} (or http://127.0.0.1:{port})")
    print(f"  🧠 Model Device:  {device} (~4.8M parameters)")
    print(f"  💾 Local Storage: data/tasks.json")
    print("=" * 68)
    print("")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    port_env = int(os.environ.get("PORT", 8000))
    host_env = os.environ.get("HOST", "0.0.0.0")
    run_server(port=port_env, host=host_env)
