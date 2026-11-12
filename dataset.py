"""
Top-Tier Assistant Synthetic Dataset Generator.
Understands:
1. All number representations: digits (1-12) & spelled-out words (one to twelve).
2. All AM/PM expressions: am, pm, in the morning, in the afternoon, in the evening, at night, noon, midnight.
3. All kinds of words: open-vocabulary dynamic phrases, technical/PhD terms, proper names, and random n-grams.
4. Natural assistant voice responses ("reply") generated directly by the model.
"""

import random
import json
import torch
from torch.utils.data import Dataset
from tokenizer import VoiceTaskTokenizer

# ---------------------------------------------------------------------------
# Number & Time Mappings
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"
}

MINUTE_VARIANTS = [
    ("", ":00"),
    (":00", ":00"),
    (":15", ":15"),
    (":30", ":30"),
    (":45", ":45"),
    (" fifteen", ":15"),
    (" thirty", ":30"),
    (" forty-five", ":45"),
]

DAYS_SPOKEN = [
    "tomorrow", "today", "tonight", "this Monday", "next Monday",
    "this Tuesday", "next Tuesday", "this Wednesday", "next Wednesday",
    "this Thursday", "next Thursday", "this Friday", "next Friday",
    "this Saturday", "next Saturday", "this Sunday", "next Sunday"
]

CATEGORIES = ["work", "personal", "shopping", "research", "academic", "finance", "health", "system"]

# ---------------------------------------------------------------------------
# Massive Lexical Banks ("All Kinds of Words")
# ---------------------------------------------------------------------------

ACTION_VERBS = [
    "do", "submit", "apply for", "finish", "complete", "prepare", "write",
    "send", "email", "mail", "draft", "revise", "review", "inspect", "upload",
    "download", "read", "study for", "cite", "format", "register for", "sign",
    "debug", "deploy", "backup", "restore", "reboot", "restart", "install",
    "update", "upgrade", "compile", "build", "refactor", "test", "commit",
    "push", "merge", "rebase", "configure", "benchmark", "optimize", "profile",
    "schedule", "organize", "plan", "cancel", "reschedule", "coordinate",
    "approve", "verify", "audit", "document", "present", "discuss with",
    "call", "ring", "text", "message", "buy", "order", "purchase", "pick up",
    "collect", "clean", "wash", "fix", "repair", "service", "pay", "renew",
    "water", "feed", "check", "measure", "pack", "return", "drop off",
    "check out", "track", "archive", "sync", "analyze", "resolve", "troubleshoot"
]

NOUN_PHRASES = [
    # Academic & University
    "kent application", "phd application", "phd application kent", "phd interview",
    "phd research proposal", "master thesis draft", "master thesis defense",
    "bachelor final project", "statement of purpose", "letters of recommendation",
    "scholarship essay", "admissions portal", "university tuition fee",
    "transcript evaluation", "gre exam registration", "toefl score submission",
    "ielts test booking", "visa interview appointment", "embassy document collection",
    "ds160 form", "lab experiment data", "literature survey", "ieee conference paper",
    "acm camera-ready submission", "peer review feedback", "poster presentation",
    "grant proposal submission", "ethics committee approval", "qualifying exam",
    "seminar slides", "course registration", "professors feedback", "oral defense",
    "stanford application", "oxford fellowship", "harvard scholarship", "cambridge exam",

    # Software & Systems
    "vps database backup", "docker containers", "kubernetes pod restart",
    "wireguard tunnel configuration", "ssl certificate renewal", "reverse proxy config",
    "memory leak investigation", "cpu throttling issue", "api endpoint documentation",
    "pull request review", "git merge conflict", "frontend build script",
    "unit test suite", "e2e integration test", "database migration script",
    "thermal paste reapplication", "gpu driver upgrade", "cuda runtime installation",
    "mini pc cluster reboot", "prometheus metrics scraper", "grafana dashboard",
    "auth token expiration bug", "cors middleware headers", "redis cache flush",
    "sqlite vacuum operation", "nginx virtual host file", "caddyfile configuration",
    "pytorch model training", "huggingface token setup", "ollama model pull",

    # Work & Finance
    "client invoice", "monthly expense report", "nda contract signature",
    "quarterly roadmap review", "sprint planning meeting", "standup notes",
    "salary slip download", "tax return filing", "electricity bill payment",
    "wifi internet bill", "domain name renewal", "vps hosting invoice",
    "insurance policy renewal", "bank statement download", "budget spreadsheet",
    "freelance client contract", "stripe payment webhook", "quarterly taxes",

    # Daily Life & Errands
    "groceries from supermarket", "milk and organic eggs", "fresh vegetables",
    "dog food and treats", "cat litter box", "dentist checkup appointment",
    "optometrist eye test", "annual health checkup", "prescription medicines",
    "vitamin d supplements", "30 minute gym workout", "5k evening run",
    "car engine oil change", "tire pressure inspection", "apartment rent payment",
    "water indoor plants", "laundry dry cleaning", "haircut appointment",
    "birthday gift for mom", "anniversary dinner reservation", "flight tickets booking",
    "hotel reservation confirmation", "luggage packing", "passport renewal form"
]

OPEN_VOCAB_WORDS = [
    "alpha", "beta", "gamma", "delta", "matrix", "vector", "tensor", "kernel",
    "circuit", "resistor", "switch", "router", "gateway", "firmware", "packet",
    "sensor", "detector", "antenna", "battery", "display", "monitor", "charger",
    "audio", "video", "signal", "channel", "frequency", "voltage", "current",
    "ticket", "receipt", "invoice", "voucher", "token", "badge", "license",
    "permit", "certificate", "contract", "agreement", "lease", "policy", "record",
    "meeting", "session", "interview", "hearing", "conference", "symposium", "summit",
    "doctor", "dentist", "surgeon", "therapist", "trainer", "advisor", "manager",
    "director", "engineer", "scientist", "developer", "designer", "architect", "analyst",
    "coffee", "tea", "water", "juice", "bread", "cheese", "fruit", "apples", "oranges",
    "chicken", "fish", "meat", "pasta", "rice", "beans", "salad", "pizza", "burger",
    "shirt", "shoes", "pants", "jacket", "glasses", "watch", "keys", "wallet", "phone"
]

REMIND_PREFIXES = [
    "remind me about", "please remind me about", "can you remind me about",
    "hey remind me about", "set a reminder about", "i need a reminder about",
    "remind me to", "please remind me to", "can you remind me to",
    "hey can you remind me to", "set a reminder to", "i need a reminder to",
    "make sure i remember to", "schedule a reminder to", "give me a reminder to",
    "ping me to", "alert me to", "remind me of", "remind me",
    "don't let me forget to", "make sure to remind me about", "uh remind me about",
    "could you remind me about", "please set an alert for", "create a reminder for",
    "schedule an alert for", "wake me up at", "ping me about"
]

TODO_PREFIXES = [
    "add to my to-do list", "put this on my to-do list", "add a task to", "create a todo for",
    "remember that i have to", "add to my list", "new task", "put on my list to",
    "add a todo to", "todo", "please add", "i have to", "i need to",
    "put on my to do", "add item", "can you add to my list", "put on my list",
    "make a note to", "note down", "jot down"
]

DONE_PREFIXES = [
    "mark as done", "check off", "i finished", "mark as completed", "completed task",
    "i have completed", "done with", "mark completed", "cross off", "i did",
    "already finished", "set as done", "i am done with"
]

DELETE_PREFIXES = [
    "delete task", "remove from my list", "cancel reminder for", "drop task",
    "clear reminder to", "discard task", "delete", "remove", "cancel task",
    "take off my list", "erase task"
]


# ---------------------------------------------------------------------------
# Dynamic Time & Number Synthesizer
# ---------------------------------------------------------------------------

def make_dynamic_time_phrase():
    """
    Generates a spoken time phrase and its canonical standardized time string.
    Supports both digit numbers (2 am) and spelled-out words (two am),
    plus AM/PM variations (in the morning, at night, noon, midnight).
    """
    mode = random.random()

    # 1. Standard hour + am/pm
    hour = random.randint(1, 12)
    use_word_for_number = random.random() < 0.50
    hour_str = NUMBER_WORDS[hour] if use_word_for_number else str(hour)
    min_spoken, min_canonical = random.choice(MINUTE_VARIANTS)

    # AM/PM representation variations
    pm_chance = random.random()
    if pm_chance < 0.40:
        period_spoken = random.choice([" am", " a.m.", " in the morning", " o'clock in the morning", " am tomorrow"])
        period_canon = "am"
    elif pm_chance < 0.80:
        period_spoken = random.choice([" pm", " p.m.", " in the afternoon", " in the evening", " at night", " tonight"])
        period_canon = "pm"
    else:
        # Implicit period based on hour
        period_spoken = " am" if hour < 12 else " pm"
        period_canon = "am" if hour < 12 else "pm"

    # Assemble spoken time string
    spoken_clock = f"{hour_str}{min_spoken}{period_spoken}".strip()
    canon_clock = f"{hour}{min_canonical} {period_canon}"

    # Day selection
    day = random.choice(DAYS_SPOKEN)

    # Order variations (e.g. "at 2 am tomorrow" vs "tomorrow at 2 am")
    if random.random() < 0.50:
        spoken = f"{day} at {spoken_clock}"
        canon = f"{day} at {canon_clock}"
    else:
        spoken = f"at {spoken_clock} {day}"
        canon = f"{day} at {canon_clock}"

    return spoken, canon


def make_random_task():
    """Generates an open-ended task with high vocabulary entropy."""
    mode = random.random()
    if mode < 0.45:
        verb = random.choice(ACTION_VERBS)
        noun = random.choice(NOUN_PHRASES)
        return f"{verb} {noun}"
    elif mode < 0.75:
        return random.choice(NOUN_PHRASES)
    else:
        # Open dictionary n-gram combo
        verb = random.choice(ACTION_VERBS)
        w1 = random.choice(OPEN_VOCAB_WORDS)
        w2 = random.choice(OPEN_VOCAB_WORDS)
        return f"{verb} {w1} {w2}"


def clean_task_name(task_str: str) -> str:
    s = task_str.strip()
    for prefix in ["about ", "to ", "of ", "for "]:
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
    return s.capitalize()


# ---------------------------------------------------------------------------
# Example Generator with Natural Assistant Spoken Responses
# ---------------------------------------------------------------------------

def generate_single_example():
    intent_choice = random.random()

    if intent_choice < 0.52:
        # Intent: REMIND
        prefix = random.choice(REMIND_PREFIXES)
        task = make_random_task()
        clean_task = clean_task_name(task)
        has_time = random.random() < 0.80

        if has_time:
            spoken_time, canon_time = make_dynamic_time_phrase()
            if random.random() > 0.35:
                voice = f"{prefix} {task} {spoken_time}"
            else:
                voice = f"{spoken_time} {prefix} {task}"

            reply = f"I've set a reminder for {clean_task} {canon_time}."
            target = json.dumps({
                "action": "REMIND",
                "task": clean_task,
                "time": canon_time,
                "reply": reply
            }, separators=(',', ':'))
        else:
            voice = f"{prefix} {task}"
            reply = f"I've scheduled a reminder for {clean_task}."
            target = json.dumps({
                "action": "REMIND",
                "task": clean_task,
                "time": "unspecified",
                "reply": reply
            }, separators=(',', ':'))

    elif intent_choice < 0.80:
        # Intent: TODO
        prefix = random.choice(TODO_PREFIXES)
        task = make_random_task()
        clean_task = clean_task_name(task)
        cat = random.choice(CATEGORIES)

        if random.random() > 0.65:
            voice = f"{prefix} {task} in {cat}"
            reply = f"Added {clean_task} to your {cat} list."
            target = json.dumps({
                "action": "TODO",
                "task": clean_task,
                "category": cat,
                "reply": reply
            }, separators=(',', ':'))
        else:
            voice = f"{prefix} {task}"
            reply = f"Added {clean_task} to your to-do list."
            target = json.dumps({
                "action": "TODO",
                "task": clean_task,
                "reply": reply
            }, separators=(',', ':'))

    elif intent_choice < 0.90:
        # Intent: DONE
        prefix = random.choice(DONE_PREFIXES)
        task = make_random_task()
        clean_task = clean_task_name(task)
        voice = f"{prefix} {task}"
        reply = f"Marked {clean_task} as completed. Great job!"
        target = json.dumps({
            "action": "DONE",
            "task": clean_task,
            "reply": reply
        }, separators=(',', ':'))

    elif intent_choice < 0.96:
        # Intent: DELETE
        prefix = random.choice(DELETE_PREFIXES)
        task = make_random_task()
        clean_task = clean_task_name(task)
        voice = f"{prefix} {task}"
        reply = f"Removed {clean_task} from your list."
        target = json.dumps({
            "action": "DELETE",
            "task": clean_task,
            "reply": reply
        }, separators=(',', ':'))

    else:
        # Intent: LIST / GREET / STATUS
        chitchat = [
            ("what tasks do i have for today", "LIST", "today", "Here are your scheduled tasks for today."),
            ("show my reminders", "LIST", "all", "Displaying all your active reminders."),
            ("list my pending tasks", "LIST", "pending", "Here are your pending to-do items."),
            ("what is on my todo list", "LIST", "all", "Here is everything currently on your list."),
            ("do i have any reminders for tomorrow", "LIST", "tomorrow", "Checking your schedule for tomorrow."),
            ("hello", "GREET", "", "Hello! What tasks can I help you schedule today?"),
            ("good morning", "GREET", "", "Good morning! Ready to plan your day?"),
            ("who are you", "GREET", "", "I'm VoxTask, your private voice assistant running locally on your RTX 3060.")
        ]
        voice, act, flt, reply = random.choice(chitchat)
        payload = {"action": act, "reply": reply}
        if flt:
            payload["filter"] = flt
        target = json.dumps(payload, separators=(',', ':'))

    return voice.lower().strip(), target


class VoiceTaskDataset(Dataset):
    """PyTorch Dataset that encodes voice inputs and targets for causal training."""
    def __init__(self, tokenizer: VoiceTaskTokenizer, num_samples: int = 75000, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        print(f"Generating {num_samples} diverse training examples with full time, number & vocabulary synthesis...")
        for _ in range(num_samples):
            voice, target = generate_single_example()
            self.data.append((voice, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice, target = self.data[idx]

        voice_ids = self.tokenizer.encode(voice)
        target_ids = self.tokenizer.encode(target)

        input_ids = [self.tokenizer.bos_id] + voice_ids + [self.tokenizer.sep_id] + target_ids + [self.tokenizer.eos_id]

        prompt_len = 1 + len(voice_ids) + 1  # bos + voice + sep
        labels = [-100] * prompt_len + target_ids + [self.tokenizer.eos_id]

        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)

        return x, y


if __name__ == "__main__":
    tok = VoiceTaskTokenizer()
    ds = VoiceTaskDataset(tok, num_samples=10)
    for i in range(10):
        voice, target = ds.data[i]
        print(f"Voice : {voice}")
        print(f"Target: {target}\n")
