"""
Interactive Inference Script for Your Custom-Trained Voice-to-Do SLM.
Loads checkpoint_best.pt, tokenizes voice input, and generates structured JSON actions.
"""

import os
import json
import torch

from model import VoiceTaskSLM
from tokenizer import VoiceTaskTokenizer
from server import normalize_voice_prompt, clean_display_title, clean_reply_text


def load_model(checkpoint_path: str = "checkpoint_best.pt", device_name: str = "auto"):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    tokenizer = VoiceTaskTokenizer()

    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  Checkpoint '{checkpoint_path}' not found!")
        print("    You must train your model first by running:")
        print("    👉 python train.py\n")
        return None, None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = VoiceTaskSLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded custom model from {checkpoint_path} (Trained on Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
    return model, tokenizer, device


def parse_voice_command(model, tokenizer, device, voice_text: str):
    """
    Feeds voice text into the model and returns parsed JSON.
    """
    normalized_text = normalize_voice_prompt(voice_text)
    voice_ids = tokenizer.encode(normalized_text.lower().strip())
    # Format: <bos> {voice} <sep>
    input_ids = [tokenizer.bos_id] + voice_ids + [tokenizer.sep_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(
        input_tensor,
        max_new_tokens=150,
        temperature=0.1,  # Low temperature for deterministic, structured JSON
        eos_id=tokenizer.eos_id
    )

    # Extract only the newly generated tokens after <sep>
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
        return parsed_json, raw_output
    except json.JSONDecodeError:
        return None, raw_output


def interactive_loop():
    print("=" * 60)
    print("  🎙️ VOICE-TO-DO SMALL LANGUAGE MODEL (SLM) INFERENCE")
    print("=" * 60)

    model, tokenizer, device = load_model()
    if model is None:
        return

    sample_commands = [
        "Remind me to call Mom tomorrow at 4 pm",
        "Add buy groceries to my shopping list",
        "Please remind me to backup VPS database tonight at 8 pm",
        "Mark submit research paper as completed",
        "Show my pending tasks for today",
    ]

    print("\n💡 Try speaking or typing commands like:")
    for s in sample_commands:
        print(f"   • \"{s}\"")
    print("\n(Type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            user_input = input("🎙️ Voice Command > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            parsed, raw = parse_voice_command(model, tokenizer, device, user_input)

            if parsed:
                print("\n✨ Model Output (Valid JSON):")
                print(json.dumps(parsed, indent=2))
                
                # Execute action
                act = parsed.get("action")
                if act == "REMIND":
                    print(f"⏰ [SCHEDULER] Scheduled alert: '{parsed.get('task')}' for {parsed.get('time')}")
                elif act == "TODO":
                    print(f"📝 [DATABASE] Saved to-do: '{parsed.get('task')}' (Category: {parsed.get('category', 'general')})")
                elif act == "DONE":
                    print(f"✅ [DATABASE] Marked task completed: '{parsed.get('task')}'")
                elif act == "DELETE":
                    print(f"🗑️ [DATABASE] Deleted task: '{parsed.get('task')}'")
                elif act == "LIST":
                    print(f"📋 [DATABASE] Querying tasks filter: {parsed.get('filter')}")
            else:
                print(f"\n⚠️ Raw generation was: {raw}")

            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    interactive_loop()
