// Browser JavaScript Tokenizer matching PyTorch VoiceTaskTokenizer
class VoiceTaskTokenizer {
  constructor(vocabData) {
    this.specialTokens = {
      "<pad>": 0,
      "<bos>": 1,
      "<sep>": 2,
      "<eos>": 3
    };
    this.tokenToId = vocabData.token_to_id;
    this.idToToken = {};
    for (const [k, v] of Object.entries(this.tokenToId)) {
      this.idToToken[v] = k;
    }
    this.vocabSize = vocabData.vocab_size;
    this.paddedVocabSize = vocabData.padded_vocab_size;

    // Filter keywords for greedy multi-char matching
    this.keywords = Object.keys(this.tokenToId)
      .filter(k => k.length > 1 && !k.startsWith("<"))
      .sort((a, b) => b.length - a.length);
  }

  get bosId() { return 1; }
  get sepId() { return 2; }
  get eosId() { return 3; }
  get padId() { return 0; }

  encode(text) {
    const tokens = [];
    let i = 0;
    const n = text.length;

    while (i < n) {
      let matched = false;
      for (const kw of this.keywords) {
        if (text.startsWith(kw, i)) {
          tokens.push(this.tokenToId[kw]);
          i += kw.length;
          matched = true;
          break;
        }
      }

      if (!matched) {
        const char = text[i];
        if (char in this.tokenToId) {
          tokens.push(this.tokenToId[char]);
        } else {
          tokens.push(this.tokenToId[" "] || 0);
        }
        i += 1;
      }
    }
    return tokens;
  }

  decode(tokenIds) {
    const res = [];
    for (const tid of tokenIds) {
      if (tid <= 3) continue; // Skip special tokens
      const str = this.idToToken[tid] || "";
      res.push(str);
    }
    return res.join("");
  }
}

// Client-side prompt normalizer (replicates server.py logic)
function normalizeVoicePrompt(text) {
  let s = text.trim();
  s = s.replace(/\bwirte\b/gi, 'write');
  s = s.replace(/\btommorrow\b/gi, 'tomorrow');
  s = s.replace(/\batomrow\b/gi, 'tomorrow');
  s = s.replace(/\basop\b/gi, 'a sop');

  const wrapperPatterns = [
    /^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?write\s+a\s+(?:to-?do|to\s+do)\s+list\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?make\s+a\s+(?:to-?do|to\s+do)\s+list\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?create\s+a\s+(?:to-?do|to\s+do)\s+list\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?write\s+(?:down\s+)?a\s+(?:to-?do|to\s+do)\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?better\s+to\s+do\s+this\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:could\s+you\s+|can\s+you\s+)?better\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:it(?:'s|\s+is)\s+)?better\s+to\s+/i,
    /^(?:please\s+|hey\s+)?(?:i'd\s+better|we'd\s+better)\s+/i,
  ];

  for (const pat of wrapperPatterns) {
    const match = s.match(pat);
    if (match) {
      const rest = s.substring(match[0].length).trim();
      const hasTime = /\b(at\s+\d+|tomorrow|tonight|today|am|pm|in\s+\d+)\b/i.test(rest);
      if (hasTime) {
        return `remind me to ${rest}`;
      } else {
        return `add a todo to ${rest}`;
      }
    }
  }
  return s;
}

function cleanDisplayTitle(rawTitle) {
  let s = rawTitle.trim();
  s = s.replace(/\s+(?:at|on|for|in|by|to)$/i, '');

  const leadingFillers = [
    /^(?:better\s+to\s+do\s+this\s+to\s+)/i,
    /^(?:better\s+to\s+do\s+this\s+)/i,
    /^(?:better\s+to\s+)/i,
    /^(?:do\s+this\s+to\s+write\s+)/i,
    /^(?:do\s+this\s+to\s+)/i,
    /^(?:do\s+this\s+)/i,
    /^(?:list\s+to\s+write\s+)/i,
    /^(?:wrist\s+to\s+write\s+)/i,
    /^(?:make\s+sure\s+to\s+)/i,
    /^(?:remember\s+to\s+)/i,
    /^(?:need\s+to\s+)/i,
    /^(?:have\s+to\s+)/i,
    /^(?:got\s+to\s+)/i,
    /^(?:i\s+need\s+to\s+)/i,
    /^(?:i\s+have\s+to\s+)/i,
    /^(?:try\s+to\s+)/i,
    /^(?:about\s+)/i,
    /^(?:to\s+)/i,
  ];

  for (const pat of leadingFillers) {
    s = s.replace(pat, '').trim();
  }

  s = s.replace(/\basop\b/gi, 'a SOP');
  s = s.replace(/\bsop\b/gi, 'SOP');
  s = s.replace(/\bphd\b/gi, 'PhD');
  s = s.replace(/\bkent\b/gi, 'Kent');

  if (s.length > 0) {
    s = s.charAt(0).toUpperCase() + s.slice(1);
  }
  return s || "Untitled Task";
}

function cleanReplyText(replyStr, cleanTask, dueTime = "") {
  if (!replyStr) {
    if (dueTime) return `I've set a reminder for ${cleanTask} ${dueTime}.`;
    return `Added ${cleanTask} to your list.`;
  }
  let s = replyStr;
  s = s.replace(/\basop\b/gi, 'a SOP');
  s = s.replace(/\bsop\b/gi, 'SOP');
  s = s.replace(/\bphd\b/gi, 'PhD');
  s = s.replace(/\bkent\b/gi, 'Kent');

  if (cleanTask && (/reminder for/i.test(s) || /added/i.test(s))) {
    if (dueTime) {
      return `I've set a reminder for ${cleanTask} ${dueTime}.`;
    } else {
      return `I've saved ${cleanTask} to your list.`;
    }
  }
  return s.trim();
}
