const fs = require("fs");
const { XMLParser } = require("fast-xml-parser");
const readline = require("readline");

const CONFIG = {
  MIN_WORD_LENGTH: 4,
  MAX_WORD_LENGTH: 15,
  DEFAULT_TARGET_WORD_COUNT: 4000,
  OUTPUT_FILE: "crossword-words.json",
};

const EXCLUDE_PATTERNS = [
  /^\p{Lu}/u,
  /[^a-záéíóúüñ]/i,
  /(.)\1{2,}/,
];

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "in",
  "is",
  "it",
  "its",
  "of",
  "on",
  "or",
  "that",
  "the",
  "this",
  "to",
  "with",
  "without",
]);

class WiktionaryParser {
  constructor(targetWordCount = CONFIG.DEFAULT_TARGET_WORD_COUNT) {
    this.words = new Map();
    this.targetWordCount = targetWordCount;
    this.xmlParser = new XMLParser({
      ignoreAttributes: false,
      parseTagValue: false,
    });
  }

  isSuitableWord(word) {
    if (
      word.length < CONFIG.MIN_WORD_LENGTH ||
      word.length > CONFIG.MAX_WORD_LENGTH
    ) {
      return false;
    }

    for (const pattern of EXCLUDE_PATTERNS) {
      if (pattern.test(word)) {
        return false;
      }
    }

    return true;
  }

  extractSpanishSection(text) {
    return text.match(
      /(?:^|\n)==\s*(?:\{\{\s*lengua\s*\|\s*es\s*\}\}|español)\s*==\s*\n([\s\S]*?)(?=\n==[^=\n]+==\s*\n|$)/i,
    );
  }

  extractPOSSection(spanishSection) {
    return spanishSection.match(
      /(?:^|\n)={3,4}\s*\{\{\s*((?:sustantivo(?:\s+(?:masculino|femenino|propio))?|verbo(?:\s+\w+)?|adjetivo(?:\s+\w+)?|adverbio(?:\s+\w+)?|pronombre(?:\s+\w+)?|preposición(?:\s+\w+)?|conjunción(?:\s+\w+)?))[^}]*\|\s*es\b[^}]*\}\}\s*={3,4}\s*\n([\s\S]*?)(?=\n={3,4}\s*|$)/i,
    );
  }

  tokenize(text) {
    return (text.toLowerCase().match(/[a-záéíóúüñ]+/gi) || []).filter(Boolean);
  }

  hasSpanishUsageExclusion(spanishSection) {
    return (
      /\{\{(?:obsoleto|arcaico|desusado|raro|dialectal|informal|coloquial|anticuado)\b/i.test(
        spanishSection,
      ) ||
      /\{\{contexto\|[^}]*\b(?:obsoleto|arcaico|desusado|raro|dialectal|informal|coloquial|anticuado)\b/i.test(
        spanishSection,
      )
    );
  }

  extractDefinitionLine(posSection) {
    const hashStyle = posSection.match(/^#(?![:*])\s*(.+)$/m);
    if (hashStyle) return hashStyle[1];

    const semicolonStyle = posSection.match(
      /^;\s*\d+\s*(?:\{\{[^}]+\}\}\s*:)?\s*(.+)$/m,
    );
    if (semicolonStyle) return semicolonStyle[1];

    const colonStyle = posSection.match(
      /^:\s*\d+\s*(?:\{\{[^}]+\}\}\s*:)?\s*(.+)$/m,
    );
    if (colonStyle) return colonStyle[1];

    return null;
  }

  escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  getWordVariants(word) {
    const lower = word.toLowerCase();
    const variants = new Set([lower]);

    if (lower.endsWith("y") && lower.length > 2) {
      variants.add(`${lower.slice(0, -1)}ies`);
    } else if (
      lower.endsWith("s") ||
      lower.endsWith("x") ||
      lower.endsWith("z") ||
      lower.endsWith("ch") ||
      lower.endsWith("sh")
    ) {
      variants.add(`${lower}es`);
    } else {
      variants.add(`${lower}s`);
    }

    if (lower.endsWith("ies") && lower.length > 4) {
      variants.add(`${lower.slice(0, -3)}y`);
    }
    if (lower.endsWith("es") && lower.length > 3) {
      variants.add(lower.slice(0, -2));
    }
    if (lower.endsWith("s") && lower.length > 3) {
      variants.add(lower.slice(0, -1));
    }

    return variants;
  }

  hasAnswerVariant(text, word) {
    if (!text) return false;
    const variants = this.getWordVariants(word);
    for (const variant of variants) {
      const regex = new RegExp(`\\b${this.escapeRegExp(variant)}\\b`, "i");
      if (regex.test(text)) return true;
    }
    return false;
  }

  cleanDefinition(definition) {
    return definition
      .replace(/<ref[^>]*>[\s\S]*?<\/ref>/gi, " ")
      .replace(/<ref[^\/]*\/>/gi, " ")
      .replace(/\{\{[^}]+\}\}/g, " ")
      .replace(/\[\[[^\]|]+\|([^\]]+)\]\]/g, "$1")
      .replace(/\[\[([^\]#|]+)(?:#[^\]]+)?\]\]/g, "$1")
      .replace(/[\[\]]/g, "")
      .replace(/'''([^']+)'''/g, "$1")
      .replace(/''([^']+)''/g, "$1")
      .replace(/\([^)]*\)/g, " ")
      .replace(/\s+/g, " ")
      .replace(/\s+([,.;:!?])/g, "$1")
      .trim()
      .replace(/^[:;,\.\s-]+/, "");
  }

  buildHint(baseText, word) {
    const variants = this.getWordVariants(word);
    const tokens = baseText
      .split(/\s+/)
      .map((token) => token.replace(/[^a-zA-Z-]/g, ""))
      .filter(Boolean)
      .filter((token) => !STOP_WORDS.has(token.toLowerCase()))
      .filter((token) => !variants.has(token.toLowerCase()));

    let hintWords = tokens.slice(0, 6);
    if (hintWords.length < 2) {
      hintWords = ["Common", "English", "term"];
    }
    if (hintWords.length > 8) {
      hintWords = hintWords.slice(0, 8);
    }

    let hint = hintWords.join(" ").trim();
    hint = hint.charAt(0).toUpperCase() + hint.slice(1);
    return hint;
  }

  buildDescription(cleanedDefinition, word) {
    let description = cleanedDefinition.split(/[.;]/)[0].trim();
    if (!description) return null;

    if (description.length > 170) {
      description = description.slice(0, 170).replace(/\s+\S*$/, "").trim();
    }

    if (!/[.!?]$/.test(description)) {
      description += ".";
    }

    const descriptionWordCount = description.split(/\s+/).filter(Boolean).length;
    if (descriptionWordCount < 5) return null;

    if (!this.hasAnswerVariant(description, word)) {
      return description;
    }

    const contextTerms = this.tokenize(cleanedDefinition)
      .filter((token) => !STOP_WORDS.has(token))
      .filter((token) => !this.getWordVariants(word).has(token))
      .slice(0, 3);

    if (!contextTerms.length) return null;

    const rewritten = `A term related to ${contextTerms.join(", ")}.`;
    if (this.hasAnswerVariant(rewritten, word)) return null;
    return rewritten;
  }

  inferCategory(word, cleanedDefinition, partOfSpeech) {
    const tokenSet = new Set(this.tokenize(`${word} ${cleanedDefinition}`));
    const has = (terms) => terms.some((term) => tokenSet.has(term));

    if (has(["language", "linguistics", "grammar", "vocabulary", "dictionary"])) return "school";
    if (has(["animal", "mammal", "bird", "fish", "insect", "reptile", "amphibian"])) return "animals";
    if (has(["fruit", "vegetable", "food", "drink", "meal", "edible", "flavor"])) return "food";
    if (has(["color", "colour", "hue", "shade", "pigment"])) return "colors";
    if (has(["minute", "hour", "day", "week", "month", "year", "season", "calendar", "time"])) return "time";
    if (has(["computer", "software", "internet", "digital", "device", "data", "code", "network"])) return "technology";
    if (has(["school", "student", "teacher", "class", "lesson", "education", "study"])) return "school";
    if (has(["house", "kitchen", "bedroom", "furniture", "home", "appliance"])) return "home";
    if (has(["city", "country", "village", "region", "location", "place"])) return "places";
    if (has(["car", "train", "plane", "bus", "ship", "vehicle", "transport"])) return "transport";
    if (has(["ball", "game", "sport", "team", "match", "athlete"])) return "sports";
    if (has(["body", "muscle", "bone", "organ", "skin", "blood"])) return "body";
    if (has(["shirt", "pants", "dress", "shoe", "hat", "clothing", "fabric"])) return "clothing";
    if (has(["tool", "object", "item", "instrument", "device", "machine"])) return "objects";
    if (has(["job", "office", "salary", "task", "career", "work"])) return "work";
    if (has(["person", "human", "man", "woman", "child", "adult"])) return "people";
    if (has(["plant", "tree", "river", "mountain", "ocean", "weather", "earth", "nature"])) return "nature";
    if (has(["science", "physics", "chemistry", "biology", "molecule", "energy"])) return "science";

    if (partOfSpeech === "Verb") return "actions";
    if (partOfSpeech === "Adverb" || partOfSpeech === "Adjective") return "abstract";

    return "other";
  }

  toGameEntry(title, definition, partOfSpeech) {
    const word = title.toLowerCase();
    const cleanedDefinition = this.cleanDefinition(definition);
    if (!cleanedDefinition || cleanedDefinition.length < 5) return null;

    const description = this.buildDescription(cleanedDefinition, word);
    if (!description) return null;

    const hint = this.buildHint(description, word);
    if (this.hasAnswerVariant(hint, word)) return null;
    if (this.hasAnswerVariant(description, word)) return null;

    const entry = {
      word,
      hint,
      description,
      difficulty: this.getDifficulty(title.length),
      category: this.inferCategory(word, cleanedDefinition, partOfSpeech),
    };

    return entry;
  }

  parseWikitext(title, text) {
    if (!text || typeof text !== "string") return null;

    const spanishMatch = this.extractSpanishSection(text);
    if (!spanishMatch) return null;

    const spanishSection = spanishMatch[1];

    if (this.hasSpanishUsageExclusion(spanishSection)) {
      return null;
    }

    const posMatch = this.extractPOSSection(spanishSection);
    if (!posMatch) return null;

    const partOfSpeech = posMatch[1];
    const definitionSection = posMatch[2];
    const definitionLine = this.extractDefinitionLine(definitionSection);
    if (!definitionLine) return null;

    return this.toGameEntry(title, definitionLine, partOfSpeech);
  }

  getDifficulty(wordLength) {
    if (wordLength >= 9) return "hard";
    if (wordLength >= 6) return "medium";
    return "easy";
  }

  resolveRevisionText(textNode) {
    if (typeof textNode === "string") return textNode;
    if (textNode && typeof textNode === "object") return textNode["#text"] || "";
    return "";
  }

  addWord(wordData) {
    if (!wordData || !wordData.word) return;
    if (this.words.size >= this.targetWordCount) return;
    if (this.words.has(wordData.word)) return;
    this.words.set(wordData.word, wordData);
  }

  parsePageContent(pageContent) {
    try {
      const result = this.xmlParser.parse(pageContent);
      if (result.page) {
        this.processPage(result.page);
      }
    } catch (err) {
      console.error("\n[XML PARSE ERROR] Failed to parse page block.");
      console.error(`[XML PARSE ERROR] ${err.message}`);
      console.error(
        `[XML PARSE ERROR] Snippet: ${pageContent.slice(0, 300).replace(/\s+/g, " ")}`,
      );
    }
  }

  processPage(page) {
    if (!page.title || !page.revision?.text) return;

    const title = page.title.trim();
    const rawText = this.resolveRevisionText(page.revision.text);
    if (!rawText) return;
    if (!this.isSuitableWord(title)) return;

    const wordData = this.parseWikitext(title, rawText);
    this.addWord(wordData);
  }

  async parseXMLDump(inputFile) {
    console.log("Starting to parse Wiktionary dump...");
    console.log("This may take several minutes for large files.\n");

    return new Promise((resolve, reject) => {
      const fileStream = fs.createReadStream(inputFile, { encoding: "utf8" });
      let inPage = false;
      let pageContent = "";

      const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity,
      });

      rl.on("line", (line) => {
        let cursor = 0;

        while (cursor < line.length) {
          if (!inPage) {
            const start = line.indexOf("<page>", cursor);
            if (start === -1) break;
            inPage = true;
            pageContent = "<page>";
            cursor = start + "<page>".length;
            continue;
          }

          const end = line.indexOf("</page>", cursor);
          if (end === -1) {
            pageContent += line.slice(cursor) + "\n";
            cursor = line.length;
          } else {
            pageContent += line.slice(cursor, end) + "</page>";
            this.parsePageContent(pageContent);
            pageContent = "";
            inPage = false;
            cursor = end + "</page>".length;
          }
        }

        if (this.words.size >= this.targetWordCount) {
          rl.close();
          fileStream.destroy();
        }
      });

      rl.on("error", reject);
      fileStream.on("error", reject);

      rl.on("close", () => {
        resolve();
      });
    });
  }

  saveToJSON() {
    const wordsArray = Array.from(this.words.values()).sort((a, b) =>
      a.word.localeCompare(b.word),
    );

    fs.writeFileSync(CONFIG.OUTPUT_FILE, JSON.stringify(wordsArray, null, 2));
    console.log(`\nSaved ${wordsArray.length} words to ${CONFIG.OUTPUT_FILE}`);

    const difficultyStats = { easy: 0, medium: 0, hard: 0 };
    wordsArray.forEach((entry) => {
      if (difficultyStats[entry.difficulty] !== undefined) {
        difficultyStats[entry.difficulty] += 1;
      }
    });

    console.log("Difficulty distribution:");
    console.log(`  easy: ${difficultyStats.easy}`);
    console.log(`  medium: ${difficultyStats.medium}`);
    console.log(`  hard: ${difficultyStats.hard}`);
  }
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log("Usage: node crosswordWords.js <path-to-wiktionary-dump.xml> [target-word-count]");
    console.log("\nExamples:");
    console.log("  node crosswordWords.js dump.xml");
    console.log("  node crosswordWords.js dump.xml 100");
    console.log("\nYou can download the dump from:");
    console.log("https://dumps.wikimedia.org/enwiktionary/latest/");
    process.exit(1);
  }

  const inputFile = args[0];
  const requestedTargetCountRaw = args[1];

  if (!fs.existsSync(inputFile)) {
    console.error(`Error: File not found: ${inputFile}`);
    process.exit(1);
  }

  let targetWordCount = CONFIG.DEFAULT_TARGET_WORD_COUNT;
  if (requestedTargetCountRaw !== undefined) {
    const isIntegerText = /^\d+$/.test(String(requestedTargetCountRaw).trim());
    const parsed = Number.parseInt(requestedTargetCountRaw, 10);

    if (!isIntegerText || !Number.isInteger(parsed) || parsed <= 0) {
      console.error(
        `Error: Invalid target word count "${requestedTargetCountRaw}". Please provide a positive integer (e.g. 10, 100, 500).`,
      );
      process.exit(1);
    }

    targetWordCount = parsed;
  }

  const parser = new WiktionaryParser(targetWordCount);

  try {
    await parser.parseXMLDump(inputFile);
    parser.saveToJSON();
    console.log("\nDone!");
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

main();
