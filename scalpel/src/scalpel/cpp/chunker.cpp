/**
 * SCALPEL Fast Chunker — C++ implementation
 *
 * Implements the hot path from chunker.py:
 *   - Abbreviation-aware sentence splitting
 *   - Semantic break detection (paragraph → sentence → word → character)
 *   - Chunk boundary generation with overlap
 *
 * Token counting is deliberately left to Python/tiktoken (already native).
 * This layer handles the string search work that is slow in pure Python.
 */

#include "chunker.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace scalpel {

// ── Abbreviation protection ───────────────────────────────────────────────

static const std::vector<std::string> DEFAULT_ABBREVIATIONS = {
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
    "vs.", "etc.", "i.e.", "e.g.", "cf.", "al.", "Fig.",
    "Eq.", "Ref.", "Vol.", "No.", "pp.", "et al.", "Ph.D.",
    "Inc.", "Ltd.", "Corp.", "Co.", "St.", "Mt.", "Ft.",
};

static constexpr std::string_view DOT_PLACEHOLDER = "\x01";

static std::string protect_abbreviations(
    const std::string& text,
    const std::vector<std::string>& abbreviations)
{
    std::string result = text;
    for (const auto& abbr : abbreviations) {
        size_t pos = 0;
        while ((pos = result.find(abbr, pos)) != std::string::npos) {
            // Replace the dot in the abbreviation with placeholder
            size_t dot_pos = result.find('.', pos);
            if (dot_pos != std::string::npos && dot_pos < pos + abbr.size()) {
                result.replace(dot_pos, 1, DOT_PLACEHOLDER);
            }
            pos += abbr.size();
        }
    }
    return result;
}

static void restore_abbreviations(std::string& text) {
    size_t pos = 0;
    while ((pos = text.find(DOT_PLACEHOLDER, pos)) != std::string::npos) {
        text.replace(pos, DOT_PLACEHOLDER.size(), ".");
        pos += 1;
    }
}

// ── Sentence splitting ────────────────────────────────────────────────────

std::vector<std::string> split_sentences(
    const std::string& text,
    const std::vector<std::string>& abbreviations)
{
    const auto& abbrs = abbreviations.empty() ? DEFAULT_ABBREVIATIONS : abbreviations;
    std::string protected_text = protect_abbreviations(text, abbrs);

    std::vector<std::string> sentences;
    std::string current;
    current.reserve(256);

    for (size_t i = 0; i < protected_text.size(); ++i) {
        char c = protected_text[i];
        current += c;

        // Check for sentence-ending punctuation
        if (c == '.' || c == '!' || c == '?') {
            // Look ahead for whitespace followed by an uppercase letter
            size_t j = i + 1;
            while (j < protected_text.size() && protected_text[j] == ' ') {
                ++j;
            }
            if (j < protected_text.size() && std::isupper((unsigned char)protected_text[j])) {
                restore_abbreviations(current);
                // Trim
                size_t start = current.find_first_not_of(" \t\n\r");
                if (start != std::string::npos) {
                    sentences.push_back(current.substr(start));
                }
                current.clear();
                // Skip the whitespace
                i = j - 1;
            }
        }
    }

    if (!current.empty()) {
        restore_abbreviations(current);
        size_t start = current.find_first_not_of(" \t\n\r");
        if (start != std::string::npos) {
            sentences.push_back(current.substr(start));
        }
    }

    return sentences;
}

// ── Semantic break detection ──────────────────────────────────────────────

/**
 * Find the best break point near target_pos.
 *
 * Priority:
 *   1. Double newline (paragraph break) in [target - window, target]
 *   2. Sentence-ending punctuation followed by space in the same range
 *   3. Single space (word break)
 *   4. target_pos itself
 */
size_t find_semantic_break(
    const std::string& text,
    size_t target_pos,
    size_t window)
{
    if (target_pos >= text.size()) {
        return text.size();
    }

    size_t search_start = (target_pos > window) ? target_pos - window : 0;
    size_t search_end   = std::min(text.size(), target_pos + window);

    // 1. Look for paragraph break (double newline) before target
    {
        size_t best = std::string::npos;
        size_t pos = search_start;
        while (pos < target_pos && pos < text.size() - 1) {
            if (text[pos] == '\n' && text[pos + 1] == '\n') {
                best = pos + 2;  // start of new paragraph
            }
            ++pos;
        }
        if (best != std::string::npos && best > search_start + window / 2) {
            return best;
        }
    }

    // 2. Look for sentence break before target
    {
        size_t best = std::string::npos;
        for (size_t pos = search_start; pos < target_pos && pos + 1 < text.size(); ++pos) {
            char c = text[pos];
            if ((c == '.' || c == '!' || c == '?') && text[pos + 1] == ' ') {
                best = pos + 2;
            }
        }
        if (best != std::string::npos && best > search_start + window / 2) {
            return best;
        }
    }

    // 3. Word break — last space before target
    {
        size_t pos = target_pos;
        while (pos > search_start) {
            if (text[pos] == ' ') {
                return pos + 1;
            }
            --pos;
        }
    }

    return target_pos;
}

// ── Chunk boundary generation ─────────────────────────────────────────────

/**
 * Generate chunk boundaries for text.
 *
 * Works in character space — caller is responsible for token counting.
 * Returns a list of (start, end) character pairs.
 *
 * @param text            Input text
 * @param chars_per_chunk Estimated characters per chunk (caller derives from token budget)
 * @param overlap_chars   Overlap between consecutive chunks in characters
 * @param window          Search window for semantic break detection
 */
std::vector<ChunkBoundary> compute_chunk_boundaries(
    const std::string& text,
    size_t chars_per_chunk,
    size_t overlap_chars,
    size_t window)
{
    std::vector<ChunkBoundary> boundaries;

    if (text.empty() || chars_per_chunk == 0) {
        return boundaries;
    }

    size_t text_len   = text.size();
    size_t current    = 0;

    while (current < text_len) {
        size_t target_end = std::min(current + chars_per_chunk, text_len);

        size_t break_pos;
        if (target_end >= text_len) {
            break_pos = text_len;
        } else {
            break_pos = find_semantic_break(text, target_end, window);
            // Ensure progress
            if (break_pos <= current) {
                break_pos = std::min(current + chars_per_chunk, text_len);
            }
        }

        boundaries.push_back({current, break_pos});

        if (break_pos >= text_len) {
            break;
        }

        // Step back by overlap_chars for the next chunk start
        if (overlap_chars > 0 && break_pos > overlap_chars) {
            size_t overlap_start = break_pos - overlap_chars;
            size_t next_start = find_semantic_break(text, overlap_start, window / 2);
            current = (next_start >= break_pos) ? break_pos : next_start;
        } else {
            current = break_pos;
        }
    }

    return boundaries;
}

} // namespace scalpel
