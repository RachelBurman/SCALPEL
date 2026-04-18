#pragma once

#include <string>
#include <vector>

namespace scalpel {

struct ChunkBoundary {
    size_t start;
    size_t end;
};

/**
 * Split text into sentences, respecting common academic abbreviations.
 * Abbreviations containing dots (e.g. "et al.", "Fig.") are protected
 * so they don't trigger false sentence boundaries.
 */
std::vector<std::string> split_sentences(
    const std::string& text,
    const std::vector<std::string>& abbreviations = {}
);

/**
 * Find the best semantic break point near target_pos.
 * Prefers paragraph breaks > sentence breaks > word breaks > character position.
 */
size_t find_semantic_break(
    const std::string& text,
    size_t target_pos,
    size_t window = 200
);

/**
 * Compute chunk (start, end) boundaries in character space.
 * Caller converts token budget to chars_per_chunk and handles actual token counting.
 */
std::vector<ChunkBoundary> compute_chunk_boundaries(
    const std::string& text,
    size_t chars_per_chunk,
    size_t overlap_chars = 0,
    size_t window = 200
);

} // namespace scalpel
