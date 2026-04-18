/**
 * pybind11 bindings for the SCALPEL C++ chunker.
 *
 * Exposes three functions to Python:
 *   split_sentences(text, abbreviations=[]) -> list[str]
 *   find_semantic_break(text, target_pos, window=200) -> int
 *   compute_chunk_boundaries(text, chars_per_chunk, overlap_chars=0, window=200)
 *       -> list[tuple[int, int]]
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chunker.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_scalpel_chunker, m) {
    m.doc() = "SCALPEL fast chunker — C++ implementation via pybind11";

    m.def(
        "split_sentences",
        [](const std::string& text, const std::vector<std::string>& abbreviations) {
            return scalpel::split_sentences(text, abbreviations);
        },
        py::arg("text"),
        py::arg("abbreviations") = std::vector<std::string>{},
        "Split text into sentences, protecting common academic abbreviations."
    );

    m.def(
        "find_semantic_break",
        [](const std::string& text, size_t target_pos, size_t window) {
            return scalpel::find_semantic_break(text, target_pos, window);
        },
        py::arg("text"),
        py::arg("target_pos"),
        py::arg("window") = size_t{200},
        "Find the best semantic break point near target_pos."
    );

    m.def(
        "compute_chunk_boundaries",
        [](const std::string& text,
           size_t chars_per_chunk,
           size_t overlap_chars,
           size_t window)
        {
            auto boundaries = scalpel::compute_chunk_boundaries(
                text, chars_per_chunk, overlap_chars, window
            );
            // Convert to list of (start, end) tuples for Python
            std::vector<std::pair<size_t, size_t>> result;
            result.reserve(boundaries.size());
            for (const auto& b : boundaries) {
                result.emplace_back(b.start, b.end);
            }
            return result;
        },
        py::arg("text"),
        py::arg("chars_per_chunk"),
        py::arg("overlap_chars") = size_t{0},
        py::arg("window") = size_t{200},
        "Compute chunk (start, end) boundaries in character space."
    );
}
