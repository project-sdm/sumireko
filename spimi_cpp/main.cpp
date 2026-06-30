#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

using DocId = std::size_t;

static constexpr std::size_t MAX_TERM_LEN = 23;
static constexpr std::size_t BATCH_SIZE = 4096;

#pragma pack(push, 1)
struct Posting {
    DocId doc_id;
    std::size_t tf;
};
#pragma pack(pop)

using PostingsList = std::vector<Posting>;
using Dictionary = std::unordered_map<std::string, PostingsList>;

struct Token {
    DocId doc_id;
    std::string term;
};

class TokenStream {
    py::object stream_obj;
    std::vector<DocId> doc_id_batch;
    std::vector<std::string> term_batch;
    std::size_t pos = 0;
    bool exhausted = false;

    void fetch_batch() {
        py::gil_scoped_acquire acquire;

        auto result = stream_obj.attr("next_batch")(BATCH_SIZE);
        doc_id_batch = result.cast<py::list>()[0].cast<std::vector<DocId>>();
        term_batch = result.cast<py::list>()[1].cast<std::vector<std::string>>();

        assert(doc_id_batch.size() == term_batch.size());
        pos = 0;

        if (doc_id_batch.empty())
            exhausted = true;
    }

public:
    explicit TokenStream(py::object stream_obj)
        : stream_obj{std::move(stream_obj)} {}

    std::optional<Token> next() {
        if (pos >= doc_id_batch.size() && !exhausted)
            fetch_batch();

        if (pos >= doc_id_batch.size())
            return std::nullopt;

        Token token{doc_id_batch[pos], std::move(term_batch[pos])};
        ++pos;

        return token;
    }
};

void add_to_dictionary(Dictionary& dictionary, std::string term) {
    dictionary.emplace(std::move(term), PostingsList{});
}

PostingsList& get_postings_list(Dictionary& dictionary, const std::string& term) {
    return dictionary.at(term);
}

void add_to_postings_list(PostingsList& postings_list, DocId doc_id) {
    if (!postings_list.empty() && postings_list.back().doc_id == doc_id)
        ++postings_list.back().tf;
    else
        postings_list.push_back({doc_id, 1});
}

std::vector<std::string> sort_terms(const Dictionary& dictionary) {
    std::vector<std::string> terms;
    terms.reserve(dictionary.size());

    for (const auto& [term, _] : dictionary)
        terms.emplace_back(term);

    std::sort(terms.begin(), terms.end());
    return terms;
}

void write_block_to_disk(const std::vector<std::string>& sorted_terms,
                         const Dictionary& dictionary,
                         const std::string& block_path_str) {
    std::filesystem::path postings_path(block_path_str);
    postings_path.replace_extension(".postings");

    std::filesystem::path dict_path(block_path_str);
    dict_path.replace_extension(".dict");

    std::ofstream dict_file(dict_path, std::ios::binary);
    std::ofstream postings_file(postings_path, std::ios::binary);

    for (const auto& term : sorted_terms) {
        const auto& postings_list = dictionary.at(term);
        std::size_t offset = postings_file.tellp();

        for (const auto& posting : postings_list) {
            auto tf_f = static_cast<float>(posting.tf);
            postings_file.write(reinterpret_cast<const char*>(&posting.doc_id),
                                sizeof(posting.doc_id));
            postings_file.write(reinterpret_cast<const char*>(&tf_f), sizeof(tf_f));
        }

        assert(term.size() <= MAX_TERM_LEN);

        char term_ch[MAX_TERM_LEN + 1] = {};
        std::memcpy(term_ch, term.c_str(), term.size() + 1);

        std::size_t len = postings_list.size();

        dict_file.write(term_ch, sizeof(term_ch));
        dict_file.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        dict_file.write(reinterpret_cast<const char*>(&len), sizeof(len));
    }
}

void spimi_invert(py::object token_stream_obj, std::string block_path, std::size_t max_memory) {
    py::gil_scoped_release release;

    TokenStream token_stream{std::move(token_stream_obj)};
    std::optional<Token> token;

    Dictionary dictionary;
    std::size_t bytes_used = 0;

    while (bytes_used < max_memory && (token = token_stream.next())) {
        auto [it, inserted] = dictionary.emplace(token->term, PostingsList{});

        if (inserted)
            bytes_used += 16 + token->term.size();

        add_to_postings_list(it->second, token->doc_id);
        bytes_used += 56;
    }

    auto sorted_terms = sort_terms(dictionary);
    write_block_to_disk(sorted_terms, dictionary, block_path);
}

PYBIND11_MODULE(spimi_cpp, m) {
    m.def("spimi_invert", &spimi_invert);
}
