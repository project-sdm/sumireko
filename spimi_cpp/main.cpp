#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

template<typename T>
struct TrackingAllocator {
    std::size_t& bytes;

    using value_type = T;

    explicit TrackingAllocator(std::size_t& bytes)
        : bytes{bytes} {}

    template<typename U>
    TrackingAllocator(const TrackingAllocator<U>& other)
        : bytes{other.bytes} {}

    value_type* allocate(std::size_t n) {
        bytes += n * sizeof(value_type);
        return static_cast<value_type*>(::operator new(n * sizeof(value_type)));
    }

    void deallocate(value_type* ptr, std::size_t n) {
        bytes -= n * sizeof(value_type);
        ::operator delete(ptr);
    }
};

static constexpr std::size_t MAX_TERM_LEN = 23;
static constexpr std::size_t BATCH_SIZE = 4096;

using DocId = std::size_t;

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

struct Posting {
    DocId doc_id;
    std::size_t tf;
};

struct StringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view sv) const {
        return std::hash<std::string_view>{}(sv);
    }
};

struct StringEqual {
    using is_transparent = void;

    bool operator()(std::string_view a, std::string_view b) const {
        return a == b;
    }
};

using Term = std::basic_string<char, std::char_traits<char>, TrackingAllocator<char>>;
using PostingsList = std::vector<Posting, TrackingAllocator<Posting>>;
using DictEntry = std::pair<const Term, PostingsList>;
using Dictionary =
    std::unordered_map<Term, PostingsList, StringHash, StringEqual, TrackingAllocator<DictEntry>>;

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
        auto it = dictionary.find(std::string_view{term});
        assert(it != dictionary.end());

        const auto& postings_list = it->second;
        std::size_t offset = postings_file.tellp();

        for (const auto& posting : postings_list) {
            auto tf_f = static_cast<float>(posting.tf);
            postings_file.write(reinterpret_cast<const char*>(&posting.doc_id),
                                sizeof(posting.doc_id));
            postings_file.write(reinterpret_cast<const char*>(&tf_f), sizeof(tf_f));
        }

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

    std::size_t bytes_used = 0;

    // these allocators track how many bytes dictionary uses
    TrackingAllocator<char> char_alloc{bytes_used};
    TrackingAllocator<Posting> posting_alloc{bytes_used};
    TrackingAllocator<DictEntry> dict_alloc{bytes_used};

    Dictionary dictionary{dict_alloc};

    while (bytes_used < max_memory && (token = token_stream.next())) {
        Term term{token->term, char_alloc};

        auto [it, _] = dictionary.try_emplace(std::move(term), PostingsList{posting_alloc});
        add_to_postings_list(it->second, token->doc_id);
    }

    auto sorted_terms = sort_terms(dictionary);
    write_block_to_disk(sorted_terms, dictionary, block_path);
}

PYBIND11_MODULE(spimi_cpp, m) {
    m.def("spimi_invert", &spimi_invert);
}
