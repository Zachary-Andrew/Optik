#pragma once

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <stdexcept>

namespace tpoptoa {

struct SeqRecord {
    std::string name;  // header line (without '>' or '@')
    std::string seq;   // nucleotide sequence
};

// Streaming reader for FASTA/FASTQ. Callback receives each record.
// Handles '-' as stdin. Returns early if callback returns false.
template <typename Callback>
void iterate_sequences(const char* path, Callback&& cb) {
    FILE* fh = (path[0] == '-' && path[1] == '\0') ? stdin : std::fopen(path, "r");
    if (!fh)
        throw std::runtime_error(std::string("cannot open file: ") + path);

    char*   line_buf  = nullptr;
    size_t  buf_cap   = 0;
    ssize_t line_len  = 0;

    SeqRecord rec;
    bool in_fastq_qual = false;
    bool have_record   = false;

    auto flush_record = [&]() {
        if (have_record && !rec.seq.empty())
            cb(rec);
        rec.name.clear();
        rec.seq.clear();
        have_record = false;
    };

    while ((line_len = getline(&line_buf, &buf_cap, fh)) != -1) {
        while (line_len > 0 &&
               (line_buf[line_len - 1] == '\n' || line_buf[line_len - 1] == '\r'))
            line_buf[--line_len] = '\0';

        if (line_len == 0) continue;

        char first = line_buf[0];

        if (first == '>') {                     // FASTA header
            flush_record();
            in_fastq_qual = false;
            have_record   = true;
            std::string_view sv(line_buf + 1, static_cast<std::size_t>(line_len - 1));
            auto space = sv.find_first_of(" \t");
            rec.name = (space == std::string_view::npos) ? std::string(sv)
                                                          : std::string(sv.substr(0, space));
            continue;
        }

        if (first == '@' && !have_record) {     // FASTQ header
            flush_record();
            in_fastq_qual = false;
            have_record   = true;
            std::string_view sv(line_buf + 1, static_cast<std::size_t>(line_len - 1));
            auto space = sv.find_first_of(" \t");
            rec.name = (space == std::string_view::npos) ? std::string(sv)
                                                          : std::string(sv.substr(0, space));
            continue;
        }

        if (first == '+' && have_record) {      // FASTQ quality separator
            in_fastq_qual = true;
            continue;
        }

        if (in_fastq_qual) continue;            // skip quality lines

        // Sequence line
        if (have_record) {
            rec.seq.append(line_buf, static_cast<std::size_t>(line_len));
        }
    }

    flush_record();

    std::free(line_buf);
    if (fh != stdin) std::fclose(fh);
}

} // namespace tpoptoa
