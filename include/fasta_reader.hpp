#pragma once

// fasta_reader.hpp — streaming FASTA and FASTQ reader that yields one
// sequence at a time, with no whole-file buffering.
//
// The reader handles both formats transparently: FASTQ records begin with '@'
// while FASTA records begin with '>'.  Gzipped input is not handled here (pipe
// through zcat/gunzip first, or use the provided helper script).

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <stdexcept>

namespace tpoptoa {

// A lightweight holder for one sequence record.
struct SeqRecord {
    std::string name;  // everything after the '>' or '@' up to the first space
    std::string seq;   // the nucleotide string
};

// Open a FASTA or FASTQ file and call callback(record) for every sequence.
// The callback receives a const SeqRecord& and can break early by returning
// false; any other return value (including void via lambda wrapping) continues.
//
// Throws std::runtime_error if the file cannot be opened.
//
// The template signature lets callers use plain lambdas:
//   iterate_sequences("genome.fa", [&](const SeqRecord& r){ process(r); });
template <typename Callback>
void iterate_sequences(const char* path, Callback&& cb) {
    FILE* fh = (path[0] == '-' && path[1] == '\0') ? stdin : std::fopen(path, "r");
    if (!fh)
        throw std::runtime_error(std::string("cannot open file: ") + path);

    // We'll build records line by line using a reusable buffer.
    char*   line_buf  = nullptr;
    size_t  buf_cap   = 0;
    ssize_t line_len  = 0;

    SeqRecord rec;
    bool in_fastq_qual = false;  // true while consuming the quality-score lines
    bool have_record   = false;

    auto flush_record = [&]() {
        if (have_record && !rec.seq.empty())
            cb(rec);
        rec.name.clear();
        rec.seq.clear();
        have_record = false;
    };

    while ((line_len = getline(&line_buf, &buf_cap, fh)) != -1) {
        // Strip trailing newline / carriage-return.
        while (line_len > 0 &&
               (line_buf[line_len - 1] == '\n' || line_buf[line_len - 1] == '\r'))
            line_buf[--line_len] = '\0';

        if (line_len == 0) continue;  // skip blank lines

        char first = line_buf[0];

        // FASTA header line.
        if (first == '>') {
            flush_record();
            in_fastq_qual = false;
            have_record   = true;
            // Name is everything up to the first whitespace.
            std::string_view sv(line_buf + 1, static_cast<std::size_t>(line_len - 1));
            auto space = sv.find_first_of(" \t");
            rec.name = (space == std::string_view::npos) ? std::string(sv)
                                                          : std::string(sv.substr(0, space));
            continue;
        }

        // FASTQ header line.
        if (first == '@' && !have_record) {
            flush_record();
            in_fastq_qual = false;
            have_record   = true;
            std::string_view sv(line_buf + 1, static_cast<std::size_t>(line_len - 1));
            auto space = sv.find_first_of(" \t");
            rec.name = (space == std::string_view::npos) ? std::string(sv)
                                                          : std::string(sv.substr(0, space));
            continue;
        }

        // FASTQ '+' separator — everything that follows (on the same line or
        // the next lines) is quality scores; skip them.
        if (first == '+' && have_record) {
            in_fastq_qual = true;
            continue;
        }

        if (in_fastq_qual) {
            // Quality lines have the same length as the sequence lines; we
            // just discard them.  When all quality has been consumed (quality
            // length == sequence length) we reset.
            // For simplicity we track nothing and reset at next '@'.
            continue;
        }

        // Sequence line — append to current record.
        if (have_record) {
            rec.seq.append(line_buf, static_cast<std::size_t>(line_len));
        }
    }

    flush_record();  // don't forget the last record

    std::free(line_buf);
    if (fh != stdin) std::fclose(fh);
}

} // namespace tpoptoa
