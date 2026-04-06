#pragma once

#include "llama.h"
#include "server-task.h"

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <map>
#include <vector>

// Simple FNV-1a hash (no external deps)
static inline uint64_t prefix_hash_fnv1a(const void * data, size_t len, uint64_t seed = 0xcbf29ce484222325ULL) {
    const uint8_t * p = (const uint8_t *)data;
    uint64_t h = seed;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct prefix_block {
    uint64_t      hash;            // hash of (tokens + pos_start)
    llama_pos     pos_start;       // block-aligned position
    int32_t       n_tokens;        // tokens in this block (block_size except possibly last)
    llama_seq_id  template_seq;    // template seq_id owning KV cells (TEMPLATE_SEQ_BASE..TEMPLATE_SEQ_MAX)
    int           ref_count;       // active sequences referencing this block
    int64_t       t_last_used;     // monotonic timestamp for LRU

    // recurrent state checkpoint at the END of this block
    std::vector<uint8_t> rs_data;

    // stored tokens for hash collision verification
    std::vector<llama_token> tokens;

    size_t memory_usage() const {
        return rs_data.size() + tokens.size() * sizeof(llama_token) + sizeof(*this);
    }
};

struct prefix_match_result {
    int n_matched_blocks;
    int n_cached_tokens;           // sum of matched blocks' n_tokens
    std::vector<prefix_block *> matched_blocks;
    bool has_rs_checkpoint;        // whether the last matched block has RS data
};

struct server_prefix_block_cache {
    static constexpr int TEMPLATE_SEQ_BASE = 64;
    static constexpr int TEMPLATE_SEQ_MAX  = 255; // LLAMA_MAX_SEQ - 1

    int32_t block_size       = 4096;
    int32_t max_blocks       = 192;  // max cached blocks (192 * 4096 = 786k tokens)
    bool    enabled          = true;
    size_t  max_rs_bytes     = 0;    // 0 = no limit on RS checkpoint memory

    // hash -> block
    std::unordered_map<uint64_t, prefix_block> blocks;

    // LRU ordering: timestamp -> hash (ascending = oldest first)
    std::map<int64_t, uint64_t> lru_order;

    // template seq_id allocation
    std::vector<bool> template_seq_used;

    int64_t timestamp_counter = 0;
    size_t  total_rs_bytes    = 0;

    void init() {
        template_seq_used.assign(TEMPLATE_SEQ_MAX - TEMPLATE_SEQ_BASE + 1, false);
        blocks.clear();
        lru_order.clear();
        timestamp_counter = 0;
        total_rs_bytes = 0;
    }

    // compute hash for a block of tokens at a given position
    uint64_t compute_hash(const llama_token * tokens, int n, llama_pos pos_start) const {
        uint64_t h = prefix_hash_fnv1a(&pos_start, sizeof(pos_start));
        return prefix_hash_fnv1a(tokens, n * sizeof(llama_token), h);
    }

    // find longest prefix of contiguous matching blocks starting from position 0
    prefix_match_result find_matches(const server_tokens & tokens) {
        prefix_match_result result = {};
        result.has_rs_checkpoint = false;

        if (!enabled || tokens.empty()) {
            return result;
        }

        const int n_total = (int)tokens.size();
        int pos = 0;

        while (pos + block_size <= n_total) {
            const llama_token * block_tokens = &tokens[pos];
            uint64_t h = compute_hash(block_tokens, block_size, (llama_pos)pos);

            auto it = blocks.find(h);
            if (it == blocks.end()) {
                break;
            }

            prefix_block & block = it->second;

            // verify tokens match (hash collision check)
            if (block.n_tokens != block_size ||
                block.pos_start != (llama_pos)pos ||
                memcmp(block.tokens.data(), block_tokens, block_size * sizeof(llama_token)) != 0) {
                break;
            }

            result.matched_blocks.push_back(&block);
            result.n_cached_tokens += block.n_tokens;
            result.n_matched_blocks++;
            result.has_rs_checkpoint = !block.rs_data.empty();

            pos += block_size;
        }

        return result;
    }

    // allocate a template sequence ID
    llama_seq_id alloc_template_seq() {
        for (int i = 0; i < (int)template_seq_used.size(); i++) {
            if (!template_seq_used[i]) {
                template_seq_used[i] = true;
                return (llama_seq_id)(TEMPLATE_SEQ_BASE + i);
            }
        }
        return -1; // exhausted
    }

    void free_template_seq(llama_seq_id seq_id) {
        int idx = seq_id - TEMPLATE_SEQ_BASE;
        if (idx >= 0 && idx < (int)template_seq_used.size()) {
            template_seq_used[idx] = false;
        }
    }

    // attach matched blocks to a slot's sequence (increment ref_count, update LRU)
    // caller must do the actual seq_cp calls
    void attach(const std::vector<prefix_block *> & matched_blocks) {
        for (auto * block : matched_blocks) {
            block->ref_count++;
            update_lru(block);
        }
    }

    // detach a slot from all its blocks (decrement ref_count)
    void detach(const std::vector<prefix_block *> & attached_blocks) {
        for (auto * block : attached_blocks) {
            if (block->ref_count > 0) {
                block->ref_count--;
            }
        }
    }

    // register a newly completed block after prompt evaluation
    // returns nullptr if max_blocks reached and eviction fails, or if template seq exhausted
    prefix_block * register_block(
            const llama_token * tokens,
            int                 n_tokens,
            llama_pos           pos_start,
            llama_seq_id        template_seq) {

        uint64_t h = compute_hash(tokens, n_tokens, pos_start);

        // already exists?
        auto it = blocks.find(h);
        if (it != blocks.end()) {
            // verify it's the same block
            if (it->second.n_tokens == n_tokens &&
                it->second.pos_start == pos_start &&
                memcmp(it->second.tokens.data(), tokens, n_tokens * sizeof(llama_token)) == 0) {
                update_lru(&it->second);
                return &it->second;
            }
            // hash collision with different content — skip registration
            return nullptr;
        }

        prefix_block block;
        block.hash          = h;
        block.pos_start     = pos_start;
        block.n_tokens      = n_tokens;
        block.template_seq  = template_seq;
        block.ref_count     = 0;
        block.t_last_used   = ++timestamp_counter;
        block.tokens.assign(tokens, tokens + n_tokens);

        auto [ins_it, ok] = blocks.emplace(h, std::move(block));
        if (!ok) {
            return nullptr;
        }

        lru_order[ins_it->second.t_last_used] = h;
        return &ins_it->second;
    }

    // store recurrent state checkpoint for a block
    void store_rs_checkpoint(prefix_block * block, const std::vector<uint8_t> & data) {
        total_rs_bytes -= block->rs_data.size();
        block->rs_data = data;
        total_rs_bytes += block->rs_data.size();
    }

    // evict unreferenced blocks (oldest first) to free space
    // returns number of blocks evicted
    // caller must call mem_attn->seq_rm() for each evicted block's template_seq
    struct eviction_entry {
        llama_seq_id template_seq;
        llama_pos    pos_start;
        int32_t      n_tokens;
    };

    std::vector<eviction_entry> evict_lru(int n_needed) {
        std::vector<eviction_entry> evicted;

        auto it = lru_order.begin();
        while (it != lru_order.end() && (int)evicted.size() < n_needed) {
            auto block_it = blocks.find(it->second);
            if (block_it == blocks.end()) {
                it = lru_order.erase(it);
                continue;
            }

            prefix_block & block = block_it->second;
            if (block.ref_count > 0) {
                ++it;
                continue;
            }

            eviction_entry entry;
            entry.template_seq = block.template_seq;
            entry.pos_start    = block.pos_start;
            entry.n_tokens     = block.n_tokens;
            evicted.push_back(entry);

            total_rs_bytes -= block.rs_data.size();
            free_template_seq(block.template_seq);
            blocks.erase(block_it);
            it = lru_order.erase(it);
        }

        return evicted;
    }

    // evict RS data only (keep KV blocks alive) when RS memory is over budget
    void evict_rs_over_budget() {
        if (max_rs_bytes == 0 || total_rs_bytes <= max_rs_bytes) {
            return;
        }

        auto it = lru_order.begin();
        while (it != lru_order.end() && total_rs_bytes > max_rs_bytes) {
            auto block_it = blocks.find(it->second);
            if (block_it == blocks.end()) {
                ++it;
                continue;
            }

            prefix_block & block = block_it->second;
            if (block.ref_count > 0 || block.rs_data.empty()) {
                ++it;
                continue;
            }

            total_rs_bytes -= block.rs_data.size();
            block.rs_data.clear();
            block.rs_data.shrink_to_fit();
            ++it;
        }
    }

    int num_blocks() const {
        return (int)blocks.size();
    }

    int num_cached_tokens() const {
        int total = 0;
        for (const auto & [h, block] : blocks) {
            total += block.n_tokens;
        }
        return total;
    }

private:
    void update_lru(prefix_block * block) {
        // remove old LRU entry
        lru_order.erase(block->t_last_used);
        // add new one
        block->t_last_used = ++timestamp_counter;
        lru_order[block->t_last_used] = block->hash;
    }
};
