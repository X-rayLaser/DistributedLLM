/*
MIT License

Copyright (c) 2023 Georgi Gerganov

Copyright (c) 2023 Evgenii Dolotov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "llama-util.h"
#include "llama.h"

#include <iostream>
#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>


template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw std::runtime_error(format("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b));
    }
    return ret;
}


static size_t llama_calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_type type) {
    size_t size = ggml_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_blck_size(type);
}


struct llama_hparams {
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    uint32_t n_mult  = 256;
    uint32_t n_head  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;

    bool operator!=(const llama_hparams & other) const {
        return static_cast<bool>(memcmp(this, &other, sizeof(llama_hparams)));
    }
};


struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};


struct llama_load_tensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    std::vector<uint32_t> ne;
    size_t file_off;
    size_t size;
    struct ggml_tensor * ggml_tensor = NULL;
    uint8_t * data;
};

struct llama_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<llama_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum llama_file_version {
    LLAMA_FILE_VERSION_GGML,
    LLAMA_FILE_VERSION_GGMF_V1, // added version field and scores in vocab
    LLAMA_FILE_VERSION_GGJT_V1, // added padding
    LLAMA_FILE_VERSION_GGJT_V2, // changed quantization format
    LLAMA_FILE_VERSION_GGJT_V3, // changed Q4 and Q8 quantization format
};


struct my_file_loader {
    llama_file file;
    llama_file_version file_version;
    llama_hparams hparams;
    llama_vocab vocab;

    my_file_loader(const char * fname, llama_load_tensors_map & tensors_map)
        : file(fname, "rb") {
        fprintf(stderr, "llama.cpp: loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(tensors_map);
    }
    void read_magic() {
        uint32_t magic = file.read_u32();

        if (magic == LLAMA_FILE_MAGIC_GGML) {
            file_version = LLAMA_FILE_VERSION_GGML;
            return;
        }

        uint32_t version = file.read_u32();

        switch (magic) {
            case LLAMA_FILE_MAGIC_GGMF:
                switch (version) {
                    case 1: file_version = LLAMA_FILE_VERSION_GGMF_V1; return;
                }
                break;
            case LLAMA_FILE_MAGIC_GGJT:
                switch (version) {
                    case 1: file_version = LLAMA_FILE_VERSION_GGJT_V1; return;
                    case 2: file_version = LLAMA_FILE_VERSION_GGJT_V2; return;
                    case 3: file_version = LLAMA_FILE_VERSION_GGJT_V3; return;
                }
        }

        throw std::runtime_error(format("unknown (magic, version) combination: %08x, %08x; is this really a GGML file?",
                     magic, version));
    }
    void read_hparams() {
        hparams.n_vocab = file.read_u32();
        hparams.n_embd = file.read_u32();
        hparams.n_mult = file.read_u32();
        hparams.n_head = file.read_u32();
        hparams.n_layer = file.read_u32();
        hparams.n_rot = file.read_u32();
        hparams.ftype = (enum llama_ftype) file.read_u32();
    }
    void read_vocab() {
        vocab.id_to_token.resize(hparams.n_vocab);

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {
            uint32_t len = file.read_u32();
            std::string word = file.read_string(len);

            float score = 0.0f;
            file.read_raw(&score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto & tok_score = vocab.id_to_token[i];
            tok_score.tok = std::move(word);
            tok_score.score = score;
        }
    }
    void read_tensor_metadata(llama_load_tensors_map & tensors_map) {
        while (file.tell() < file.size) {
            llama_load_tensor tensor;
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            tensor.type = (enum ggml_type) file.read_u32();
            tensor.ne.resize(n_dims);
            file.read_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * n_dims);
            std::string name = file.read_string(name_len);
            if (n_dims < 1 || n_dims > 2) {
                throw std::runtime_error(format("llama.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims));
            }
            switch (tensor.type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    break;
                default: {
                    throw std::runtime_error(format("unrecognized tensor type %u\n", tensor.type));
                }
            }

            // skip to the next multiple of 32 bytes
            file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);

            tensor.file_off = file.tell();
            tensor.name = name;
            tensor.size = llama_calc_tensor_size(tensor.ne, tensor.type);
            file.seek(tensor.size, SEEK_CUR);

            tensors_map.tensors.push_back(tensor);
            tensors_map.name_to_idx[name] = tensors_map.tensors.size() - 1;
        }
    }
};


struct my_file_saver {
    llama_file file;
    my_file_loader * any_file_loader;
    my_file_saver(const char * fname, my_file_loader * any_file_loader, enum llama_ftype new_ftype, uint32_t new_n_layer, uint32_t first_layer)
        : file(fname, "wb"), any_file_loader(any_file_loader) {
        fprintf(stderr, "llama.cpp: saving model to %s\n", fname);
        write_magic();
        write_hparams(new_ftype, new_n_layer, first_layer);
        write_vocab();
    }
    void write_magic() {
        file.write_u32(LLAMA_FILE_MAGIC);   // magic
        file.write_u32(LLAMA_FILE_VERSION); // version
    }
    void write_hparams(enum llama_ftype new_ftype, uint new_n_layer, uint first_layer) {
        const llama_hparams & hparams = any_file_loader->hparams;
        file.write_u32(hparams.n_vocab);
        file.write_u32(hparams.n_embd);
        file.write_u32(hparams.n_mult);
        file.write_u32(hparams.n_head);
        file.write_u32(new_n_layer);
        file.write_u32(hparams.n_rot);
        file.write_u32(first_layer);
        file.write_u32(new_ftype);
    }
    void write_vocab() {
        if (any_file_loader->file_version == LLAMA_FILE_VERSION_GGML) {
            fprintf(stderr, "llama.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
        }
        uint32_t n_vocab = any_file_loader->hparams.n_vocab;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const auto & token_score = any_file_loader->vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token_score.tok.size());
            file.write_raw(token_score.tok.data(), token_score.tok.size());
            file.write_raw(&token_score.score, sizeof(token_score.score));
        }
    }
    void write_tensor(llama_load_tensor & tensor, enum ggml_type new_type, const void * new_data, size_t new_size) {
        switch (new_type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                break;
            default: LLAMA_ASSERT(false);
        }
        file.write_u32((uint32_t) tensor.ne.size());
        file.write_u32((uint32_t) tensor.name.size());
        file.write_u32(new_type);
        file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
        file.write_raw(tensor.name.data(), tensor.name.size());
        file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
        LLAMA_ASSERT(new_size == llama_calc_tensor_size(tensor.ne, new_type));
        file.write_raw(new_data, new_size);
    }
};


bool starts_with(std::string name, std::string prefix) {
    auto res = std::mismatch(prefix.begin(), prefix.end(), name.begin());
    return (res.first == prefix.end());
}


bool is_block_layer(llama_load_tensor & tensor, int block_idx) {
    std::string prefix = "layers." + std::to_string(block_idx) + ".";

    auto res = std::mismatch(prefix.begin(), prefix.end(), tensor.name.begin());

    if (res.first == prefix.end()) {
        return true;
    } else {
        return false;
    }
}


bool should_include_layer(llama_load_tensor & tensor, int idx_start, int idx_stop) {
    for (int i = idx_start; i <= idx_stop; i++) {
        if (is_block_layer(tensor, i)) {
            return true;
        }
    }

    return false;
}


class LayerSelector {
    public:
        virtual bool should_include(llama_load_tensor & tensor) = 0;
};


class SelectExtraLayers : public LayerSelector {
    public:
        bool should_include(llama_load_tensor & tensor) override {
            std::string name = tensor.name;
            return (starts_with(name, "norm") || starts_with(name, "output") || starts_with(name, "tok_embeddings"));
        }
};


class SelectSlice : public LayerSelector {
    int m_idx_from;
    int m_idx_to;
    public:
        SelectSlice(int idx_from, int idx_to): m_idx_from(idx_from), m_idx_to(idx_to) {}
        bool should_include(llama_load_tensor & tensor) override {
            return should_include_layer(tensor, m_idx_from, m_idx_to);
        }
};


int main(int argc, char ** argv) {
    int idx_from;
    int idx_to;

    uint32_t new_n_layer;

    if (argc < 3) {
        std::cout << "Expected at least 2 arguments: command and model_path. Got " << argc - 1 << "\n";
        return 1;
    }

    std::string command = argv[1];
    std::string fname_inp = argv[2];
    size_t lastindex = fname_inp.find_last_of(".");
    std::string file_name = fname_inp.substr(0, lastindex);

    LayerSelector* selector;
    std::string fname_out;

    if (command == "extra_layers") {
        idx_from = -1;
        idx_to = -1;
        new_n_layer = 0;

        selector = new SelectExtraLayers;

        if (argc == 3) {
            fname_out = file_name + "_extra_layers.bin";
        } else {
            fname_out = argv[3];
        }
        std::cout << "Extracting extra layers from a model " << fname_inp << "\n";
    } else if (command == "slice") {
        if (argc == 5 || argc == 6) {
            idx_from = atoi(argv[3]);
            idx_to = atoi(argv[4]);
            new_n_layer = idx_to - idx_from + 1;
            selector = new SelectSlice(idx_from, idx_to);
            
            if (argc == 5) {
                fname_out = file_name + "_layers_" + std::to_string(idx_from) + "_" + std::to_string(idx_to) + ".bin";
            } else {
                fname_out = argv[5];
            }
            std::cout << "Taking a slice of a model " << fname_inp << " from " << idx_from << " to " << idx_to << "\n";
        } else {
            std::cout << "Command slice expects arguments model_path, idx_from, idx_to. Got " << argc - 1 << "\n";
            return 1;
        }
    } else {
        std::cout << "Invalid command argument. Expected one of [extra_layers, slice]. Got " << command << "\n";
        return 1;
    }

    std::cout << "Saving it to " << fname_out << "\n";

    llama_load_tensors_map tensors_map;
    std::unique_ptr<my_file_loader> file_loader(new my_file_loader(fname_inp.c_str(), tensors_map));

    std::unique_ptr<my_file_saver> file_saver(new my_file_saver(fname_out.c_str(), file_loader.get(), file_loader->hparams.ftype, new_n_layer, idx_from));

    std::cout << "Ready to start slicing" << "\n";
    for (llama_load_tensor & tensor : tensors_map.tensors) {
        if (selector->should_include(tensor)) {
            std::string s = "";
            for (auto d : tensor.ne) {
                s = s + " dim " +  std::to_string(d);
            }
            
            std::cout << "Saving tensor " << tensor.name << "\n";

            llama_buffer read_data;
            read_data.resize(tensor.size);
            tensor.data = read_data.addr;
            llama_file & file = file_loader->file;
            file.seek(tensor.file_off, SEEK_SET);
            file.read_raw(tensor.data, tensor.size);

            file_saver->write_tensor(tensor, tensor.type, tensor.data, tensor.size);
            std::cout << "Succeeded" << "\n";
        }
    }

    std::cout << "Completed\n";
    return 0;
}