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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define LLAMA_MAX_SCRATCH_BUFFERS 16


#include "common.h"
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

static llama_context ** g_ctx;


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
    uint32_t first_layer = 0; // first layer number (generally, it will not be 0)
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
        //fprintf(stderr, "llama.cpp: loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(tensors_map);
        //std::cout << "loaded " << fname << "\n";
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
        hparams.first_layer = file.read_u32();
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


bool starts_with(std::string str, std::string prefix) {
    auto res = std::mismatch(prefix.begin(), prefix.end(), str.begin());

    if (res.first == prefix.end()) {
        return true;
    } else {
        return false;
    }
}


bool is_block_layer(llama_load_tensor & tensor, int block_idx) {
    std::string prefix = "layers." + std::to_string(block_idx) + ".";
    return starts_with(tensor.name, prefix);
}


bool should_include_layer(llama_load_tensor & tensor, int idx_start, int idx_stop) {
    for (int i = idx_start; i <= idx_stop; i++) {
        if (is_block_layer(tensor, i)) {
            return true;
        }
    }

    return false;
}


static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}


struct llama_kv_cache {
    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_ctx_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_data(k);
        ggml_cuda_free_data(v);
#endif // GGML_USE_CUBLAS
    }
};


enum e_model {
    MODEL_UNKNOWN,
    MODEL_3B,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
};


struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};



struct llama_model {
    e_model type = MODEL_UNKNOWN;

    llama_hparams hparams;

    std::vector<llama_layer> layers;
    int n_gpu_layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_ctx_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_mlock mlock_buf;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    llama_vocab vocab;

    ~llama_model() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};


struct llama_context {
    llama_context(const llama_model & model, const llama_vocab & vocab) : model(model), vocab(vocab), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const llama_model & model;
    const llama_vocab & vocab;

    bool model_owner = false;

    int64_t t_load_us;
    int64_t t_start_us;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    size_t mem_per_token = 0;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    llama_ctx_buffer buf_compute;
    llama_ctx_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};


typedef void (*offload_func_t)(struct ggml_tensor * tensor);

void llama_nop(struct ggml_tensor * tensor);


static bool llama_eval_internal(
         llama_context & lctx,
           const float * embd,
                   int   n_tokens,
                   int   n_past,
                   int   n_threads,
            const char * cgraph_fname) {

    //std::cout << "enter llama_eval_internal\n";
    const int64_t t_start_us = ggml_time_us();

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    LLAMA_ASSERT(!!kv_self.ctx);

    const int n_embd       = hparams.n_embd;
    const int n_layer      = hparams.n_layer;
    const int n_ctx        = hparams.n_ctx;
    const int n_head       = hparams.n_head;
    const int n_vocab      = hparams.n_vocab;
    const int n_rot        = hparams.n_embd/hparams.n_head;
    const int n_gpu_layers = model.n_gpu_layers;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.addr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph gf = {};

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    n_threads = N >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : n_threads;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);
    memcpy(inpL->data, embd, N * n_embd * sizeof(float));
    
    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    //
    // with the low VRAM option VRAM scratch is disabled in llama_load_model_internal
    // in that case ggml_cuda_assign_buffers has no effect
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

    for (int il = 0; il < n_layer; ++il) {

        //std::cout << "adding layer " << il << "to the graph\n";
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

        struct ggml_tensor * inpSA = inpL;

        lctx.use_buf(ctx0, 0);


        //std::cout << "about to add normalization\n";

        // norm
        {

            //std::cout << "about to ggml_rms_norm\n";
            cur = ggml_rms_norm(ctx0, inpL);
            
            //std::cout << "done ggml_rms_norm\n";
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attention_norm(broadcasted)

            auto x = model.layers[il].attention_norm;
            //std::cout << "about to mul\n";

            cur = ggml_mul(ctx0, cur, model.layers[il].attention_norm);

            //std::cout << "done mul\n";
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }


        //std::cout << "added normalization\n";

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd/n_head, n_head, N), n_past, n_rot, 0, 0);
            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            struct ggml_tensor * Qcur = ggml_rope_inplace(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd/n_head, n_head, N), n_past, n_rot, 0, 0);
            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");



            //std::cout << "computed Q and K and RoPE them\n";
            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd, N));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd + n_past*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }


            //std::cout << "stored key and value to memory\n";

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scale = ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head));
            ggml_set_name(KQ_scale, "1/sqrt(n_embd/n_head)");

            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_element_size(kv_self.v),
                        n_ctx*ggml_element_size(kv_self.v)*n_embd/n_head,
                        il*n_ctx*ggml_element_size(kv_self.v)*n_embd);
            offload_func_v(V);
            ggml_set_name(V, "V");

#if 1
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_tensor * V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd/n_head, n_head));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        lctx.use_buf(ctx0, 1);

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        //std::cout << "About to do feed-forward pass\n";

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }


    //std::cout << "Include all layers in a graph\n";

    //after all 32 layers
    //lctx.use_buf(ctx0, 0);
    // was used right after lm_head
    lctx.use_buf(ctx0, -1);

    // run the computation
    ggml_build_forward_expand(&gf, cur);

    ggml_graph_compute_helper(lctx.work_buffer, &gf, n_threads);

    // update kv token count
    lctx.kv_self.n = n_past + N;

    struct ggml_tensor * res = gf.nodes[gf.n_nodes - 1];

    if (cgraph_fname) {
        ggml_graph_export(&gf, cgraph_fname);
    }


    //std::cout << "About to extract result\n";

    auto & embedding_out = lctx.embedding;

    //embedding_out.resize(n_embd);
    //memcpy(embedding_out.data(), (float *) ggml_get_data(res) + (n_embd*(N - 1)), sizeof(float)*n_embd);

    embedding_out.resize(n_embd * N);
    memcpy(embedding_out.data(), (float *) ggml_get_data(res), sizeof(float)*n_embd * N);
    //std::cout << "Completed extracting embeddings\n";

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

    ggml_free(ctx0);
    
    return true;
}


static std::string llama_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}


static const char *llama_file_version_name(llama_file_version version) {
    switch (version) {
        case LLAMA_FILE_VERSION_GGML: return "'ggml' (old version with low tokenizer quality and no mmap support)";
        case LLAMA_FILE_VERSION_GGMF_V1: return "ggmf v1 (old version with no mmap support)";
        case LLAMA_FILE_VERSION_GGJT_V1: return "ggjt v1 (pre #1405)";
        case LLAMA_FILE_VERSION_GGJT_V2: return "ggjt v2 (pre #1508)";
        case LLAMA_FILE_VERSION_GGJT_V3: return "ggjt v3 (latest)";
    }

    return "unknown";
}


static const char *llama_ftype_name(enum llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:  return "mostly F16";
        case LLAMA_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case LLAMA_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";
        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K: return "mostly Q2_K";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S: return "mostly Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M: return "mostly Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return "mostly Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S: return "mostly Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "mostly Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S: return "mostly Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return "mostly Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K: return "mostly Q6_K";
        default:                      return "unknown, may not work";
    }
}


static const char *llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_3B: return "3B";
        case MODEL_7B: return "7B";
        case MODEL_13B: return "13B";
        case MODEL_30B: return "30B";
        case MODEL_65B: return "65B";
        default: LLAMA_ASSERT(false);
    }
}

static const size_t MB = 1024*1024;

static const std::map<e_model, size_t> & MEM_REQ_SCRATCH0()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,    256ull * MB },
        { MODEL_7B,    512ull * MB },
        { MODEL_13B,   512ull * MB },
        { MODEL_30B,   512ull * MB },
        { MODEL_65B,  1024ull * MB },
    };
    return k_sizes;
}

static const std::map<e_model, size_t> & MEM_REQ_SCRATCH1()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,    256ull * MB },
        { MODEL_7B,    512ull * MB },
        { MODEL_13B,   512ull * MB },
        { MODEL_30B,   512ull * MB },
        { MODEL_65B,  1024ull * MB },
    };
    return k_sizes;
}

// 2*n_embd*n_ctx*n_layer*sizeof(float16)
static const std::map<e_model, size_t> & MEM_REQ_KV_SELF()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,    682ull * MB },
        { MODEL_7B,   1026ull * MB },
        { MODEL_13B,  1608ull * MB },
        { MODEL_30B,  3124ull * MB },
        { MODEL_65B,  5120ull * MB },
    };
    return k_sizes;
}

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t> & MEM_REQ_EVAL()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   512ull * MB },
        { MODEL_7B,   768ull * MB },
        { MODEL_13B, 1024ull * MB },
        { MODEL_30B, 1280ull * MB },
        { MODEL_65B, 1536ull * MB },
    };
    return k_sizes;
}


struct llama_model_loader {
    std::unique_ptr<my_file_loader> file_loader;
    llama_load_tensors_map tensors_map;
    bool use_mmap;
    size_t num_ggml_tensors_created = 0;
    struct ggml_context * ggml_ctx = NULL;
    std::unique_ptr<llama_mmap> mapping;

    llama_model_loader(const std::string & fname_base, bool use_mmap) {
        file_loader = std::unique_ptr<my_file_loader>(new my_file_loader(fname_base.c_str(), tensors_map));
        if (!llama_mmap::SUPPORTED) {
            use_mmap = false;
        }
        this->use_mmap = use_mmap;
    }

    void calc_sizes(size_t * ctx_size_p, size_t * mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const llama_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
        }
    }

    struct ggml_tensor * get_tensor(const std::string & name, const std::vector<uint32_t> & ne, ggml_backend backend) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw std::runtime_error(std::runtime_error(format("llama.cpp: tensor '%s' is missing from model", name.c_str())));
        }
        llama_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw std::runtime_error(format("llama.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                         name.c_str(), llama_format_tensor_shape(ne).c_str(), llama_format_tensor_shape(lt.ne).c_str()));
        }

        //std::cout << "about to get_tensor_for " << name << "\n";
        return get_tensor_for(lt, backend);
    }

    struct ggml_tensor * get_tensor_for(llama_load_tensor & lt, ggml_backend backend) {
        struct ggml_tensor * tensor;
        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ggml_ctx, true);
        }

        if (lt.ne.size() == 2) {
            tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            LLAMA_ASSERT(lt.ne.size() == 1);
            tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne.at(0));
        }

        ggml_set_name(tensor, lt.name.c_str());
        LLAMA_ASSERT(lt.ggml_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor

        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ggml_ctx, use_mmap);
        }
        tensor->backend = backend;
        lt.ggml_tensor = tensor;
        num_ggml_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() const {
        if (num_ggml_tensors_created != tensors_map.tensors.size()) {
            throw std::runtime_error(std::string("llama.cpp: file contained more tensors than expected"));
        }
    }

    void load_all_data(llama_progress_callback progress_callback, void *  progress_callback_user_data, llama_mlock * lmlock) {
        size_t data_size = 0;
        size_t prefetch_size = 0;
        size_t lock_size = 0;
        for (const llama_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
            if (lt.ggml_tensor->backend == GGML_BACKEND_CPU) {
                prefetch_size += lt.size;
            }
        }

        if (use_mmap) {
            mapping.reset(new llama_mmap(&file_loader->file, prefetch_size, ggml_is_numa()));
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (llama_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            LLAMA_ASSERT(lt.ggml_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_tensor->data;

            // allocate temp buffer if not using mmap
            if (!use_mmap && lt.data == NULL) {
                GGML_ASSERT(lt.ggml_tensor->backend != GGML_BACKEND_CPU);
                lt.data = (uint8_t*)malloc(ggml_nbytes(lt.ggml_tensor));
            }

            load_data_for(lt);

            switch(lt.ggml_tensor->backend) {
                case GGML_BACKEND_CPU:
                    lt.ggml_tensor->data = lt.data;
                    if (use_mmap && lmlock) {
                        lock_size += lt.size;
                        lmlock->grow_to(lock_size);
                    }
                    break;
#if defined(GGML_USE_CUBLAS)
                case GGML_BACKEND_GPU:
                case GGML_BACKEND_GPU_SPLIT:
                    ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);
                    if (!use_mmap) {
                        free(lt.data);
                    }
                    break;
#elif defined(GGML_USE_CLBLAST)
                case GGML_BACKEND_GPU:
                    ggml_cl_transform_tensor(lt.data, lt.ggml_tensor);
                    if (!use_mmap) {
                        free(lt.data);
                    }
                    break;
#endif
                default:
                    continue;
            }

            done_size += lt.size;
        }
    }

    void load_data_for(llama_load_tensor & lt) {
        if (use_mmap) {
            lt.data = (uint8_t *) mapping->addr + lt.file_off;
        } else {
            llama_file & file = file_loader->file;
            file.seek(lt.file_off, SEEK_SET);
            file.read_raw(lt.data, lt.size);
        }

        if (0) {
            print_checksum(lt);
        }
    }

    static void print_checksum(llama_load_tensor & lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
                llama_format_tensor_shape(lt.ne).c_str(), lt.size);
    }

};


static bool kv_cache_init(
        const struct llama_hparams & hparams,
             struct llama_kv_cache & cache,
                         ggml_type   wtype,
                               int   n_ctx,
                               int   n_gpu_layers) {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);
    cache.n = 0;

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.addr;
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    ggml_set_name(cache.k, "cache_k");
    ggml_set_name(cache.v, "cache_v");

    (void) n_gpu_layers;
#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer + 1) {
        ggml_cuda_assign_buffers_no_scratch(cache.v);
    }
    if (n_gpu_layers > n_layer + 2) {
        ggml_cuda_assign_buffers_no_scratch(cache.k);
    }
#endif // GGML_USE_CUBLAS

    return true;
}



struct llama_context * my_llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {

    if (!model) {
        return nullptr;
    }

    llama_context * ctx = new llama_context(*model, model->vocab);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                fprintf(stderr, ".");
                fflush(stderr);
                if (percentage >= 100) {
                    fprintf(stderr, "\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, ctx->model.hparams.n_ctx, params.n_gpu_layers)) {
            fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->kv_self.k) + ggml_nbytes(ctx->kv_self.v);
            fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        std::cout << "reserved logits\n";
        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        std::cout << "resized ctx->embedding\n";

        ctx->buf_compute.resize(MEM_REQ_EVAL().at(ctx->model.type));

        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0().at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1().at(ctx->model.type));
    }
    std::cout << "prepared context\n";
    return ctx;
}


static void llama_model_load_internal(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {

    model.t_start_us = ggml_time_us();

    std::unique_ptr<llama_model_loader> ml(new llama_model_loader(fname, use_mmap));

    vocab = std::move(ml->file_loader->vocab);
    model.hparams = ml->file_loader->hparams;
    model.n_gpu_layers = n_gpu_layers;

    uint32_t first_layer = model.hparams.first_layer;
    llama_file_version file_version = ml->file_loader->file_version;
    auto & hparams = model.hparams;

    {
        switch (hparams.n_layer) {
            case 26: model.type = e_model::MODEL_3B; break;
            case 32: model.type = e_model::MODEL_7B; break;
            case 40: model.type = e_model::MODEL_13B; break;
            case 60: model.type = e_model::MODEL_30B; break;
            case 80: model.type = e_model::MODEL_65B; break;
            default:
                {
                    if (hparams.n_layer < 32) {
                        model.type = e_model::MODEL_7B;
                    }
                } break;
        }

        hparams.n_ctx = n_ctx;
    }

    const uint32_t n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

    {
        fprintf(stderr, "%s: format     = %s\n",  __func__, llama_file_version_name(file_version));
        fprintf(stderr, "%s: n_vocab    = %u\n",  __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_ctx      = %u\n",  __func__, hparams.n_ctx);
        fprintf(stderr, "%s: n_embd     = %u\n",  __func__, hparams.n_embd);
        fprintf(stderr, "%s: n_mult     = %u\n",  __func__, hparams.n_mult);
        fprintf(stderr, "%s: n_head     = %u\n",  __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer    = %u\n",  __func__, hparams.n_layer);
        fprintf(stderr, "%s: n_rot      = %u\n",  __func__, hparams.n_rot);
        fprintf(stderr, "%s: ftype      = %u (%s)\n", __func__, hparams.ftype, llama_ftype_name(hparams.ftype));
        fprintf(stderr, "%s: n_ff       = %u\n",  __func__, n_ff);
        fprintf(stderr, "%s: model size = %s\n",  __func__, llama_model_type_name(model.type));
    }

    if (file_version < LLAMA_FILE_VERSION_GGJT_V2) {
        if (hparams.ftype != LLAMA_FTYPE_ALL_F32     &&
            hparams.ftype != LLAMA_FTYPE_MOSTLY_F16  &&
            hparams.ftype != LLAMA_FTYPE_MOSTLY_Q8_0) {
            throw std::runtime_error(format("this format is no longer supported (see https://github.com/ggerganov/llama.cpp/pull/1405)"));
        }
    }

    if (file_version < LLAMA_FILE_VERSION_GGJT_V3) {
        if (hparams.ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ||
            hparams.ftype == LLAMA_FTYPE_MOSTLY_Q4_1 ||
            hparams.ftype == LLAMA_FTYPE_MOSTLY_Q8_0) {
            throw std::runtime_error(format("this format is no longer supported (see https://github.com/ggerganov/llama.cpp/pull/1508)"));
        }
    }

    if (vocab_only) {
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    size_t mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    fprintf(stderr, "%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init(model.buf.addr);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.addr,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw std::runtime_error(format("ggml_init() failed"));
        }
    }

    (void) main_gpu;

#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU

    // prepare memory for the weights
    size_t vram_weights = 0;
    size_t vram_scratch = 0;
    {
        const uint32_t n_embd  = hparams.n_embd;
        const uint32_t n_layer = hparams.n_layer;
        const uint32_t n_vocab = hparams.n_vocab;

        ml->ggml_ctx = ctx;

        //model.tok_embeddings = ml->get_tensor("tok_embeddings.weight", {n_embd, n_vocab}, GGML_BACKEND_CPU);

        const int i_gpu_start = n_layer - n_gpu_layers;

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            std::cout << "loading layer " << i << "\n";
            const ggml_backend backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
            const ggml_backend backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

            auto & layer = model.layers[i];

            std::string layers_i = "layers." + std::to_string(i + first_layer);

            layer.attention_norm = ml->get_tensor(layers_i + ".attention_norm.weight", {n_embd}, backend);

            layer.wq = ml->get_tensor(layers_i + ".attention.wq.weight", {n_embd, n_embd}, backend_split);
            layer.wk = ml->get_tensor(layers_i + ".attention.wk.weight", {n_embd, n_embd}, backend_split);
            layer.wv = ml->get_tensor(layers_i + ".attention.wv.weight", {n_embd, n_embd}, backend_split);
            layer.wo = ml->get_tensor(layers_i + ".attention.wo.weight", {n_embd, n_embd}, backend_split);

            layer.ffn_norm = ml->get_tensor(layers_i + ".ffn_norm.weight", {n_embd}, backend);

            layer.w1 = ml->get_tensor(layers_i + ".feed_forward.w1.weight", {n_embd,   n_ff},   backend_split);
            layer.w2 = ml->get_tensor(layers_i + ".feed_forward.w2.weight", {  n_ff,   n_embd}, backend_split);
            layer.w3 = ml->get_tensor(layers_i + ".feed_forward.w3.weight", {n_embd,   n_ff},   backend_split);

            if (backend == GGML_BACKEND_GPU) {
                vram_weights +=
                    ggml_nbytes(layer.attention_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk)             +
                    ggml_nbytes(layer.wv)             + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                    ggml_nbytes(layer.w1)             + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
            }

            std::cout << "done loading tensor " << i << "\n";
        }

        std::cout << "load all tensors" << "\n";
    }

    std::cout << "about to call done" << "\n";
    ml->done_getting_tensors();
    std::cout << "done getting tensors, model.type" << model.type << "\n";

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
            ctx_size +
            mmapped_size - vram_weights + // weights in VRAM not in memory
            MEM_REQ_SCRATCH0().at(model.type) +
            MEM_REQ_SCRATCH1().at(model.type) +
            MEM_REQ_EVAL().at    (model.type);

        // this is the memory required by one llama_state
        const size_t mem_required_state =
            scale*MEM_REQ_KV_SELF().at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);

        (void) vram_scratch;
        (void) n_batch;

        (void) n_gpu_layers;
    }

    // populate `tensors_by_name`
    for (llama_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
    }

    (void) tensor_split;

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    if (progress_callback) {
        progress_callback(1.0f, progress_callback_user_data);
    }

    model.mapping = std::move(ml->mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_time_us() - model.t_start_us;
    std::cout << "internally loaded model" << "\n";
}


static bool my_llama_model_load(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gpu_layers,
        int main_gpu,
        float * tensor_split,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_model_load_internal(fname, model, vocab, n_ctx, n_batch, n_gpu_layers, main_gpu, tensor_split, low_vram, memory_type,
                                  use_mmap, use_mlock, vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::exception & err) {
        fprintf(stderr, "error loading model: %s\n", err.what());
        return false;
    }
}



struct llama_model * my_llama_load_model_from_file(
                             const char * path_model,
            struct llama_context_params   params) {
    ggml_time_init();

    llama_model * model = new llama_model;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!my_llama_model_load(path_model, *model, model->vocab, params.n_ctx, params.n_batch, params.n_gpu_layers,
                params.main_gpu, params.tensor_split, params.low_vram, memory_type, params.use_mmap, params.use_mlock,
                params.vocab_only, params.progress_callback, params.progress_callback_user_data)) {
        delete model;
        fprintf(stderr, "%s: failed to load model\n", __func__);
        return nullptr;
    }

    return model;
}


std::tuple<struct llama_model *, struct llama_context *> my_llama_init_from_gpt_params(const gpt_params & params) {
    auto lparams = llama_context_params_from_gpt_params(params);

    llama_model * model  = my_llama_load_model_from_file(params.model.c_str(), lparams);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return std::make_tuple(nullptr, nullptr);
    }

    llama_context * lctx = my_llama_new_context_with_model(model, lparams);
    if (lctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return std::make_tuple(nullptr, nullptr);
    }

    return std::make_tuple(model, lctx);
}


class TransformerSlice {
    std::string slice_path;
    gpt_params params;
    int n_past;
    int n_threads;

    llama_model * model;
    llama_context * ctx;
    public:
        TransformerSlice(std::string slice_path, gpt_params params, int n_threads) {
            this->slice_path = slice_path;
            this->params = params;
            this->n_threads = n_threads;

            this->params.model = slice_path;

            n_past = 0;
            //g_ctx = &ctx;
            std::tie(model, ctx) = my_llama_init_from_gpt_params(this->params);
            if (model == NULL) {
                fprintf(stderr, "%s: error: unable to load model\n", __func__);
            }
        }

        void clear_context() {
            n_past = 0;
            delete ctx;
            ctx = NULL;
            auto lparams = llama_context_params_from_gpt_params(params);
            ctx = my_llama_new_context_with_model(model, lparams);
            if (ctx == NULL) {
                fprintf(stderr, "%s: error: failed to recreate context for model '%s'\n", __func__, params.model.c_str());
            }
        }

        int forward(const std::vector<float>& embeddings, std::vector<float>& res) {
            int n_embd = get_n_embd();

            int N = (int) (embeddings.size() / n_embd);
            //std::cout << "N is " << N << "\n";
            //int N = n_past + 1;
            if (!llama_eval_internal(*ctx, embeddings.data(), N, n_past, n_threads, nullptr)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }

            //n_past++;
            n_past = n_past + N;

            res.resize(ctx->embedding.size());
            memcpy(res.data(), ctx->embedding.data(), sizeof(float) * ctx->embedding.size());
            //std::cout << "Done Context embedding size " << ctx->embedding.size();
            //std::cout << " last elem " << ctx->embedding[ctx->embedding.size() - 1] <<  "\n";

            //std::cout << " first elem " << ctx->embedding[0] <<  "\n";
            return 0;
        }

        int get_n_embd() {
            return model->hparams.n_embd;
        }

        int get_n_vocab() {
            return model->hparams.n_vocab;
        }

        llama_vocab get_vocab() {
            return ctx->vocab;
        }

        ~TransformerSlice() {
            delete ctx;
            delete model;
        }
};


static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}


struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};


struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};


struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};


std::vector<llama_vocab::id> llama_tokenize(const llama_vocab & vocab, const std::string & text, bool bos) {
    llama_tokenizer tokenizer(vocab);
    std::vector<llama_vocab::id> output;

    if (text.empty()) {
        return output;
    }

    if (bos) {
        output.push_back(llama_token_bos());
    }

    tokenizer.tokenize(text, output);
    return output;
}


class LLMExtra {
    private:
        llama_buffer weight_buf;
        uint32_t n_embd;
        uint32_t n_vocab;
        llama_model_loader* loader;

        struct ggml_tensor* tok_embeddings;
        struct ggml_tensor* norm_weight;
        struct ggml_tensor* output_weight;

        struct ggml_tensor* load_tensor(const std::string & name, const std::vector<uint32_t> & ne) {
            struct ggml_tensor * tensor = loader->get_tensor(name, ne, GGML_BACKEND_CPU);
            auto it = loader->tensors_map.name_to_idx.find(name);


            llama_load_tensor & lt = loader->tensors_map.tensors.at(it->second);
            lt.data = (uint8_t *) tensor->data;
            loader->load_data_for(lt);
            return tensor;
        }
    public:
        LLMExtra(std::string extra_layers_path)
        {
            loader = new llama_model_loader(extra_layers_path, false);
            tok_embeddings = NULL;
            norm_weight = NULL;
            output_weight = NULL;
            n_embd = loader->file_loader->hparams.n_embd;
            n_vocab = loader->file_loader->hparams.n_vocab;

            size_t ctx_size;
            size_t mmapped_size;
            loader->calc_sizes(&ctx_size, &mmapped_size);

            size_t buf_size = ctx_size;
            weight_buf.resize(buf_size);

            struct ggml_init_params weight_params = {
                /*.mem_size   =*/ weight_buf.size,
                /*.mem_buffer =*/ weight_buf.addr,
                /*.no_alloc   =*/ false,
            };
            struct ggml_context * weight_ctx = ggml_init(weight_params);

            loader->ggml_ctx = weight_ctx;
        }

        void load_layers() {
            tok_embeddings = load_tensor("tok_embeddings.weight", {n_embd, n_vocab});
            norm_weight = load_tensor("norm.weight", {n_embd});
            output_weight = load_tensor("output.weight", {n_embd, n_vocab});
        }

        llama_vocab get_vocab() const {
            return loader->file_loader->vocab;
        }

        uint32_t get_n_embd() const {
            return n_embd;
        }

        uint32_t get_n_vocab() const {
            return n_vocab;
        }

        struct ggml_tensor* get_token_embeddings() const {
            return tok_embeddings;
        }

        struct ggml_tensor* get_norm_weight() const {
            return norm_weight;
        }

        struct ggml_tensor* get_output_weight() const {
            return output_weight;
        }

        ~LLMExtra() {
            ggml_free(loader->ggml_ctx);
            delete loader;
        }
};


std::vector<float> get_inputs(const LLMExtra* llm_extra, const llama_token * tokens, int n_tokens, int n_threads) {
    struct ggml_tensor * tok_embeddings = llm_extra->get_token_embeddings();

    //this buffer used to store tokens is way to large than it needs to be (n_tokens * 4 bytes should be enough)
    auto n_embd = llm_extra->get_n_embd();
    auto n_vocab = llm_extra->get_n_vocab();
    size_t buf_size = n_vocab * n_embd;

    llama_buffer buf;
    buf.resize(buf_size);    
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf.size,
        /*.mem_buffer =*/ buf.addr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx0 = ggml_init(params);

    //std::cout << "Created ggml context \n";
    ggml_cgraph gf = {};
    std::vector<uint8_t> work_buffer;

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

    memcpy(inp_tokens->data, tokens, n_tokens * ggml_element_size(inp_tokens));
    ggml_set_name(inp_tokens, "inp_tokens");

    struct ggml_tensor * rows = ggml_get_rows(ctx0, tok_embeddings, inp_tokens);

    ggml_build_forward_expand(&gf, rows);

    ggml_graph_compute_helper(work_buffer, &gf, n_threads);

    //std::cout << "Computed agraph \n";
    struct ggml_tensor * res = gf.nodes[gf.n_nodes - 1];
  
    std::vector<float> inputs;

    inputs.resize(n_tokens * n_embd);
    memcpy(inputs.data(), (float *) ggml_get_data(res), sizeof(float)*n_embd * n_tokens);
    
    ggml_free(ctx0);
    return inputs;
}


std::vector<float> get_llm_output(const LLMExtra* llm_extra, const std::vector<float>& embeddings,
                                  bool all_logits=false) {
    auto n_embd = llm_extra->get_n_embd();
    auto n_vocab = llm_extra->get_n_vocab();
    auto N = (int) (embeddings.size() / n_embd);

    //std::cout << "About to load tok weights\n";
    struct ggml_tensor * norm_matrix = llm_extra->get_norm_weight();
    struct ggml_tensor * output_matrix = llm_extra->get_output_weight();

    //what's the proper way to calculate the memory requirement here
    size_t buf_size = 10000 * 10000;
    llama_buffer buf;
    buf.resize(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf.size,
        /*.mem_buffer =*/ buf.addr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph gf = {};
    std::vector<uint8_t> work_buffer;

    struct ggml_tensor * cur;
    struct ggml_tensor * embeddings_tensor;

    embeddings_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);
    memcpy(embeddings_tensor->data, embeddings.data(), N * n_embd * ggml_element_size(embeddings_tensor));

    //lctx.use_buf(ctx0, 0);

    // norm
    {
        cur = ggml_rms_norm(ctx0, embeddings_tensor);
        //offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, norm_matrix);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, output_matrix, cur);
    ggml_set_name(cur, "result_output");

    //lctx.use_buf(ctx0, -1);

    ggml_build_forward_expand(&gf, cur);

    ggml_graph_compute_helper(work_buffer, &gf, 1);

    struct ggml_tensor * res = gf.nodes[gf.n_nodes - 1];

    std::vector<float> logits_out;
    size_t logits_total;
    int logits_offset;
    if (all_logits) {
        logits_total = N * n_vocab;
        logits_offset = 0;
    } else {
        logits_total = n_vocab;
        logits_offset = n_vocab * (N - 1);
    }

    //std::cout << "IN get_llm_output: logits_total IS " << logits_total << std::endl;
    logits_out.resize(logits_total);
    memcpy(logits_out.data(), (float *) ggml_get_data(res) + logits_offset, sizeof(float) * logits_total);

    ggml_free(ctx0);
    return logits_out;
}

llama_token sample_next_token(const LLMExtra* llm_extra, const std::vector<float>& embeddings) {
    std::vector<float> logits_out = get_llm_output(llm_extra, embeddings);
    
    float max_value = -(1000000000000.0);
    llama_token token_id = 0;
    
    for (size_t i = 0; i < logits_out.size(); i++) {
        if (logits_out[i] > max_value) {
            max_value = logits_out[i];
            token_id = i;
        }
    }

    return token_id;
}

TransformerSlice *slice;
LLMExtra* llm_extra = NULL;


void load_llm_extra(std::string extra_layer_path) {
    if (llm_extra == NULL) {
        llm_extra = new LLMExtra(extra_layer_path);
        llm_extra->load_layers();
    }
}


static PyObject *
load_slice(PyObject *self, PyObject *args) {
    gpt_params params;

    int n_threads = 3;
    const char *cstr_path;
    if (!PyArg_ParseTuple(args, "s", &cstr_path))
        return NULL;

    std::string slice_path(cstr_path);
    
    slice = new TransformerSlice(slice_path, params, n_threads);

    return PyLong_FromLong(0);
}


static PyObject *
clear_context(PyObject *self, PyObject *args) {
    std::cout << "in clear context" << std::endl;
    if (slice) {
        slice->clear_context();
        std::cout << "cleared context" << std::endl;
    }

    return PyLong_FromLong(0);
}

static PyObject *
unload_slice(PyObject *self, PyObject *args) {
    if (slice) {
        delete slice;
    }

    return PyLong_FromLong(0);
}

static PyObject *
tokenize_prompt(PyObject *self, PyObject *args) {
    const char *extra_layers_path;
    const char *prompt_cstr;
    if (!PyArg_ParseTuple(args, "ss", &extra_layers_path, &prompt_cstr))
        return NULL;
    
    std::string prompt = prompt_cstr;
    std::string llm_extra_path = extra_layers_path;
    load_llm_extra(llm_extra_path);

    llama_vocab vocab = llm_extra->get_vocab();

    std::vector<llama_token> tokens = llama_tokenize(vocab, prompt, true);    

    PyObject* result = PyList_New(tokens.size());

    for (size_t i = 0; i < tokens.size(); i++)
    {
        llama_token token = tokens[i];
        PyObject* python_value = Py_BuildValue("i", token);
        PyList_SetItem(result, i, python_value);
    }

    return result;
} 

static PyObject *
prepare_embeddings(PyObject *self, PyObject *args) {
    const char* extra_layer_path_cstr;
    PyObject *tokens_list;

    if (!PyArg_ParseTuple(args, "sO", &extra_layer_path_cstr, &tokens_list))
        return NULL;
    
    std::string extra_layers_path(extra_layer_path_cstr);
    PyObject *iter = PyObject_GetIter(tokens_list);
    if (!iter) {
        return NULL;
    }

    std::vector<llama_token> tokens;

    while(true) {
        PyObject *token_id = PyIter_Next(iter);
        if (!token_id) {
            break;
        }
        if (!PyLong_Check(token_id)) {
            return NULL;
        }

        llama_token token = static_cast<llama_token>(PyLong_AsLong(token_id));
        tokens.push_back(token);
    }

    const int n_threads = 3;
    load_llm_extra(extra_layers_path);
    std::vector<float> embeddings = get_inputs(llm_extra, tokens.data(), tokens.size(), n_threads);

    PyObject* result = PyList_New(embeddings.size());

    for (size_t i = 0; i < embeddings.size(); i++)
    {
        float value = embeddings[i];
        PyObject* python_value = Py_BuildValue("f", value);
        PyList_SetItem(result, i, python_value);
    }

    return result;
}


int python_list_to_embeddings_vector(PyObject *embeddings_list, std::vector<float>& dest) {
    PyObject *iter = PyObject_GetIter(embeddings_list);
    if (!iter) {
        return 1;
    }

    while(true) {
        PyObject *e = PyIter_Next(iter);
        if (!e) {
            break;
        }
        if (!PyFloat_Check(e)) {
            return 2;
        }

        float embedding_value = static_cast<float>(PyFloat_AsDouble(e));
        dest.push_back(embedding_value);
    }

    return 0;
}


static PyObject *
propagate_forward(PyObject *self, PyObject *args) {
    PyObject *embeddings_list;

    if (!PyArg_ParseTuple(args, "O", &embeddings_list))
        return NULL;
    
    std::vector<float> embeddings;
    int error_code = python_list_to_embeddings_vector(embeddings_list, embeddings);
    
    if (error_code) {
        return NULL;
    }
    
    std::vector<float> output;
    
    int n_embd = slice->get_n_embd();
    int n_vocab = slice->get_n_vocab();
    int n_threads = 3;

    int status = slice->forward(embeddings, output);
    if (status != 0) {
        std::cout << "something went wrong1\n";
        return PyLong_FromLong(status);
    }

    PyObject* result = PyList_New(output.size());

    for (size_t i = 0; i < output.size(); i++)
    {
        float value = output[i];
        PyObject* python_value = Py_BuildValue("f", value);
        PyList_SetItem(result, i, python_value);
    }

    return result;
}

static PyObject *
get_logits(PyObject *self, PyObject *args) {
    const char* extra_layers_path_cstr;
    PyObject *embeddings_list;
    int all_logits;
    
    //todo: add option to specify whether to return logits per all input embeddings
    if (!PyArg_ParseTuple(args, "sOp", &extra_layers_path_cstr, &embeddings_list, &all_logits))
        return NULL;

    std::string extra_layers_path(extra_layers_path_cstr);    
    std::vector<float> embeddings;
    int error_code = python_list_to_embeddings_vector(embeddings_list, embeddings);
    
    if (error_code) {
        return NULL;
    }

    load_llm_extra(extra_layers_path);
    std::vector<float> logits = get_llm_output(llm_extra, embeddings, (bool) all_logits);
    //std::cout << "IN GET_LOGITS: LOGITS.SIZE() IS " << logits.size() << std::endl;
    PyObject* result = PyList_New(logits.size());

    for (size_t i = 0; i < logits.size(); i++)
    {
        float value = logits[i];
        PyObject* python_value = Py_BuildValue("f", value);
        PyList_SetItem(result, i, python_value);
    }
    return result;
}


static PyObject * 
llm_get_next_token(PyObject *self, PyObject *args) {
    const char* extra_layers_path_cstr;
    PyObject *embeddings_list;
    
    if (!PyArg_ParseTuple(args, "sO", &extra_layers_path_cstr, &embeddings_list))
        return NULL;
    
    std::string extra_layers_path(extra_layers_path_cstr);
    std::vector<float> embeddings;
    int error_code = python_list_to_embeddings_vector(embeddings_list, embeddings);
    
    if (error_code) {
        return NULL;
    }

    load_llm_extra(extra_layers_path);
    llama_token next_token = sample_next_token(llm_extra, embeddings);

    return PyLong_FromLong(static_cast<int>(next_token));
}


static PyObject *
decode_token(PyObject *self, PyObject *args) {
    const char* extra_layers_path;
    int token_id;

    if (!PyArg_ParseTuple(args, "si", &extra_layers_path, &token_id))
        return NULL;
    
    std::string extra_layers_path_string = extra_layers_path;
    load_llm_extra(extra_layers_path_string);
    
    llama_vocab vocab = llm_extra->get_vocab();

    std::string tok = vocab.id_to_token[token_id].tok;
    return PyUnicode_FromString(tok.c_str());
}


static PyMethodDef SpamMethods[] = {
     {"load_slice",  load_slice, METH_VARARGS,
     "Load all transformer block layers in the slice"},
     {"unload_slice", unload_slice, METH_VARARGS,
     "Unload slice currently loaded in memory"},
     {"clear_context", clear_context, METH_VARARGS,
     "Clear cached keys and values"},
     {"tokenize_prompt", tokenize_prompt, METH_VARARGS,
     "Convert text prompt into a list of tokens"},
     {"prepare_embeddings", prepare_embeddings, METH_VARARGS,
     "Create embeddings for the first slice by embedding tokens"},
     {"propagate_forward", propagate_forward, METH_VARARGS,
     "Propagate embeddings vector through the layers of LLMs slice"},
     {"get_logits", get_logits, METH_VARARGS,
     "Apply output layer to embeddings to get logits"},
     {"get_next_token", llm_get_next_token, METH_VARARGS,
     "Get next token"},
     {"decode_token", decode_token, METH_VARARGS,
     "Convert token to unicode string"},
     //{"generate", generate, METH_VARARGS,
     // "Generate text"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef llmmodule = {
    PyModuleDef_HEAD_INIT,
    "llm",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};


PyMODINIT_FUNC
PyInit_llm(void) {
    return PyModule_Create(&llmmodule);
}
