/*
Checkpoint is where model is saved by training program
*/
#ifndef CHECKPOINT_H
#define CHECKPOINT_H

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // weights for rmsnorms
    float* rms_att_weight; // dim rmsnorm weights
    float* rms_ffn_weight; // dim
    // weights for matmuls
    float* wq; // (dim, dim)
    float* wk; // (dim, dim)
    float* wv; // (dim, dim)
    float* wo; // (dim, dim)
    // weights for ffn
    float* w1; // (hidden_dim, dim)
    float* w2; // (dim, hidden_dim)
    float* w3; // (hidden_dim, dim)
} TransformerLayerWeights;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // layers
    TransformerLayerWeights* layers; //array of number of arrays 
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
} TransformerWeights;

int loadCheckpoint(char *checkpointFileName,Config *config,TransformerWeights *weights);
void free_weights(TransformerWeights* w,int layers);

#endif