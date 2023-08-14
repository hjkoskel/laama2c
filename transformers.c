/*
Transformer and run state goes together.

This is the "hard part" of whole llama2
*/
#include "transformers.h"
#include "functions.h"

int malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(      p->dim, sizeof(float));
    s->xb = calloc(     p->dim, sizeof(float));
    s->xb2 = calloc(    p->dim, sizeof(float));
    s->hb = calloc(     p->hidden_dim, sizeof(float));
    s->hb2 = calloc(    p->hidden_dim, sizeof(float));
    s->q = calloc(      p->dim, sizeof(float));
    s->k = calloc(      p->dim, sizeof(float));
    s->v = calloc(      p->dim, sizeof(float));
    s->att = calloc(    p->seq_len, sizeof(float));
    
    s->probindex = calloc(p->vocab_size, sizeof(ProbIndex));

    s->layers=calloc(p->n_layers,sizeof(LayerRunState));
    for (int i=0;i <p->n_layers;i++){
        s->layers[i].key_cache=calloc(p->seq_len * p->dim, sizeof(float));
        s->layers[i].value_cache=calloc(p->seq_len * p->dim, sizeof(float));
        if (!s->layers[i].key_cache || !s->layers[i].value_cache){
            return -1;
        }
    }
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->layers || !s->probindex) {
        return -1;
    }
    return 0;
}

void free_run_state(RunState* s,int layers) {
    for (int i=0;i<layers;i++){
        free(s->layers[i].key_cache);
        free(s->layers[i].value_cache);
        s->layers[i].key_cache=NULL;
        s->layers[i].value_cache=NULL;
    }
    free(s->layers);
    s->layers=NULL;

    free(s->x);
    s->x=NULL;
    free(s->xb);
    s->xb=NULL;
    free(s->xb2);
    s->xb2=NULL;
    free(s->hb);
    s->hb=NULL;
    free(s->hb2);
    s->hb2=NULL;
    free(s->q);
    s->q=NULL;
    free(s->k);
    s->k=NULL;
    free(s->v);
    s->v=NULL;
    free(s->att);
    s->att=NULL;
    free(s->probindex);
    s->probindex=NULL;
}


void transformerLayer(int pos,TransformerLayerWeights *layerw,LayerRunState *slay,RunState* s,Config* conf,float* freq_cis_real_row,float* freq_cis_imag_row){
    int dim = conf->dim;
    int hidden_dim =  conf->hidden_dim;
    int head_size = dim / conf->n_heads;

    // attention rmsnorm
    rmsnorm(s->xb, s->x, layerw->rms_att_weight, dim);
    // qkv matmuls for this position
    matmul(s->q, s->xb, layerw->wq, dim, dim);
    matmul(s->k, s->xb, layerw->wk, dim, dim);
    matmul(s->v, s->xb, layerw->wv, dim, dim);

    // apply RoPE rotation to the q and k vectors for each head
    for (int h = 0; h < conf->n_heads; h++) {
        // get the q and k vectors for this head
        float* q = s->q + h * head_size;
        float* k = s->k + h * head_size;
        // rotate q and k by the freq_cis_real and freq_cis_imag
        for (int i = 0; i < head_size; i+=2) {
            float q0 = q[i];
            float q1 = q[i+1];
            float k0 = k[i];
            float k1 = k[i+1];
            float fcr = freq_cis_real_row[i/2];
            float fci = freq_cis_imag_row[i/2];
            q[i]   = q0 * fcr - q1 * fci;
            q[i+1] = q0 * fci + q1 * fcr;
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }

    // save key,value at this time step (pos) to our kv cache
    float* key_cache_row = &slay->key_cache[pos*dim];
    float* value_cache_row = &slay->value_cache[pos*dim];

    memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
    memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));
        
    // multihead attention. iterate over all heads
    for (int h = 0; h < conf->n_heads; h++) {
        // get the query vector for this head
        float* q = s->q + h * head_size;    
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = slay->key_cache+ t * dim + h * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            s->att[t] = score;
        }
        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(s->att, pos + 1);
        // weighted sum of the values, store back into xb
        for (int i = 0; i < head_size; i++) {
            float val = 0.0f;
            for (int t = 0; t <= pos; t++) {
                val += s->att[t] * slay->value_cache[t * dim + h * head_size + i]; // note bad locality
            }
            s->xb[h * head_size + i] = val;
        }
    }
    // final matmul to get the output of the attention        
    matmul(s->xb2, s->xb, layerw->wo, dim, dim);
    // residual connection back into x
    accum(s->x, s->xb2, dim);
    // ffn rmsnorm
    rmsnorm(s->xb, s->x, layerw->rms_ffn_weight, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s->hb, s->xb, layerw->w1, dim, hidden_dim);
    matmul(s->hb2, s->xb, layerw->w3, dim, hidden_dim);

    // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid AND  elementwise multiply with w3(x)
    for (int i = 0; i < hidden_dim; i++) {
        s->hb[i] = s->hb2[i] * s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
    }    

    // final matmul to get the output of the ffn
    matmul(s->xb, s->hb, layerw->w2, hidden_dim, dim);
    // residual connection
    accum(s->x, s->xb, dim);
}

//Logits is response
void transformer(int token, int pos, Config* conf, RunState* s, TransformerWeights* w, float *logits) {
    // a few convenice variables
    int head_size = conf->dim / conf->n_heads;

    // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * conf->dim]);
    memcpy(s->x, content_row, conf->dim*sizeof(*s->x));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for(int lay = 0; lay < conf->n_layers; lay++) {
        transformerLayer(pos,&w->layers[lay],&s->layers[lay],s,conf,freq_cis_real_row,freq_cis_imag_row);
    }
    // final rmsnorm
    rmsnorm(s->x, s->x, w->rms_final_weight, conf->dim);
    // classifier into logits
    matmul(logits, s->x, w->token_embedding_table, conf->dim, conf->vocab_size);
}
