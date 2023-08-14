#ifndef TRANSFORMERS_H
#define TRANSFORMERS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "checkpoint.h"
#include "sampler.h"

typedef struct{
    float* key_cache;   // (seq_len, dim)
    float* value_cache; // (seq_len, dim)
}LayerRunState;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)  TODO splitattuna headeihin?
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (seq_len,)
    ProbIndex *probindex; // buffer used in top-p sampling
    // kv cache
    LayerRunState *layers;
} RunState;

int malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s,int layers);
void transformer(int token, int pos, Config* conf, RunState* s, TransformerWeights* w, float *logits);

#endif