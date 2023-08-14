#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>


int malloc_layerweights(TransformerLayerWeights* w, int dim, int hiddenDim) {
    w->rms_att_weight = calloc(dim,             sizeof(float));
    w->rms_ffn_weight = calloc(dim,             sizeof(float));
    w->wq =             calloc(dim*dim,         sizeof(float));
    w->wk =             calloc(dim*dim,         sizeof(float));
    w->wv =             calloc(dim*dim,         sizeof(float));
    w->wo =             calloc(dim*dim,         sizeof(float));
    w->w1 =             calloc(hiddenDim*dim,  sizeof(float));
    w->w2 =             calloc(dim*hiddenDim,  sizeof(float));
    w->w3 =             calloc(hiddenDim*dim,   sizeof(float));

    if (!w->rms_att_weight || !w->rms_ffn_weight || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3) {
        fprintf(stderr,"malloc_layerweights failed!\n");
        return -1;
    }
    return 0;
}

int malloc_weights(TransformerWeights* w, Config* p) {
    w->token_embedding_table = calloc(p->vocab_size * p->dim, sizeof(float));

    w->layers=calloc(p->n_layers,sizeof(TransformerLayerWeights));
    for (int i=0;i <p->n_layers;i++){
        int ret=malloc_layerweights(&w->layers[i],p->dim,p->hidden_dim);
        if (ret){
            return ret;
        }
    }
    w->rms_final_weight = calloc(p->dim, sizeof(float));
    w->freq_cis_real = calloc(p->seq_len * p->dim / 2, sizeof(float));
    w->freq_cis_imag = calloc(p->seq_len * p->dim / 2, sizeof(float));
    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag) {
        fprintf(stderr,"malloc_weights failed!\n");
        return -1;
    }
    return 0;
}

void free_layerweights(TransformerLayerWeights* w) {
    free(w->rms_att_weight);
    w->rms_att_weight=NULL;  //Yes, annoying. My habit to also null pointers after free. Have prevented some problems

    free(w->rms_ffn_weight);
    w->rms_ffn_weight=NULL;
    
    free(w->wq);
    w->wq=NULL;

    free(w->wk);
    w->wk=NULL;

    free(w->wv);
    w->wv=NULL;

    free(w->wo);
    w->wo=NULL;

    free(w->w1);
    w->w1=NULL;

    free(w->w2);
    w->w2=NULL;

    free(w->w3);
    w->w3=NULL;
}


void free_weights(TransformerWeights* w,int layers) {
    free(w->token_embedding_table);
    for (int i=0;i <layers;i++){
        free_layerweights(&w->layers[i]);
    }
    free(w->layers);
    w->layers=NULL;
    free(w->rms_final_weight);
    w->rms_final_weight=NULL;
    free(w->freq_cis_real);
    w->freq_cis_real=NULL;
    free(w->freq_cis_imag);
    w->freq_cis_imag=NULL;
}


int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f) {
    if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, f) != p->vocab_size * p->dim) return 1;
    
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].rms_att_weight, sizeof(float), p->dim, f) != p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].wq, sizeof(float), p->dim * p->dim, f) != p->dim * p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].wk, sizeof(float), p->dim * p->dim, f) != p->dim * p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].wv, sizeof(float), p->dim * p->dim, f) != p->dim * p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].wo, sizeof(float), p->dim * p->dim, f) !=  p->dim * p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].rms_ffn_weight, sizeof(float), p->dim, f) !=  p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].w1, sizeof(float), p->dim * p->hidden_dim, f) != p->dim * p->hidden_dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].w2, sizeof(float), p->hidden_dim * p->dim, f) != p->hidden_dim * p->dim) return 1;
    }
    for (int i=0;i <p->n_layers;i++){
        if (fread(w->layers[i].w3, sizeof(float), p->dim * p->hidden_dim, f) != p->dim * p->hidden_dim) return 1;
    }
    
    if (fread(w->rms_final_weight, sizeof(float), p->dim, f) != p->dim) return 1;
    int head_size = p->dim / p->n_heads;
    if (fread(w->freq_cis_real, sizeof(float), p->seq_len * head_size / 2, f) != p->seq_len * head_size / 2) return 1;
    if (fread(w->freq_cis_imag, sizeof(float), p->seq_len * head_size / 2, f) != p->seq_len * head_size / 2) return 1;
    return 0;
}

int loadCheckpoint(char *checkpointFileName,Config *config,TransformerWeights *weights){
    FILE *file = fopen(checkpointFileName, "rb");
    if (!file) {
        fprintf(stderr,"unable to open the checkpoint file %s!\n", checkpointFileName);
        return 1;
    }
    // read in the config header
    if(fread(config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr,"error loading config header from file %s\n",checkpointFileName);
        return 1;
    }
    // malloc and read in the Transformer weights
    int ret=malloc_weights(weights, config);
    if (ret){
        return ret;
    }
    if(checkpoint_init_weights(weights, config, file)) {
        fprintf(stderr,"error loading weights from file %s\n",checkpointFileName); return 1; 
    }
    return fclose(file);
}
