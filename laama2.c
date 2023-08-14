/*
Simplified llama2.c

Goal is to create implementation so it is easier to explain how this works

*/

#include "laama2.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>


int resetLaama2(Laama2 *laama) { //TODO zero out something?
    laama->token=1;
    laama->pos=0;
    return 0;
}

int initLaama2(Laama2 *result ,char *checkpointFileName,char *tokenizerFileName){
    if (loadCheckpoint(checkpointFileName,&result->config,&result->weights)){
        fprintf(stderr,"initLaama2 failed loading checkpoint file %s\n",checkpointFileName);
        return -1;
    }
    if (loadVocab(tokenizerFileName,&result->vocab,result->config.vocab_size)){
        fprintf(stderr,"initLaama2 failed,error loading vocub from file %s with size %d\n",tokenizerFileName,result->config.vocab_size);
        return -2;
    }
    if (malloc_run_state(&result->state, &result->config)){
        return -3;
    }
    result->logits = calloc(result->config.vocab_size, sizeof(float));
    if (!result->logits){
        fprintf(stderr,"initLaama2 failed, error allocating memory for logits with size %d\n",result->config.vocab_size);
        return -4;
    }

    return resetLaama2(result);
}


int freeLaama2(Laama2 *laama){
    // memory cleanup
    free(laama->logits);
    free_run_state(&laama->state,laama->config.n_layers);
    free_weights(&laama->weights,laama->config.n_layers);
    freeVocab(&laama->vocab);
    return 0;
}

//Point to dictionary entry. Predicts what comes next
int laamaPredict(char **result, Laama2 *laama,float temperature,float topp) {
    if ((laama->config.seq_len<=laama->pos) || ((laama->token==1)&&(0<laama->pos))) {
        laama->token=1;
        result[0]=laama->vocab.tokens[laama->token];
        return 0;//reached max length
    }
    transformer(laama->token, laama->pos, &laama->config, &laama->state, &laama->weights,laama->logits);

    if(temperature == 0.0f){ // not really sampling, picking up just most propable
        laama->token=argmax(laama->logits,laama->vocab.n);
    }else{
        laama->token=sampler(laama->logits,laama->vocab.n,temperature,topp,laama->state.probindex);
    }
    //give pointer to new token string
    result[0]=laama->vocab.tokens[laama->token];
    if (laama->token==1){
        return 0;//Not anymore
    }
    laama->pos++;
    return 1; //got one
}

//Use initial prompt... or in between? RETURNS FIRST TOKEN!
int laamaFeedPrompt(char **result,int *tokensFound,Laama2 *laama,char *prompt, float temperature,float topp){
    tokensFound[0]=0;
    if(strlen(prompt)==0) {
        return 0;
    }
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)calloc(strlen(prompt) , sizeof(int));
    if (!prompt_tokens) {
        fprintf(stderr,"memory allocation failed");
        return -1;
    }

    //TODO return error?
    bpe_encode(prompt, &laama->vocab, prompt_tokens, &num_prompt_tokens);
    if (laama->pos==0) {
        transformer(1, laama->pos, &laama->config, &laama->state, &laama->weights,laama->logits);
        laama->pos++;
    }
        
    for(int i=0;i<num_prompt_tokens-1;i++) {
        transformer(prompt_tokens[i], laama->pos, &laama->config, &laama->state, &laama->weights,laama->logits);
        laama->pos++;
    }
    laama->token=prompt_tokens[num_prompt_tokens-1];

    tokensFound[0]=laamaPredict(result,laama,temperature,topp);
    free(prompt_tokens);
    return 0;
}
