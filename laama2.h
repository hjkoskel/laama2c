/*

llama2c made by Andrei Karpathy,

This is cleaned up and some modifications made so this is usable as library

*/
#ifndef LAAMA2_H
#define LAAMA2_H

#include "transformers.h"
#include "functions.h"
#include "checkpoint.h"
#include "sampler.h"
#include "tokenizer.h"


//Capsulates all needed, keeps llama2.c status and also state of laama
typedef struct{
    Config config;
    TransformerWeights weights;
    Vocabulary vocab;
    RunState state;
    float *logits;
    int token; // 1 = BOS token in Llama-2 sentencepiece
    int pos;
}Laama2;

int laamaFeedPrompt(char **result,int *tokensFound,Laama2 *laama,char *prompt, float temperature,float topp);
int laamaPredict(char **result, Laama2 *laama,float temperature,float topp);
int freeLaama2(Laama2 *laama);
int initLaama2(Laama2 *result ,char *checkpointFileName,char *tokenizerFileName);
int resetLaama2(Laama2 *laama);

#endif