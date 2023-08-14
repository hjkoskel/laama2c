/*
This contains sampling algorithms.. 
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

int sampler(float *logits,int nLogits ,float temperature,float topp,ProbIndex *probindex);

#endif