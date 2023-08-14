/*
Very basic math functions separated from run.c
*/
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

void matmul(float* xout, float* x, float* w, int n, int d);
void softmax(float* x, int size);
void rmsnorm(float* o, float* x, float* weight, int size);
void accum(float *a, float *b, int size);
int argmax(float* v, int n);

void randomize(unsigned long long seed); //seed by time
unsigned int random_u32();
float random_f32();

#endif