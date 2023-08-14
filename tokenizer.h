/*
Module for loading tokenizer vocabulary, using it and related functionalities for tokenizing string
*/
#ifndef TOKENIZER_H
#define TOKENIZER_H

//Vocabulary is binary file... now much more complex
typedef struct{
    int n;
    int max_token_length;
    char **tokens; //n length
    float *scores; //n length
}Vocabulary;

int loadVocab(char *vocabFileName,Vocabulary *vocab,int expectedNumberOfEntries);
int freeVocab(Vocabulary *vocab);
void bpe_encode(char *text, Vocabulary *vocab, int *tokens, int *n_tokens);

#endif