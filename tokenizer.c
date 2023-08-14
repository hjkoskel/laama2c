#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

int loadVocab(char *vocabFileName,Vocabulary *vocab,int expectedNumberOfEntries){
    vocab->n=expectedNumberOfEntries;
    vocab->tokens=(char**)malloc(expectedNumberOfEntries * sizeof(char*));
    vocab->scores=(float*)malloc(expectedNumberOfEntries * sizeof(float));

    FILE *file = fopen(vocabFileName, "r");
    if (!file) {
        fprintf(stderr,"unable to open the tokenizer vocabulary file %s! Run python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n",vocabFileName);
        return 1;
    }

    if (fread(&vocab->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read max_token_length\n"); return 1; 
    }

    int len;
    for (int i = 0; i < expectedNumberOfEntries; i++) {
        if (fread(&vocab->scores[i], sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1;}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
        vocab->tokens[i] = (char *)malloc(len + 1);
        if (fread(vocab->tokens[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); return 1; }
        vocab->tokens[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
    return 0;
}


int freeVocab(Vocabulary *vocab){
    for(int i=0;i<vocab->n;i++){
        free(vocab->tokens[i]);
        vocab->tokens[i]=NULL;
    }
    free(vocab->tokens);
    vocab->tokens=NULL;
    free(vocab->scores);
    vocab->scores=NULL;
    return 0;
}


int str_lookup(char *str, char **vocab, int vocab_size) {
    // find the first perfect match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void bpe_encode(char *text, Vocabulary *vocab, int *tokens, int *n_tokens) {
    // a temporary buffer to merge two consecutive tokens
    char* str_buffer = malloc((vocab->max_token_length*2+1) * sizeof(char)); // *2 for concat, +1 for null terminator
    // first encode every individual byte in the input string
    *n_tokens = 0; // the number of tokens
    for (char *c = text; *c != '\0'; c++) {
        sprintf(str_buffer, "%c", *c);
        int id = str_lookup(str_buffer, vocab->tokens, vocab->n);
        if (id == -1) { fprintf(stderr, "not good\n"); exit(EXIT_FAILURE); }
        tokens[*n_tokens] = id;
        (*n_tokens)++;
    }
    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab->tokens[tokens[i]], vocab->tokens[tokens[i+1]]);
            int id = str_lookup(str_buffer, vocab->tokens, vocab->n);
            if (id != -1 && vocab->scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab->scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }
    free(str_buffer);
}