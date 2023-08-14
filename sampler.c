/*
Different samplers?
*/

#include "sampler.h"
#include "functions.h"

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".

    // quicksort indices in descending order of probabilities
    for (int i = 0; i < n; i++) {
        probindex[i].index = i;
        probindex[i].prob = probabilities[i];
    }
    qsort(probindex, n, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < n; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = random_f32() * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = random_f32(); //(float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int sampler(float *logits,int nLogits ,float temperature,float topp,ProbIndex *probindex){
    if (temperature==0){
        return -1; //prevent div by 0
    }
    // apply the temperature to the logits
    for (int q=0; q<nLogits; q++) { 
        logits[q] /= temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, nLogits);
    // we now want to sample from this distribution to get the next token
    if (topp<=0)
        return sample(logits,nLogits);   

    // top-p (nucleus) sampling, clamping the least likely tokens to zero
    return sample_topp(logits, nLogits, topp, probindex);
}



