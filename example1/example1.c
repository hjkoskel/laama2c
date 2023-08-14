#include "laama2.h"
#include "functions.h"

long time_in_ms() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000 + time.tv_usec / 1000;
}


int main(int argc, char *argv[]) {
    randomize(0);
    Laama2 laama;

    if (initLaama2(&laama ,"/home/henri/aimallit/llama2.c/model.bin","/home/henri/aimallit/llama2.c/tokenizer.bin")) {
        printf("initializing and loading laama failed\n");
        return -1;
    }

    char *newText;
    while (0<laamaPredict(&newText,&laama,0,0)) {
        printf("%s",newText);
        fflush(stdout);
    }
    printf("\n");

    printf("---RESET----\n");
    resetLaama2(&laama);
    
    char *lastPart;
    int tokensFound=0;
    laamaFeedPrompt(&lastPart,&tokensFound, &laama,"One day, Lily met a Shoggoth", 0.8,0);
    printf("%s",lastPart);

    while (0<laamaPredict(&newText,&laama,0,0)) {
        printf("%s",newText);
    }
    printf("\n");

    if (freeLaama2(&laama)) {
        printf("freeing laama failed");
        return -1;
    }
    return 0;
}

