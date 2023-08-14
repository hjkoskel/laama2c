package main

import (
	"fmt"

	"github.com/hjkoskel/laama2c"
)

func main() {
	//Randomize is required, 0 picks seed number by time. More than 0 means fixed seed
	laama2c.Randomize(0)

	fmt.Printf("--testing laama 2 ---\n")
	laama, initErr := laama2c.InitLaama2("/home/henri/aimallit/llama2.c/model.bin", "/home/henri/aimallit/llama2.c/tokenizer.bin")
	if initErr != nil {
		fmt.Printf("init fail %s\n", initErr.Error())
		return
	}

	for {
		newS, _ := laama.Predict()
		if len(newS) == 0 {
			break
		}
		fmt.Printf("%s", newS)
	}
	fmt.Printf("---RESET----\n")
	laama.Reset()

	laama.Temperature = 0.4
	firstS, _ := laama.FeedPrompt("One day, Lily met a Shoggoth")
	fmt.Printf("%s", firstS)

	for {
		newS, _ := laama.Predict()
		if len(newS) == 0 {
			break
		}
		fmt.Printf("%s", newS)
	}

	laama.Free()
}
