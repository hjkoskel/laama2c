package laama2c

// #include <stdlib.h>
// #include <laama2.h>
// #cgo LDFLAGS: -lm
import "C"
import (
	"fmt"
	"unsafe"
)

type Laama2 struct {
	laamaPtr    C.Laama2
	Temperature float32
	Topp        float32
}

// InitLaama2 load checkpoint file and tokenizer file
func InitLaama2(checkpointFileName string, tokenizerFileName string) (Laama2, error) {
	result := Laama2{}
	ret := C.initLaama2(&result.laamaPtr, C.CString(checkpointFileName), C.CString(tokenizerFileName))
	if int(ret) == 0 {
		return result, nil
	}
	return result, fmt.Errorf("init failed")
}

// Reset laama to default status but do not unload loaded checkpoint. Used when creating new prompt
func (p *Laama2) Reset() error {
	if int(C.resetLaama2(&p.laamaPtr)) == 0 {
		return nil
	}
	return fmt.Errorf("reset failed")
}

// FeedPrompt is used at begining when feeding prompt.. after intialization or reset of prompt. Returns next token as string
func (p *Laama2) FeedPrompt(prompt string) (string, error) {
	resp := C.CString("")

	pro := C.CString(prompt)
	tokensFound := C.int(0)

	ret := C.laamaFeedPrompt(&resp, &tokensFound, &p.laamaPtr, pro, C.float(p.Temperature), C.float(p.Topp))
	C.free(unsafe.Pointer(pro))

	sResp := C.GoString(resp)
	C.free(unsafe.Pointer(resp))
	if int(ret) == 0 {
		return "", nil //TODO LATER ERROR?
	}
	return sResp, nil
}

// Predict results returns next token as string. Empty string when there end of tokens are reached
func (p *Laama2) Predict() (string, error) {
	resp := C.CString("") //Not going to free, laamaPredicts to just points to existing token

	n := C.laamaPredict(&resp, &p.laamaPtr, C.float(0), C.float(p.Topp))
	sResp := C.GoString(resp)

	if int(n) == 0 {
		return "", nil
	}
	return sResp, nil
}

// Unloads model and frees memory
func (p *Laama2) Free() error {
	if int(C.freeLaama2(&p.laamaPtr)) == 0 {
		return nil
	}
	return fmt.Errorf("free failed")
}

// Randomize internal pseudorandom generator seed value. 0=seed by time, 0< seed is set
func Randomize(seed uint64) {
	C.randomize(C.ulonglong(seed))
}
