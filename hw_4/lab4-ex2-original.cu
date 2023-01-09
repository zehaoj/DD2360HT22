
#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {out[idx] = in1[idx] + in2[idx];}
  else {return;}
}

//@@ Insert code to implement timer start

double getTimer() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  
  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output

  int inputActualSize = inputLength * sizeof(DataType);
  hostInput1 = (DataType*) malloc(inputActualSize);
  if (hostInput1 == 0) {printf("hostInput1 malloc fail\n"); return 1;}
  hostInput2 = (DataType*) malloc(inputActualSize);
  if (hostInput2 == 0) {printf("hostInput2 malloc fail\n"); return 1;}
  hostOutput = (DataType*) malloc(inputActualSize);
  if (hostOutput == 0) {printf("hostOutput malloc fail\n"); return 1;}
  resultRef = (DataType*) malloc(inputActualSize);
  if (resultRef == 0) {printf("resultRef malloc fail\n"); return 1;}
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU

  for (int i = 0; i < inputLength; i++) {
      DataType randomNumber1 = (DataType) rand() / RAND_MAX;
      DataType randomNumber2 = (DataType) rand() / RAND_MAX;
      hostInput1[i] = randomNumber1;
      hostInput2[i] = randomNumber2;
      resultRef[i] = randomNumber1 + randomNumber2;
  }

  //@@ Insert code below to allocate GPU memory here


  cudaMalloc(&deviceInput1, inputActualSize);
  cudaMalloc(&deviceInput2, inputActualSize);
  cudaMalloc(&deviceOutput, inputActualSize);

  //@@ Insert code to below to Copy memory to the GPU here

  double start = getTimer();
  cudaMemcpy(deviceInput1, hostInput1, inputActualSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputActualSize, cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here

  int threadPerBlock = 64;
  int blockNum = (inputLength + threadPerBlock - 1) / threadPerBlock;
  printf("threads per block: %i \n", threadPerBlock);
  printf("blocks num: %i \n", blockNum);
  
  //@@ Launch the GPU Kernel here

  vecAdd <<<blockNum, threadPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostOutput, deviceOutput, inputActualSize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double duration = getTimer() - start;
  printf("Total Time: %f\n", duration);

  //@@ Insert code below to compare the output with the reference

  bool allClose = true;
  for (int i = 0; i < inputLength; i++) {
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-8) {
      allClose = false;
      break;
    }
  }

  allClose ? printf("All good!\n") : printf("Something not equal\n");

  //@@ Free the GPU memory here

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  
  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
