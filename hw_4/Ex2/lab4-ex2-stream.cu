
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
  int nStreams;
  int segSize;
  // DataType *deviceMemory;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;

  //@@ Insert code below to read in inputLength from args
  
  inputLength = atoi(argv[1]);
  nStreams = atoi(argv[2]);
  segSize = inputLength / nStreams;

  printf("The input length is %d\n", inputLength);
  printf("The segment size is %d\n", segSize);
  printf("The segment num is %d\n", nStreams);

  
  //@@ Insert code below to allocate Host memory for input and output

  int inputActualSize = inputLength * sizeof(DataType);
  cudaHostAlloc((void **) &hostInput1, inputActualSize, cudaHostAllocDefault);
  cudaHostAlloc((void **) &hostInput2, inputActualSize, cudaHostAllocDefault);
  cudaHostAlloc((void **) &hostOutput, inputActualSize, cudaHostAllocDefault);
  resultRef = (DataType*)malloc(inputActualSize);
  
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


  cudaStream_t stream[nStreams];

  for(int i = 0; i < nStreams; i++)
    cudaStreamCreate(&stream[i]);


  //@@ Initialize the 1D grid and block dimensions here

  int threadPerBlock = 64;
  int blockNum = (segSize + threadPerBlock - 1) / threadPerBlock;
  printf("threads per block: %i \n", threadPerBlock);
  printf("blocks num: %i \n", blockNum);
  

  double start = getTimer();

  for(int i = 0; i < nStreams; i++){
    int offset = i * segSize;
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], segSize * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset],  segSize * sizeof(DataType), cudaMemcpyHostToDevice, stream[i]);
    vecAdd<<<blockNum,threadPerBlock,0,stream[i]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, segSize);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],  segSize * sizeof(DataType), cudaMemcpyDeviceToHost, stream[i]);
  }
  
  cudaDeviceSynchronize();

  double duration = getTimer() - start;
  printf("Total Time: %f\n", duration);


  for(int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(stream[i]);
  }
  //@@ Insert code below to compare the output with the reference

  bool allClose = true;
  for (int i = 0; i < inputLength; i++) {
    if (fabs(hostOutput[i] - resultRef[i]) > 1e-4) {
      allClose = false;
      break;
    }
  }

  allClose ? printf("All good!\n") : printf("Something not equal\n");

  //@@ Free the GPU memory here
  for (int i=0; i < nStreams; i++)
    cudaStreamDestroy(stream[i]);
  
  // cudaFree(deviceMemory);
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  
  //@@ Free the CPU memory here
  cudaFree(hostInput1);
  cudaFree(hostInput2);
  cudaFree(hostOutput);
  free(resultRef);

  return 0;
}
