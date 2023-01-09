
#include <stdio.h>
#include <sys/time.h>

#define DataType double

double getTimer() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                     int numAColumns, int numBRows, int numBColumns)
{
  //@@ Insert code to implement matrix multiplication here
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if ((col >= numBColumns) || (row >= numARows))
    return;

  DataType tmpSum = 0.0;
  for (int k = 0; k < numAColumns; k++)
  {
    tmpSum += A[row * numAColumns + k] * B[k * numBColumns + col];
  }
  C[row * numBColumns + col] = tmpSum;
}

int main(int argc, char **argv)
{

  DataType *hostA;     // The A matrix
  DataType *hostB;     // The B matrix
  DataType *hostC;     // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output

  int totalSizeA = numARows * numAColumns * sizeof(DataType);
  int totalSizeB = numBRows * numBColumns * sizeof(DataType);
  int totalSizeC = numCRows * numCColumns * sizeof(DataType);
  hostA = (DataType *)malloc(totalSizeA);
  hostB = (DataType *)malloc(totalSizeB);
  hostC = (DataType *)malloc(totalSizeC);
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU

  for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numAColumns; j++)
    {
      DataType randomNumber = (DataType)rand() / RAND_MAX;
      hostA[i * numAColumns + j] = randomNumber;
    }
  }

  for (int i = 0; i < numBRows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      DataType randomNumber = (DataType)rand() / RAND_MAX;
      hostB[i * numBColumns + j] = randomNumber;
    }
  }

  for (int i = 0; i < numARows; i++)
  {
    for (int j = 0; j < numBColumns; j++)
    {
      resultRef[i * numBColumns + j] = 0.0;
      for (int k = 0; k < numAColumns; k++)
      {
        resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
    }
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceA, totalSizeA);
  cudaMalloc(&deviceB, totalSizeB);
  cudaMalloc(&deviceC, totalSizeC);

  //@@ Insert code to below to Copy memory to the GPU here

  double start = getTimer();
  cudaMemcpy(deviceA, hostA, totalSizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, totalSizeB, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double duration = getTimer() - start;
  printf("Host to Device Time: %f\n", duration);

  //@@ Initialize the grid and block dimensions here

  int threadPerBlockX = 32;
  int threadPerBlockY = 32;
  int blockNumX = (numCColumns + threadPerBlockX - 1) / threadPerBlockX;
  int blockNumY = (numCRows + threadPerBlockY - 1) / threadPerBlockY;
  printf("threads per block x: %i y: %i\n", threadPerBlockX, threadPerBlockY);
  printf("blocks num x: %i, y: %i \n", blockNumX, blockNumY);

  //@@ Launch the GPU Kernel here

  start = getTimer();
  gemm<<<dim3(blockNumX, blockNumY, 1), dim3(threadPerBlockX, threadPerBlockY, 1)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  duration = getTimer() - start;
  printf("CUDA Kernel: %f\n", duration);

  //@@ Copy the GPU memory back to the CPU here

  start = getTimer();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  duration = getTimer() - start;
  printf("Device to Host Time: %f\n", duration);

  //@@ Insert code below to compare the output with the reference

  bool allClose = true;
  for (int i = 0; i < numCRows; ++i)
  {
    for (int j = 0; j < numCColumns; ++j)
    {
      if (fabs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > 1e-8)
      {
        allClose = false;
        break;
      }
    }
  }

  allClose ? printf("All good!\n") : printf("Something not equal\n");

  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
