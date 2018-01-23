#define NSavingThreads 10
#define ClippedHistCutOffSlope 3
#include "CudaHeader/EHCuda.h"

CreateThread(Reading_Thread);
CreateMutex(ReadingThread_Mutex);
CreateCond(ReadingThread_Cond);

CreateThread(Saving_Thread[NSavingThreads]);
CreateMutex(SavingThread_Mutex[NSavingThreads]);
CreateCond(SavingThread_Cond[NSavingThreads]);

CreateStream(Reading_Stream);
CreateStream(Processing_Stream);

CudaFunction
void
ClipHist(u32* hist, u32 n, f32 cutoffSlope)
{
	u32 C = (u32)((cutoffSlope * (f32)n / (f32)u08_n) + 0.5);
	u32 S = 0;
	
	u32 top = C;
	u32 bottom = 0;

	while ((top - bottom) > 1)
	{
		u32 middle = (top + bottom) / 2;
		S = 0;
		
		for (	u32 index = 0;
			index < u08_n;
			++index )
		{
			if (hist[index] >= middle)
			{
				S += (hist[index] - middle);
			}
		}

		if (S > ((C - middle) * u08_n))
		{
			top = middle;
		}
		else
		{
			bottom = middle;
		}
	}

	u32 P = bottom + (u32)(((f32)S / (f32)u08_n) + 0.5);
	u32 L = C - P;

	for (	u32 index = 0;
		index < u08_n;
		++index )
	{
		if (hist[index] < P)
		{
			hist[index] += L;
		}
		else
		{
			hist[index] = C;
		}
	}
}

CudaFunction
void
CreateLookup(u32* clippedHist)
{
	u32 total = 0;

	for (	u32 index = 0;
		index < u08_n;
		++index )
	{
		total += clippedHist[index];
		clippedHist[index] = total;
	}
	
	for (	u32 index = 0;
		index < u08_n;
		++index )
	{
		clippedHist[index] = (u32)(((f32)clippedHist[index] / (f32)total * (f32)u08_max) + 0.5);
	}
}

CudaKernel
ZeroHistograms(u32 *histograms, dim3 histogramGridDims)
{
	u32 histogramGridN = Dim3N(histogramGridDims);	

	OneDCudaLoop(index, histogramGridN * u08_n)
	{
		histograms[index] = 0;
	}
}

CudaKernel
FillHistograms(u08 *image, u32 *histograms, dim3 imageDims, dim3 histogramGridDims, dim3 histogramLocalDims)
{
	dim3 histogramGlobalDims = Dim3Hadamard(histogramGridDims, histogramLocalDims);
	u32 histogramGlobalN = Dim3N(histogramGlobalDims);

	OneDCudaLoop(index, histogramGlobalN)
	{
		int3 histogramGlobalPosition_asInt = OneDToThreeD(index, histogramGlobalDims);
		dim3 histogramGlobalPosition_asDim = Int3ToDim3(histogramGlobalPosition_asInt);
		dim3 histogramGridPosition = Dim3Divide(histogramGlobalPosition_asDim, histogramLocalDims);
		u32 histogramIndex = ThreeDToOneD(histogramGridPosition, histogramGridDims);
		dim3 imageCoords = CoordinateMap_Reflection(histogramGlobalPosition_asInt, imageDims);
		u32 imageIndex = ThreeDToOneD(imageCoords, imageDims);

		u32 *hist = histograms + (histogramIndex * u08_n);
		atomicAdd(hist + image[imageIndex], 1);
	}
}

CudaKernel
CreateLookups(u32 *histograms, dim3 histogramGridDims, dim3 histogramLocalDims)
{
	u32 histogramLocalN = Dim3N(histogramLocalDims);
	u32 histogramGridN = Dim3N(histogramGridDims);	
	
	OneDCudaLoop(index, histogramGridN)
	{
		u32 *hist = histograms + (index * u08_n);
		
		ClipHist(hist, histogramLocalN, ClippedHistCutOffSlope);
		CreateLookup(hist);
	}
}

CudaFunction
dim3
GetCurrentAndNextHistsAndDist(u32 coord, u32 histDim, u32 nHists)
{
	dim3 result;

	u32 halfHistDim = histDim / 2;
	if (coord < halfHistDim)
	{
		result.x = 0;
		result.y = 0;
		result.z = 0;
	}
	else
	{
		result.x = (coord - halfHistDim) / histDim;
		result.y = result.x + 1;
		result.z = coord - ((result.x * histDim) + halfHistDim);
	}

	if (result.y >= nHists)
	{
		result.y = result.x;
		result.z = 0;
	}

	return(result);
}

CudaFunction
f32
BiLinear(u32 *histogramIndex, u32 *histograms, f32 disX, f32 disY, u08 pixelVal)
{
	f32 omDisX = 1.0 - disX;
	f32 omDisY = 1.0 - disY;

	f32 values[4];
	for ( 	u32 index = 0;
		index < 4;
		++index )
	{
		values[index] = (f32)((histograms + (histogramIndex[index] * u08_n))[pixelVal]);

	}

	f32 result = 	(omDisX	* omDisY	*	values[0]) +
			(disX 	* omDisY	*	values[1]) +
			(omDisX	* disY		*	values[2]) +
			(disX	* disY		*	values[3]);

	return(result);
}

CudaKernel
AHE(u08 *image, u08 *out, u32 *histograms_1, u32 *histograms_2, f32 interpolation, dim3 imageDims, dim3 histogramGridDims, dim3 histogramLocalDims)
{	
	u32 imageN = Dim3N(imageDims);

	OneDCudaLoop(index, imageN)
	{
		u08 pixelVal = image[index];

		dim3 imageCoords = Int3ToDim3(OneDToThreeD(index, imageDims));

		dim3 interpX = GetCurrentAndNextHistsAndDist(imageCoords.x, histogramLocalDims.x, histogramGridDims.x);
		dim3 interpY = GetCurrentAndNextHistsAndDist(imageCoords.y, histogramLocalDims.y, histogramGridDims.y);
		f32 disX = (f32)interpX.z / (f32)histogramLocalDims.x;
		f32 disY = (f32)interpY.z / (f32)histogramLocalDims.y;

		dim3 histogramGridPosition[4];
		histogramGridPosition[0].x = interpX.x;
		histogramGridPosition[0].y = interpY.x;
		histogramGridPosition[0].z = 0;
		histogramGridPosition[1].x = interpX.y;
		histogramGridPosition[1].y = interpY.x;
		histogramGridPosition[1].z = 0;
		histogramGridPosition[2].x = interpX.x;
		histogramGridPosition[2].y = interpY.y;
		histogramGridPosition[2].z = 0;
		histogramGridPosition[3].x = interpX.y;
		histogramGridPosition[3].y = interpY.y;
		histogramGridPosition[3].z = 0;

		u32 histogramIndex[4];
		for (	u32 histIndex = 0;
			histIndex < 4;
			++histIndex )
		{
			histogramIndex[histIndex] = ThreeDToOneD(histogramGridPosition[histIndex], histogramGridDims);
		}

		f32 planeInterpVals[2];
		planeInterpVals[0] = BiLinear(histogramIndex, histograms_1, disX, disY, pixelVal);
		planeInterpVals[1] = BiLinear(histogramIndex, histograms_2, disX, disY, pixelVal);

		out[index] = (u08)((((1.0 - interpolation) * planeInterpVals[0]) + (interpolation * planeInterpVals[1])) + 0.5);
	}	
}

struct
slab_histograms
{
	uint2 slabDims;
	three_d_volume *histVol;
	u32 *imageHistograms;
};

struct
slab_histograms_LL_node
{
	slab_histograms *slabHistogram;
	slab_histograms_LL_node *next;
};

struct
interpolation_histograms
{
	slab_histograms_LL_node *slabHistograms;
	u32 doInterpolation;
	u32 slabCounter;
};

void
InitialiseSlabHistograms(slab_histograms *slabHistograms, uint2 slabDims, three_d_volume *histVol)
{
	slabHistograms->histVol = histVol;	
	slabHistograms->slabDims = slabDims;
	CudaMallocManaged(slabHistograms->imageHistograms, slabDims.x * slabDims.y * u08_n * sizeof(u32));
}

void
FreeSlabHistograms(slab_histograms *slabHistograms)
{
	cudaFree(slabHistograms->imageHistograms);
}

void
SlabHistogramsLLAddAfter(slab_histograms_LL_node *head, slab_histograms_LL_node *tail)
{
	head->next = tail;
}

void
InitialiseSlabHistogramLLNode(slab_histograms_LL_node *node, uint2 slabDims, three_d_volume *histVol)
{
	InitialiseSlabHistograms(node->slabHistogram, slabDims, histVol);
}

void
FreeSlabHistogramLLNode(slab_histograms_LL_node *node)
{
	FreeSlabHistograms(node->slabHistogram);
}

void
InitialiseInterpolationHistograms(interpolation_histograms *interpolationHistograms, uint2 slabDims, three_d_volume *histVol)
{

	InitialiseSlabHistogramLLNode(interpolationHistograms->slabHistograms, slabDims, histVol);
	interpolationHistograms->doInterpolation = 0;
	interpolationHistograms->slabCounter = 0;
}

struct rolling_image_buffer
{
	u32 capacity;
	image_data *buffers[2];
	slab_histograms_LL_node *slabHistograms;
	three_d_volume fullImageVolume;
	image_loader *loader;
	u32 currentZStart;
	to_eight_bit_converter *converter;
};

void
CreateRollingImageBuffer(rolling_image_buffer *buffer, image_loader *loader, image_data *im1, image_data *im2, u32 localZ, to_eight_bit_converter *converter)
{

	CreateImageData_FromInt(im1, ImageLoaderGetWidth(loader), ImageLoaderGetHeight(loader), localZ);
	CreateImageData_FromInt(im2, ImageLoaderGetWidth(loader), ImageLoaderGetHeight(loader), localZ);
	
	buffer->converter = converter;
	buffer->buffers[0] = im1;
	buffer->buffers[1] = im2;
	buffer->loader = loader;
	buffer->currentZStart = 0;
	buffer->capacity = 2 * localZ;

	three_d_volume vol;
	u32 trackSize;
	
	u32 *startingIndex = GetImageLoaderCurrentTrackIndexAndSize(loader, &trackSize);
	CreateThreeDVol_FromInt(&vol, ImageLoaderGetWidth(loader), ImageLoaderGetHeight(loader), trackSize - *startingIndex);
	
	buffer->fullImageVolume = vol;
}

void
FreeRollingImageBuffer(rolling_image_buffer *buffer)
{
	FreeImageData(buffer->buffers[0]);
	FreeImageData(buffer->buffers[1]);
	FreeImageLoader(buffer->loader);
	FreeToEightBitConverter(buffer->converter);
}

three_d_volume*
GetImageVolume_FromImageBuffer(rolling_image_buffer *buffer)
{
	return(&buffer->fullImageVolume);
}

u08*
GetContentsOfCurrentBuffer(rolling_image_buffer *buffer)
{
	return(buffer->buffers[0]->image);	
}

u08*
GetContentsOfNextBuffer(rolling_image_buffer *buffer)
{
	return(buffer->buffers[1]->image);	
}

void
CalculateNewHistograms(image_data *im, slab_histograms *slabHist, u32 actualZSize)
{
	u32 *hist = slabHist->imageHistograms;
	
	dim3 imageDims = im->vol.dims;
	imageDims.z = actualZSize;

	dim3 histGridDims;
	histGridDims.x = slabHist->slabDims.x;
	histGridDims.y = slabHist->slabDims.y;
	histGridDims.z = 1;

	LaunchCudaKernel_Simple_Stream(ZeroHistograms, *Reading_Stream, hist, histGridDims);
	LaunchCudaKernel_Simple_Stream(FillHistograms, *Reading_Stream, im->image, hist, imageDims, histGridDims, slabHist->histVol->dims);
	LaunchCudaKernel_Simple_Stream(CreateLookups, *Reading_Stream, hist, histGridDims, slabHist->histVol->dims);
	cudaStreamSynchronize(*Reading_Stream);
}

struct
reading_thread_data_in
{
	image_data *im;
	image_loader *loader;
	slab_histograms *hist;
	to_eight_bit_converter *converter;
};

reading_thread_data_in *ReadingThread_DataIn;
threadSig *ReadingThreadContinueSignal;
threadSig *ReadingThreadRunSignal;

void
FillSingleBuffer_threaded(image_data *im, image_loader *loader, slab_histograms *hist, to_eight_bit_converter *converter)
{
	while (*ReadingThreadContinueSignal) {}
	FenceIn(*ReadingThreadContinueSignal = 1);
	LockMutex(ReadingThread_Mutex);
	SignalCondition(ReadingThread_Cond);
		
	ReadingThread_DataIn->im = im;
	ReadingThread_DataIn->loader = loader;
	ReadingThread_DataIn->hist = hist;
	ReadingThread_DataIn->converter = converter;

	UnlockMutex(ReadingThread_Mutex);
}

void
FillSingleBuffer(image_data *im, image_loader *loader, slab_histograms *slabHistogram, to_eight_bit_converter *converter)
{
	u32 nPixelsPlane = im->vol.dims.x * im->vol.dims.y;
	memptr nBytesPlane = nPixelsPlane * sizeof(u08);
	u32 localactualCapacity = 0;

	for (	;
		localactualCapacity < im->vol.dims.z;
		++localactualCapacity)
	{
		
		if(!LoadCurrentImageAndAdvanceIndex(loader, converter->buffer)) break;

		ConvertToEightBit(converter, im->image + (localactualCapacity * nPixelsPlane));
	}

	if (localactualCapacity)
	{
		CalculateNewHistograms(im, slabHistogram, localactualCapacity);
	}
}

void
FillRollingBuffer(rolling_image_buffer *buffer)
{
	FillSingleBuffer_threaded(buffer->buffers[0], buffer->loader, buffer->slabHistograms->slabHistogram, buffer->converter);
	FillSingleBuffer_threaded(buffer->buffers[1], buffer->loader, buffer->slabHistograms->next->slabHistogram, buffer->converter);
}

u08*
GetImagePlane(rolling_image_buffer *buffer, u32 zIndex)
{
	u32 bufferOffset = zIndex - buffer->currentZStart;
	u32 bin = (2 * bufferOffset) / buffer->capacity;
	bufferOffset -= (bin * (buffer->capacity / 2));

	u08 *result = buffer->buffers[bin]->image + (bufferOffset * buffer->fullImageVolume.dims.x * buffer->fullImageVolume.dims.y);

	if (bin)
	{
		buffer->currentZStart += (buffer->capacity / 2);
		
		image_data *tmp;
		tmp = buffer->buffers[0];
		buffer->buffers[0] = buffer->buffers[1];
		buffer->buffers[1] = tmp;

		buffer->slabHistograms = buffer->slabHistograms->next;

		FillSingleBuffer_threaded(buffer->buffers[1], buffer->loader, buffer->slabHistograms->next->slabHistogram, buffer->converter);
	}

	return(result);
}

void
FillNewHistograms(interpolation_histograms *interpolationHistograms, rolling_image_buffer *imageBuffer)
{
	three_d_volume *imVol = GetImageVolume_FromImageBuffer(imageBuffer);

	u32 atStart = interpolationHistograms->slabCounter == 0;
	u32 atOne = interpolationHistograms->slabCounter == 1;
	u32 atEnd = interpolationHistograms->slabCounter == IntDivideCeil(imVol->dims.z, interpolationHistograms->slabHistograms->slabHistogram->histVol->dims.z);

	++interpolationHistograms->slabCounter;

	if (atStart)
	{
	
	}
	else if (atOne)
	{
		interpolationHistograms->doInterpolation = 1;
	}
	else
	{
		if (atEnd)
		{
			interpolationHistograms->doInterpolation = 0;
		}

		interpolationHistograms->slabHistograms = interpolationHistograms->slabHistograms->next;
	}
}

void
CreateInterpolationHistograms(interpolation_histograms *interpolationHistograms, u32 x, u32 y, three_d_volume *histVol, dim3 histDims)
{
	CreateThreeDVol_FromDim(histVol, histDims);
	
	uint2 slabDims;
	slabDims.x = IntDivideCeil(x, histVol->dims.x);
	slabDims.y = IntDivideCeil(y, histVol->dims.y);

	InitialiseInterpolationHistograms(interpolationHistograms, slabDims, histVol);	
}

void
CalcResultImage(rolling_image_buffer *imInBuffer, image_data *imDataOut, interpolation_histograms *interpolationHistograms, u32 currentZ, f32 interpDis)
{
	u08 *imIn = GetImagePlane(imInBuffer, currentZ);
	u08 *imOut = imDataOut->image;
	
	u32 *histograms_1 = interpolationHistograms->slabHistograms->slabHistogram->imageHistograms;
	u32 *histograms_2;
	f32 interpDis_local;
	if (interpolationHistograms->doInterpolation)
	{
		histograms_2 = interpolationHistograms->slabHistograms->next->slabHistogram->imageHistograms;
		interpDis_local = interpDis;
	}
	else
	{
		histograms_2 = histograms_1;
		interpDis_local = 0;
	}

	three_d_volume *imVol = GetImageVolume_FromImageBuffer(imInBuffer);
	dim3 imageDims = imVol->dims;
	imageDims.z = 1;
	dim3 localHistogramGridDims;
	localHistogramGridDims.x = interpolationHistograms->slabHistograms->slabHistogram->slabDims.x;
	localHistogramGridDims.y = interpolationHistograms->slabHistograms->slabHistogram->slabDims.y;
	localHistogramGridDims.z = 1;

	LaunchCudaKernel_Simple_Stream(AHE, *Processing_Stream, imIn, imOut, histograms_1, histograms_2, interpDis_local, imageDims, localHistogramGridDims, interpolationHistograms->slabHistograms->slabHistogram->histVol->dims);
	cudaStreamSynchronize(*Processing_Stream);
}


void *
ReadingThreadFunc(void *dataIn)
{	
	reading_thread_data_in *readingThreadDataIn;
	readingThreadDataIn = (reading_thread_data_in *)dataIn;

	while (*ReadingThreadRunSignal)
	{
		LockMutex(ReadingThread_Mutex);
		FenceIn(*ReadingThreadContinueSignal = 0);
		WaitOnCond(ReadingThread_Cond, ReadingThread_Mutex);
		UnlockMutex(ReadingThread_Mutex);
		
		if (*ReadingThreadRunSignal)
		{
			FillSingleBuffer(readingThreadDataIn->im, readingThreadDataIn->loader, readingThreadDataIn->hist, readingThreadDataIn->converter);
		}
	}

	return(NULL);
}

struct
write_buffer_data
{
	image_data *im;
	u32 threadIndex;
	memory_arena *zlibArena_comp;
	SSIF_file_for_writing *SSIFfile_writing;
	u08 *compBuffer;
	image_file_coords coordsToWrite;
};

void
InitialiseWriteBufferData(write_buffer_data *data, u32 threadIndex, memory_arena *zlibArena_comp, SSIF_file_for_writing *SSIFfile_writing, u08 *compBuffer)
{
	data->threadIndex = threadIndex;
	data->zlibArena_comp = zlibArena_comp;
	data->SSIFfile_writing = SSIFfile_writing;
	data->compBuffer = compBuffer;
}

struct
write_buffer_node
{
	write_buffer_data *data;
	write_buffer_node *next;
	write_buffer_node *prev;
};

struct
write_buffer
{
	threadSig nFreeNodes;
	write_buffer_node *bufferHead;
};

write_buffer *WriteBuffer;
CreateMutex(WriteBufferMutex);

write_buffer_node *SavingThread_DataIn[NSavingThreads];
threadSig *SavingThreadContinueSignal[NSavingThreads];
threadSig *SavingThreadRunSignal[NSavingThreads];

void
WriteBuffer_InsertAfter(write_buffer_node *head, write_buffer_node *tail)
{
	tail->next = head->next;
	tail->next->prev = tail;
	head->next = tail;
	tail->prev = head;
}

void
WriteBuffer_InsertBefore(write_buffer_node *head, write_buffer_node *tail)
{
	tail->prev->next = head;
	head->prev = tail->prev;
	head->next = tail;
	tail->prev = head;
}

void
WriteBuffer_Remove(write_buffer_node *node)
{
	node->prev->next = node->next;
	node->next->prev = node->prev;
}

void
WriteBuffer_InsertAtStart(write_buffer *buff, write_buffer_node *node)
{
	WriteBuffer_InsertAfter(buff->bufferHead, node);
}

write_buffer_node*
WriteBuffer_RemoveFromEnd(write_buffer *buff)
{
	write_buffer_node *node = buff->bufferHead->prev;
	WriteBuffer_Remove(node);

	return(node);
}

void
SaveResultImage_threaded(image_data *im, image_file_coords coords, u32 threadIndex)
{
	while (*SavingThreadContinueSignal[threadIndex]) {}
	FenceIn(*SavingThreadContinueSignal[threadIndex] = 1);
	LockMutex(SavingThread_Mutex[threadIndex]);
	SignalCondition(SavingThread_Cond[threadIndex]);
		
	SavingThread_DataIn[threadIndex]->data->im = im;
	SavingThread_DataIn[threadIndex]->data->coordsToWrite = coords;
	UnlockMutex(SavingThread_Mutex[threadIndex]);
}

image_data *
AddDataToWriteBuffer(write_buffer *buff, image_data *im, image_file_coords coords)
{
	while(buff->nFreeNodes == 0) {}

	LockMutex(WriteBufferMutex);
	write_buffer_node *node = WriteBuffer_RemoveFromEnd(buff);
	image_data *freeBuffer = node->data->im;
	--buff->nFreeNodes;
	UnlockMutex(WriteBufferMutex);

	SaveResultImage_threaded(im, coords, node->data->threadIndex);

	return(freeBuffer);
}

void
SaveResultImage(write_buffer_data *data)
{
	while(!WriteImageToSSIFFile(data->zlibArena_comp, data->SSIFfile_writing, data->im->image, data->im->nBytes, data->compBuffer, &data->coordsToWrite)) {}
}

void *
SavingThreadFunc(void *dataIn)
{
	write_buffer_node *savingThreadDataIn;
	savingThreadDataIn = (write_buffer_node *)dataIn;

	while (*SavingThreadRunSignal[savingThreadDataIn->data->threadIndex])
	{
		LockMutex(SavingThread_Mutex[savingThreadDataIn->data->threadIndex]);
		FenceIn(*SavingThreadContinueSignal[savingThreadDataIn->data->threadIndex] = 0);
		WaitOnCond(SavingThread_Cond[savingThreadDataIn->data->threadIndex], SavingThread_Mutex[savingThreadDataIn->data->threadIndex]);
		UnlockMutex(SavingThread_Mutex[savingThreadDataIn->data->threadIndex]);
		
		if (*SavingThreadRunSignal[savingThreadDataIn->data->threadIndex])
		{
			SaveResultImage(savingThreadDataIn->data);
			LockMutex(WriteBufferMutex);
			WriteBuffer_InsertAtStart(WriteBuffer, savingThreadDataIn);
			++WriteBuffer->nFreeNodes;
			UnlockMutex(WriteBufferMutex);
		}
	}

	return(NULL);
}

void
ShutDownReadingThread()
{
	while (*ReadingThreadContinueSignal) {}
	FenceIn(*ReadingThreadContinueSignal = 1);
	LockMutex(ReadingThread_Mutex);
	SignalCondition(ReadingThread_Cond);
	
	FenceIn(*ReadingThreadRunSignal = 0);

	UnlockMutex(ReadingThread_Mutex);
}

void
ShutDownSavingThread(u32 index)
{
	while (*SavingThreadContinueSignal[index]) {}
	FenceIn(*SavingThreadContinueSignal[index] = 1);
	LockMutex(SavingThread_Mutex[index]);
	SignalCondition(SavingThread_Cond[index]);
	
	FenceIn(*SavingThreadRunSignal[index] = 0);

	UnlockMutex(SavingThread_Mutex[index]);
}

void
CreateWriteBuffer(write_buffer *buff)
{
	buff->bufferHead->next = buff->bufferHead->prev = buff->bufferHead;

	buff->nFreeNodes = NSavingThreads;
}

void
AddNodesToWriteBuffer(write_buffer *buff)
{
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		WriteBuffer_InsertAtStart(buff, SavingThread_DataIn[index]);
	}
}

SSIF_file_for_writing *
CreateOutputFile(memory_arena *arena, image_loader *loader, program_arguments *pArgs)
{
	char *inputFile = pArgs->inputFile;
	
	SSIF_file_header *newHeader = PushStructP(arena, SSIF_file_header);
	u32 nameLen = CopySSIFHeader(loader->SSIFfile->SSIFfile->header, newHeader, loader->SSIFfile->SSIFfile->header->name);
	CopySSIFHeaderName((char *)"_contrastEqualised\0", newHeader->name + nameLen, nameLen);

	newHeader->depth = (u32)pArgs->zRange.end + 1 - (u32)pArgs->zRange.start;
	newHeader->timepoints = (u32)pArgs->tRange.end + 1 - (u32)pArgs->tRange.start;
	newHeader->channels = (u32)pArgs->cRange.end + 1 - (u32)pArgs->cRange.start;
	newHeader->bytesPerPixel = 1; // grey scale 8 bit image

	switch (pArgs->track)
	{
		case track_depth:
			{
				newHeader->packingOrder = ZTC;		
			} break;

		case track_timepoint:
			{
				newHeader->packingOrder = TZC;
			} break;

		case track_channel:
			{
				newHeader->packingOrder = CZT;
			} break;
	}

	char buff[128];
	u32 fileNameLength = CopyNullTerminatedString(inputFile, buff);
	CopyNullTerminatedString((char *)"_contrastEqualised.ssif\0", buff + fileNameLength - 5);

	SSIF_file_for_writing *SSIFfile_writing = OpenSSIFFileForWriting(arena, newHeader, buff);

	return(SSIFfile_writing);
}

u32
ParseInputParams(program_arguments *programArguments, u32 nArgs, const char **args)
{
	u32 sucsess = 0;
	
	u32 histDimSet = 0;
	u32 zRangeSet = 0;
	u32 tRangeSet = 0;
	u32 cRangeSet = 0;
	u32 trackSet = 0; 

	CopyNullTerminatedString((char *)*args, programArguments->inputFile);
	
	for (	u32 index = 1;
		index < nArgs;
		++index )
	{
		if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--histDims\0"))
		{
			histDimSet = 1;
			string_to_int_result parse_x = StringToInt((char *)*(args + index + 1));		
			string_to_int_result parse_y = StringToInt((char *)*(args + index + 2));
			string_to_int_result parse_z = StringToInt((char *)*(args + index + 3));
			programArguments->histDims.x = parse_x.integerValue;
			programArguments->histDims.y = parse_y.integerValue;
			programArguments->histDims.z = parse_z.integerValue;

			index += 3;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--zRange\0"))
		{
			zRangeSet = 1;
			string_to_int_result parse_zRange_start = StringToInt((char *)*(args + index + 1));
			string_to_int_result parse_zRange_end = StringToInt((char *)*(args + index + 2));
			programArguments->zRange.start = (s32)parse_zRange_start.integerValue;
			programArguments->zRange.end = (s32)parse_zRange_end.integerValue;

			index += 2;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--tRange\0"))
		{
			tRangeSet = 1;
			string_to_int_result parse_tRange_start = StringToInt((char *)*(args + index + 1));
			string_to_int_result parse_tRange_end = StringToInt((char *)*(args + index + 2));
			programArguments->tRange.start = (s32)parse_tRange_start.integerValue;
			programArguments->tRange.end = (s32)parse_tRange_end.integerValue;

			index += 2;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--cRange\0"))
		{
			cRangeSet = 1;
			string_to_int_result parse_cRange_start = StringToInt((char *)*(args + index + 1));
			string_to_int_result parse_cRange_end = StringToInt((char *)*(args + index + 2));
			programArguments->cRange.start = (s32)parse_cRange_start.integerValue;
			programArguments->cRange.end = (s32)parse_cRange_end.integerValue;

			index += 2;
		}
		else if (AreNullTerminatedStringsEqual((char *)*(args + index), (char *)"--track\0"))
		{
			if (AreNullTerminatedStringsEqual((char *)*(args + index + 1), (char *)"z\0"))
			{
				trackSet = 1;
				programArguments->track = track_depth;
			}
			else if(AreNullTerminatedStringsEqual((char *)*(args + index + 1), (char *)"t\0"))
			{
				trackSet = 1;
				programArguments->track = track_timepoint;
			}
			else if(AreNullTerminatedStringsEqual((char *)*(args + index + 1), (char *)"c\0"))
			{
				trackSet = 1;
				programArguments->track = track_channel;
			}

			index += 1;
		}
	}

	sucsess = histDimSet; // must set histDims

	if (!zRangeSet)
	{
		programArguments->zRange.E = {0, -1};
	}
	if (!tRangeSet)
	{
		programArguments->tRange.E = {0, -1};
	}
	if (!cRangeSet)
	{
		programArguments->cRange.E = {0, -1};
	}

	if (!trackSet)
	{
		programArguments->track = track_timepoint;
	}

	return(sucsess);
}

MainArgs
{
	memory_arena arena;
	CreateMemoryArena(arena, MegaByte(4 * (NSavingThreads + 1)));

	program_arguments *pArgs = PushStruct(arena, program_arguments);
	pArgs->inputFile = PushArray(arena, char, 128);
	if (!ParseInputParams(pArgs, ArgCount - 1, ArgBuffer + 1))
	{
		printf("--histDims must be supplied\n");
		return(1);
	}

	Reading_Stream = PushStruct(arena, stream);
	Processing_Stream = PushStruct(arena, stream);
	cudaStreamCreate(Reading_Stream);
	cudaStreamCreate(Processing_Stream);
	
	ReadingThreadContinueSignal = PushStruct(arena, threadSig);
	ReadingThreadRunSignal = PushStruct(arena, threadSig);	
	FenceIn(*ReadingThreadContinueSignal = 1);
	FenceIn(*ReadingThreadRunSignal = 1);
	ReadingThread_DataIn = PushStruct(arena, reading_thread_data_in);
	ReadingThread_Mutex = PushStruct(arena, mutex);
	ReadingThread_Cond = PushStruct(arena, cond);
	InitialiseMutex(ReadingThread_Mutex);
	InitialiseCond(ReadingThread_Cond);
	Reading_Thread = PushStruct(arena, thread);
	LaunchThread(Reading_Thread, ReadingThreadFunc, (void *)ReadingThread_DataIn);
	
	image_loader *imageLoader = PushStruct(arena, image_loader);
	
	memory_arena *zlibDecompArena = PushSubArena(arena, KiloByte(64));
	CreateImageLoader(&arena, zlibDecompArena, imageLoader, pArgs);

	SSIF_file_for_writing *SSIFfile_writing = CreateOutputFile(&arena, imageLoader, pArgs);

	WriteBuffer = PushStruct(arena, write_buffer);
	write_buffer_node *head = PushStruct(arena, write_buffer_node);
	WriteBuffer->bufferHead = head;
	CreateWriteBuffer(WriteBuffer);
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		SavingThread_DataIn[index] = PushStruct(arena, write_buffer_node);
		
		write_buffer_data *writeBufferData = PushStruct(arena, write_buffer_data);
		
		memory_arena *zlibCompArena = PushSubArena(arena, KiloByte(512));
		u08 *compBuffer = PushArray(arena, u08, 2 * GetSSIFPixelsPerImage(imageLoader->SSIFfile->SSIFfile));

		InitialiseWriteBufferData(writeBufferData, index, zlibCompArena, SSIFfile_writing, compBuffer);
		SavingThread_DataIn[index]->data = writeBufferData;

		SavingThreadContinueSignal[index] = PushStruct(arena, threadSig);
		SavingThreadRunSignal[index] = PushStruct(arena, threadSig);
		FenceIn(*SavingThreadContinueSignal[index] = 1);
		FenceIn(*SavingThreadRunSignal[index] = 1);
		SavingThread_Mutex[index] = PushStruct(arena, mutex);
		SavingThread_Cond[index] = PushStruct(arena, cond);
		InitialiseMutex(SavingThread_Mutex[index]);
		InitialiseCond(SavingThread_Cond[index]);
		Saving_Thread[index] = PushStruct(arena, thread);
		LaunchThread(Saving_Thread[index], SavingThreadFunc, (void *)SavingThread_DataIn[index]);
	}
	AddNodesToWriteBuffer(WriteBuffer);

	WriteBufferMutex = PushStruct(arena, mutex);
	InitialiseMutex(WriteBufferMutex);

	u32 NSavingBuffers = NSavingThreads + 1;
	image_data *result = PushArray(arena, image_data, NSavingBuffers);
	for (	u32 index = 0;
		index < NSavingBuffers;
		++index )
	{
		CreateImageData_FromInt(result + index, ImageLoaderGetWidth(imageLoader), ImageLoaderGetHeight(imageLoader), 1);
	}
	u32 currentResultIndex = 0;
	image_data *currentResultBuffer, *freeBuffer;

	interpolation_histograms *interpolationHistograms = PushStruct(arena, interpolation_histograms);
	slab_histograms *slabHistograms = PushArray(arena, slab_histograms, 3);
	slab_histograms_LL_node *nodes = PushArray(arena, slab_histograms_LL_node, 3);

	nodes->slabHistogram = slabHistograms;
	(nodes + 1)->slabHistogram = slabHistograms + 1;
	(nodes + 2)->slabHistogram = slabHistograms + 2;

	interpolationHistograms->slabHistograms = nodes;
	three_d_volume *histVol = PushStruct(arena, three_d_volume);
	CreateInterpolationHistograms(interpolationHistograms, ImageLoaderGetWidth(imageLoader), ImageLoaderGetHeight(imageLoader), histVol, pArgs->histDims);

	InitialiseSlabHistogramLLNode(nodes + 1, interpolationHistograms->slabHistograms->slabHistogram->slabDims, histVol);
	InitialiseSlabHistogramLLNode(nodes + 2, interpolationHistograms->slabHistograms->slabHistogram->slabDims, histVol);

	SlabHistogramsLLAddAfter(nodes, nodes + 1);
	SlabHistogramsLLAddAfter(nodes + 1, nodes + 2);
	SlabHistogramsLLAddAfter(nodes + 2, nodes);

	u32 localZ = histVol->dims.z;
	u32 halfLocalZ = localZ / 2;

	image_data *inputIm = PushArray(arena, image_data, 2);
	rolling_image_buffer *imageBuffer = PushStruct(arena, rolling_image_buffer);
	imageBuffer->slabHistograms = nodes;
	
	to_eight_bit_converter *converter = PushStruct(arena, to_eight_bit_converter);
	CreateStream(converterStream);
	converterStream = PushStruct(arena, stream);
	cudaStreamCreate(converterStream);
	CreateToEightBitConverter(converter, GetSSIFPixelsPerImage(imageLoader->SSIFfile->SSIFfile), converterStream, GetSSIFBytesPerPixel(imageLoader->SSIFfile->SSIFfile));

	CreateRollingImageBuffer(imageBuffer, imageLoader, inputIm, inputIm + 1, localZ, converter);
	
	u32 nImagesTrack;
	u32 *startingIndex = GetImageLoaderCurrentTrackIndexAndSize(imageLoader, &nImagesTrack);
	nImagesTrack -= *startingIndex;

	u32 nImagesToLoad = (ImageLoaderGetChannels(imageLoader) - imageLoader->channel) * (ImageLoaderGetTimePoints(imageLoader) - imageLoader->timepoint) * (ImageLoaderGetDepth(imageLoader) - imageLoader->depth);	

	u32 currentInterpStart;

	for (	u32 index = 0;
		index < nImagesToLoad;
		++index )
	{
		u32 localIndex = index % nImagesTrack;
		
		if (localIndex == 0)
		{
			if (index)
			{
				switch (imageLoader->track)
				{
					case track_depth:
						{
							imageLoader->depth = (u32)pArgs->zRange.start;
						} break;

					case track_timepoint:
						{
							imageLoader->timepoint = (u32)pArgs->tRange.start;
						} break;

					case track_channel:
						{
							imageLoader->channel = (u32)pArgs->cRange.start;
						} break;
				}
				
				switch (SSIFfile_writing->SSIFfile->header->packingOrder)
				{
					case ZTC:
						{
							if (++imageLoader->timepoint == ImageLoaderGetTimePoints(imageLoader))
							{
								imageLoader->timepoint = (u32)pArgs->tRange.start;
								++imageLoader->channel;
							}
						} break;
					
					case CTZ:
						{
							if (++imageLoader->timepoint == ImageLoaderGetTimePoints(imageLoader))
							{
								imageLoader->timepoint = (u32)pArgs->tRange.start;
								++imageLoader->depth;
							}
						} break;
					
					case TZC:
						{
							if (++imageLoader->depth == ImageLoaderGetDepth(imageLoader))
							{
								imageLoader->depth = (u32)pArgs->zRange.start;
								++imageLoader->channel;
							}
						} break;
					
					case CZT:
						{
							if (++imageLoader->depth == ImageLoaderGetDepth(imageLoader))
							{
								imageLoader->depth = (u32)pArgs->zRange.start;
								++imageLoader->timepoint;
							}
						} break;

					case ZCT:
						{
							if (++imageLoader->channel == ImageLoaderGetChannels(imageLoader))
							{
								imageLoader->channel = (u32)pArgs->cRange.start;
								++imageLoader->timepoint;
							}
						} break;
					
					case TCZ:
						{
							if (++imageLoader->channel == ImageLoaderGetChannels(imageLoader))
							{
								imageLoader->channel = (u32)pArgs->cRange.start;
								++imageLoader->depth;
							}
						} break;
				}
			}
			
			currentInterpStart = 0;
			
			interpolationHistograms->slabHistograms = nodes;
			interpolationHistograms->slabCounter = 0;
			interpolationHistograms->doInterpolation = 0;
			
			imageBuffer->slabHistograms = nodes;
			imageBuffer->currentZStart = 0;

			FillRollingBuffer(imageBuffer);
			FillNewHistograms(interpolationHistograms, imageBuffer);
		}
		
		if (currentResultIndex < NSavingBuffers)
		{
			currentResultBuffer = result + currentResultIndex++;
		}
		else
		{
			currentResultBuffer = freeBuffer;
		}
	
		if ((localIndex >= halfLocalZ) && (((localIndex - halfLocalZ) % localZ) == 0))
		{
			FillNewHistograms(interpolationHistograms, imageBuffer);
			currentInterpStart = localIndex;
		}

		f32 interpDis = (f32)(localIndex - currentInterpStart) / (f32)localZ;
		CalcResultImage(imageBuffer, currentResultBuffer, interpolationHistograms, localIndex, interpDis);

		image_file_coords coords = LinearIndexToImageCoords(index, SSIFfile_writing->SSIFfile->header);

		freeBuffer = AddDataToWriteBuffer(WriteBuffer, currentResultBuffer, coords);
		
		printf("\r%1.2f%% complete...", 100.0*(f32)(index + 1)/(f32)nImagesToLoad);
		fflush(stdout);
	}
	
	ShutDownReadingThread();
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		ShutDownSavingThread(index);
	}
	WaitForThread(Reading_Thread);
	for (	u32 index = 0;
		index < NSavingThreads;
		++index )
	{
		WaitForThread(Saving_Thread[index]);
	}

	FreeRollingImageBuffer(imageBuffer);
	for (	u32 index = 0;
		index < NSavingBuffers;
		++index )
	{
		FreeImageData(result + index);
	}
	CloseSSIFFile(SSIFfile_writing->SSIFfile);
	FreeSlabHistogramLLNode(nodes);
	FreeSlabHistogramLLNode(nodes + 1);
	FreeSlabHistogramLLNode(nodes + 2);
	FreeMemoryArena(arena);

	EndMain;
}
