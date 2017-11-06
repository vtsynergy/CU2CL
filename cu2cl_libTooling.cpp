/*
* CU2CL - A prototype CUDA-to-OpenCL translator built on the Clang compiler infrastructure
* Version 0.8.0b (beta)
*
* (c) 2010-2017 Virginia Tech
*
*    This library is free software; you can redistribute it and/or modify it under the terms of the attached GNU Lesser General Public License v2.1 as published by the Free Software Foundation.
*
*    This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
*
*   You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
* 
* Authors: Paul Sathre, Gabriel Martinez
*
*/

#define CU2CL_LICENSE \
	"/* (c) 2010-2017 Virginia Tech\n" \
	"*\n" \
	"*    This library is free software; you can redistribute it and/or modify it under the terms of the attached GNU Lesser General Public License v2.1 as published by the Free Software Foundation.\n" \
	"*\n" \
	"*    This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.\n" \
	"*\n" \
	"*   You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA \n" \
	"*/\n" 
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
//Added to fix CUDA attributes being undeclared
#include "clang/AST/Attr.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

//Added during the libTooling conversion
#include "clang/Driver/Options.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
//Added during the libTooling conversion
#include "clang/Frontend/FrontendActions.h"

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"

#include "clang/Rewrite/Core/Rewriter.h"

//Added during the libTooling conversion
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
//Support the RefactoringTool class
#include "clang/Tooling/Refactoring.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/CommandLine.h"

#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <iostream>
#include <cstdio>
#include <memory>

//Injects a small amount of code to time the translation process
#define CU2CL_ENABLE_TIMING

#ifdef CU2CL_ENABLE_TIMING
#include <sys/time.h>
#endif

/*
* The following macros define data structures, functions, and kernels
*  that make up a "CU2CL Runtime", providing synthesized analogues of
*  CUDA features, that do not have native equivalences in OpenCL.
*/

//A scaffold for supporting as much of cudaDeviceProp as possible
#define CL_DEVICE_PROP \
    "struct __cu2cl_DeviceProp {\n" \
    "    char name[256];\n" \
    "    cl_ulong totalGlobalMem;\n" \
    "    cl_ulong sharedMemPerBlock;\n" \
    "    cl_uint regsPerBlock;\n" \
    "    cl_uint warpSize;\n" \
    "    size_t memPitch; //Unsupported!\n" \
    "    size_t maxThreadsPerBlock;\n" \
    "    size_t maxThreadsDim[3];\n" \
    "    int maxGridSize[3]; //Unsupported!\n" \
    "    cl_uint clockRate;\n" \
    "    size_t totalConstMem; //Unsupported!\n" \
    "    cl_uint major;\n" \
    "    cl_uint minor;\n" \
    "    size_t textureAlignment; //Unsupported!\n" \
    "    cl_bool deviceOverlap;\n" \
    "    cl_uint multiProcessorCount;\n" \
    "    cl_bool kernelExecTimeoutEnabled;\n" \
    "    cl_bool integrated;\n" \
    "    int canMapHostMemory; //Unsupported!\n" \
    "    int computeMode; //Unsupported!\n" \
    "    int maxTexture1D; //Unsupported!\n" \
    "    int maxTexture2D[2]; //Unsupported!\n" \
    "    int maxTexture3D[3]; //Unsupported!\n" \
    "    int maxTexture2DArray[3]; //Unsupported!\n" \
    "    size_t surfaceAlignment; //Unsupported!\n" \
    "    int concurrentKernels; //Unsupported!\n" \
    "    cl_bool ECCEnabled;\n" \
    "    int pciBusID; //Unsupported!\n" \
    "    int pciDeviceID; //Unsupported!\n" \
    "    int tccDriver; //Unsupported!\n" \
    "    //int __cudaReserved[21];\n" \
    "};\n\n"

//Encapsulation for reading a .cl kernel file at runtime
#define LOAD_PROGRAM_SOURCE_H \
    "size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc);\n"

#define LOAD_PROGRAM_SOURCE \
    "size_t __cu2cl_LoadProgramSource(const char *filename, const char **progSrc) {\n" \
    "    FILE *f = fopen(filename, \"r\");\n" \
    "    fseek(f, 0, SEEK_END);\n" \
    "    size_t len = (size_t) ftell(f);\n" \
    "    *progSrc = (const char *) malloc(sizeof(char)*len);\n" \
    "    rewind(f);\n" \
    "    fread((void *) *progSrc, len, 1, f);\n" \
    "    fclose(f);\n" \
    "    return len;\n" \
    "}\n\n"

//The host-side portion of a kernel to emulate the behavior of cudaMemset
#define CL_MEMSET_H \
    "cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count);\n"

#define CL_MEMSET \
    "cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count) {\n" \
    "    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 0, sizeof(cl_mem), &devPtr);\n" \
    "    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 1, sizeof(cl_uchar), &value);\n" \
    "    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 2, sizeof(cl_uint), &count);\n" \
    "    globalWorkSize[0] = count;\n" \
    "    return clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel___cu2cl_Memset, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);\n" \
    "}\n\n"

//The device-side kernel that emulates the behavior of cudaMemset
#define CL_MEMSET_KERNEL \
    "__kernel void __cu2cl_Memset(__global uchar *ptr, uchar value, uint num) {\n" \
    "    size_t id = get_global_id(0);\n" \
    "    if (get_global_id(0) < num) {\n" \
    "        ptr[id] = value;\n" \
    "    }\n" \
    "}\n\n"

//A stub to query a specific property in __cu2cl_DeviceProp
// can be used independently of CL_GET_DEVICE_PROPS, but is not intended
#define CL_GET_DEVICE_INFO(TYPE, NAME) \
    "    ret |= clGetDeviceInfo(device, CL_DEVICE_" #TYPE ", sizeof(prop->" \
    #NAME "), &prop->" #NAME ", NULL);\n"

//A function to query the OpenCL properties which have direct analogues in cudaDeviceProp
#define CL_GET_DEVICE_PROPS_H \
    "cl_int __cu2cl_GetDeviceProperties(struct __cu2cl_DeviceProp * prop, cl_device_id device);\n"

#define CL_GET_DEVICE_PROPS \
    "cl_int __cu2cl_GetDeviceProperties(struct __cu2cl_DeviceProp *prop, cl_device_id device) {\n" \
    "    cl_int ret = CL_SUCCESS;\n" \
    CL_GET_DEVICE_INFO(NAME, name) \
    CL_GET_DEVICE_INFO(GLOBAL_MEM_SIZE, totalGlobalMem) \
    CL_GET_DEVICE_INFO(LOCAL_MEM_SIZE, sharedMemPerBlock) \
    CL_GET_DEVICE_INFO(REGISTERS_PER_BLOCK_NV, regsPerBlock) \
    CL_GET_DEVICE_INFO(WARP_SIZE_NV, warpSize) \
    CL_GET_DEVICE_INFO(MAX_WORK_GROUP_SIZE, maxThreadsPerBlock) \
    CL_GET_DEVICE_INFO(MAX_WORK_ITEM_SIZES, maxThreadsDim) \
    CL_GET_DEVICE_INFO(MAX_CLOCK_FREQUENCY, clockRate) \
    CL_GET_DEVICE_INFO(COMPUTE_CAPABILITY_MAJOR_NV, major) \
    CL_GET_DEVICE_INFO(COMPUTE_CAPABILITY_MINOR_NV, minor) \
    CL_GET_DEVICE_INFO(GPU_OVERLAP_NV, deviceOverlap) \
    CL_GET_DEVICE_INFO(MAX_COMPUTE_UNITS, multiProcessorCount) \
    CL_GET_DEVICE_INFO(KERNEL_EXEC_TIMEOUT_NV, kernelExecTimeoutEnabled) \
    CL_GET_DEVICE_INFO(INTEGRATED_MEMORY_NV, integrated) \
    CL_GET_DEVICE_INFO(ERROR_CORRECTION_SUPPORT, ECCEnabled) \
    "    return ret;\n" \
    "}\n\n"

//A function to check the status of the command queue, emulating cudaStreamQuery
#define CL_COMMAND_QUEUE_QUERY_H \
    "cl_int __cu2cl_CommandQueueQuery(cl_command_queue commands);\n"

#define CL_COMMAND_QUEUE_QUERY \
    "cl_int __cu2cl_CommandQueueQuery(cl_command_queue commands) {\n" \
    "   cl_int ret;\n" \
    "   cl_event event;\n" \
    "   clEnqueueMarker(commands, &event);\n" \
    "   clGetEventInfo(commands, &event);\n" \
    "}\n\n"

//A function to take the time between two events, emulating cudaEventElapsedTime
#define CL_EVENT_ELAPSED_TIME_H \
    "cl_int __cu2cl_EventElapsedTime(float *ms, cl_event start, cl_event end);\n"

#define CL_EVENT_ELAPSED_TIME \
    "cl_int __cu2cl_EventElapsedTime(float *ms, cl_event start, cl_event end) {\n" \
    "    cl_int ret;\n" \
    "    cl_ulong s, e;\n" \
    "    float fs, fe;\n" \
    "    ret |= clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &s, NULL);\n" \
    "    ret |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL);\n" \
    "    s = e - s;\n" \
    "    *ms = ((float) s)/1000000.0;\n" \
    "    return ret;\n" \
    "}\n\n"

//A function to check whether the command queue has hit an injected event yet, emulating cudaEventQuery
#define CL_EVENT_QUERY_H \
    "cl_int __cu2cl_EventQuery(cl_event event);\n"

#define CL_EVENT_QUERY \
    "cl_int __cu2cl_EventQuery(cl_event event) {\n" \
    "    cl_int ret;\n" \
    "    clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL);\n" \
    "    return ret;\n" \
    "}\n\n"

//A function to emulate the behavior (not necessarily semantics) of cudaMallocHost
// allocates a device buffer, then maps it into the host address space, and returns a pointer to it
#define CL_MALLOC_HOST_H \
    "cl_int __cu2cl_MallocHost(void **ptr, size_t size, cl_mem *clMem);\n"

#define CL_MALLOC_HOST \
    "cl_int __cu2cl_MallocHost(void **ptr, size_t size, cl_mem *clMem) {\n" \
    "    cl_int ret;\n" \
    "    *clMem = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);\n" \
    "    *ptr = clEnqueueMapBuffer(__cu2cl_CommandQueue, *clMem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);\n" \
    "    return ret;\n" \
    "}\n\n"

//A function to emulate the behavior (not necessarily semantics) of cudaFreeHost
// unmaps a buffer allocated with __cu2cl_MallocHost, then releases the associated device buffer
#define CL_FREE_HOST_H \
    "cl_int __cu2cl_FreeHost(void *ptr, cl_mem clMem);\n"

#define CL_FREE_HOST \
    "cl_int __cu2cl_FreeHost(void *ptr, cl_mem clMem) {\n" \
    "    cl_int ret;\n" \
    "    ret = clEnqueueUnmapMemObject(__cu2cl_CommandQueue, clMem, ptr, 0, NULL, NULL);\n" \
    "    ret |= clReleaseMemObject(clMem);\n" \
    "    return ret;\n" \
    "}\n\n"

//A helper function to scan all platforms for all devices and accumulate them into a single array
// can be used independently of __cu2cl_setDevice, but not intended
#define CU2CL_SCAN_DEVICES_H \
    "void __cu2cl_ScanDevices();\n"

#define CU2CL_SCAN_DEVICES \
    "void __cu2cl_ScanDevices() {\n" \
    "   int i;\n" \
    "   cl_uint num_platforms = 0;\n" \
    "   cl_uint num_devices = 0;\n" \
    "   cl_uint p_dev_count, d_idx;\n" \
    "\n" \
    "   //allocate space for platforms\n" \
    "   clGetPlatformIDs(0, 0, &num_platforms);\n" \
    "   cl_platform_id * platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);\n" \
    "\n" \
    "   //get all platforms\n" \
    "   clGetPlatformIDs(num_platforms, &platforms[0], 0);\n" \
    "\n" \
    "   //count devices over all platforms\n" \
    "   for (i = 0; i < num_platforms; i++) {\n" \
    "       p_dev_count = 0;\n" \
    "       clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, 0, &p_dev_count);\n" \
    "       num_devices += p_dev_count;\n" \
    "   }\n" \
    "\n" \
    "   //allocate space for devices\n" \
    "   __cu2cl_AllDevices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);\n" \
    "\n" \
    "   //get all devices\n" \
    "   d_idx = 0;\n" \
    "   for ( i = 0; i < num_platforms; i++) {\n" \
    "       clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices-d_idx, &__cu2cl_AllDevices[d_idx], &p_dev_count);\n" \
    "       d_idx += p_dev_count;\n" \
    "       p_dev_count = 0;\n" \
    "   }\n" \
    "\n" \
    "   __cu2cl_AllDevices_size = d_idx;\n" \
    "   free(platforms);\n" \
    "}\n\n"

//A function to reset the OpenCL context and queues for the Nth device among all system devices
// uses __cu2cl_ScanDevices to enumerate, and thus uses whatever device ordering it provides
//FIXME: cudaSetDevice preserves the context when switching, ours destroys it, need to modify
// to internally manage and intelligently deconstruct the context(s)
#define CU2CL_SET_DEVICE_H \
    "void __cu2cl_SetDevice(cl_uint devID);\n"

#define CU2CL_SET_DEVICE \
    "void __cu2cl_SetDevice(cl_uint devID) {\n" \
    "   if (__cu2cl_AllDevices_size == 0) {\n" \
    "       __cu2cl_ScanDevices();\n" \
    "   }\n" \
    "   //only switch devices if it's a valid choice\n" \
    "   if (devID < __cu2cl_AllDevices_size) {\n" \
    "       //Assume auto-initialized queue and context, and free them\n" \
    "       clReleaseCommandQueue(__cu2cl_CommandQueue);\n" \
    "       clReleaseContext(__cu2cl_Context);\n" \
    "       //update device and platform references\n" \
    "       __cu2cl_AllDevices_curr_idx = devID;\n" \
    "       __cu2cl_Device = __cu2cl_AllDevices[devID];\n" \
    "       clGetDeviceInfo(__cu2cl_Device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &__cu2cl_Platform, NULL);\n" \
    "       //and make a new context and queue for the selected device\n" \
    "       __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);\n" \
    "       __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);\n" \
    "   }\n" \
    "}\n\n" 

using namespace clang;
using namespace clang::tooling;
using namespace llvm::sys::path;

namespace {
    //Flags used to ensure certain pieces of boilerplate only get added once
    // Hoisted to the Tool level so they can act over all files when generating cu2cl_util.c/h
    bool UsesCUDADeviceProp = false;
    bool UsesCUDAMemset = false;
    bool UsesCUDAStreamQuery = false;
    bool UsesCUDAEventElapsedTime = false;
    bool UsesCUDAEventQuery = false;
    bool UsesCUDAMallocHost = false;
    bool UsesCUDAFreeHost = false;
    bool UsesCUDASetDevice = false;
    bool UsesCU2CLUtilCL = false;
    bool UsesCU2CLLoadSrc = false;

    //internal flags for command-line toggles
    bool AddInlineComments = true; //defaults to ON, turn off with '--inline-comments=false' at the command line
    //Extra Arguments to be appended to all generated clBuildProgram calls.
    std::string ExtraBuildArgs; //defaults to "", add more with '--cl-build-args="<args>"'
    bool FilterKernelName = false; //defaults to OFF, turn on with '--rename-kernel-files' or '--rename-kernel-files=true'

    bool UseGCCPaths = false; //defaults to OFF, turn on with '--import-gcc-paths'
    //We borrow the OutputFile data structure from Clang's CompilerInstance.h
    // So that we can use it to store output streams and emulate their temp
    // file usage at the tool level
    struct OutputFile {
	std::string Filename;
	std::string TempFilename;
	raw_ostream *OS;

	OutputFile(const std::string &filename, const std::string &tempFilename, raw_ostream *os) : Filename(filename), TempFilename(tempFilename), OS(os) { }
    };

    typedef std::map<std::string, std::vector<std::string> > FileStrCacheMap;
    typedef std::map<std::string, OutputFile *> IDOutFileMap;

    //Index structures for looking up all references to a given Decl
    //typedef std::map<Decl *, std::vector<DeclRefExpr *> > DeclToRefMap;
    typedef std::tuple<SourceManager *, Preprocessor *, LangOptions *, ASTContext *> SourceTuple;
    typedef std::map<std::string, std::vector<DeclRefExpr *> > DeclToRefMap;
    typedef std::map<std::string, std::vector<std::pair<FunctionDecl *, SourceTuple *> > > CanonicalFuncDeclMap;
    typedef std::vector<std::pair<NamedDecl*, SourceTuple *> > FlaggedDeclVec;

	bool hasFlaggedDecl(FlaggedDeclVec * vec, NamedDecl * decl) {
		for (FlaggedDeclVec::iterator itr = vec->begin(); itr != vec->end(); itr++){
			if (itr->first == decl) return true;
		}
		return false;
	}

    //Simple Vector to hold retained SourceManagers to use at the tool layer
    typedef std::vector<SourceManager *> SMVec;
    //A simple structure to retain ASTContexts so they can be later used at the tool layer (and appropriately released)
    typedef std::vector<ASTContext *> ASTContVec;
    //Global Replacement structs, contributed to by each instance of the translator (one-per-main-source-file)
    // only written to after local deduplication and coalescing
    std::vector<Replacement> GlobalHostReplace;
    std::vector<Replacement> GlobalKernReplace;

    std::map<SourceLocation, Replacement> GlobalHostVecVars;
    //All ASTContexts get pushed here as their translation units get processed
    // so that their member elements can be referred to after TU processing
    ASTContVec AllASTs;
    SMVec AllSMs;

    //All Declarations and references to them are recorded to propagate cl_mem and other critical rewrites across TU boundaries
    FlaggedDeclVec DeclsToTranslate;
    DeclToRefMap AllDeclRefsByDecl;
    CanonicalFuncDeclMap AllFuncsByCanon;   
 
    //Global outFiles maps, moved so that they can be shared and written to at the tool level
    IDOutFileMap OutFiles;
    IDOutFileMap KernelOutFiles;

    //Global map of declaration statements to the files that own them (all others declare them "extern")
    //Filenames are original (not *-cl.cl/cpp/h) except cu2cl_util.c/h/cl
    FileStrCacheMap GlobalCDecls;
    FileStrCacheMap LocalBoilDefs;

    //Global boilerplate strings
    std::string CU2CLInit;
    std::string CU2CLClean;
    
    std::vector<std::string> GlobalHDecls, GlobalCFuncs, GlobalCLFuncs, UtilKernels;
    //We also borrow the loose method of dealing with temporary output files from
    // CompilerInstance::clearOutputFiles
    void clearOutputFile(OutputFile *OF, FileManager *FM) {
	if(!OF->TempFilename.empty()) {
	    SmallString<128> NewOutFile(OF->Filename);
	    FM->FixupRelativePath(NewOutFile);
	    if (llvm::error_code ec = llvm::sys::fs::rename(OF->TempFilename, NewOutFile.str()))
		llvm::errs() << "Unable to move CU2CL temporary output [" << OF->TempFilename << "] to [" << OF->Filename << "]!\n\t Diag Msg: " << ec.message() << "\n";
	    llvm::sys::fs::remove(OF->TempFilename);
	} else {
	    llvm::sys::fs::remove(OF->Filename);
	}
	delete OF->OS;
    }

    //Replace all instances of the phrase "kernel" with "knl"
    // Used to rename files as per Altera's kernel filename requirement
    std::string kernelNameFilter(std::string str) {
	std::string newStr = str;
	if (!FilterKernelName) return newStr;
	size_t pos = newStr.rfind("/"); //Only rewrite the file, not the path
	if (pos == std::string::npos) pos = 0;
	for (; ; pos += 3) {
	    pos = newStr.find("kernel", pos);
	    if (pos == std::string::npos) break;
	    newStr.erase(pos, 6);
	    newStr.insert(pos, "knl");
	}
	return newStr;
    }


bool isInBannedInclude(SourceLocation loc, SourceManager * SM, LangOptions * LO) {
	SourceLocation sloc = SM->getSpellingLoc(loc);
	//if (loc.isMacroID()) sloc = SM->getSpellingLoc(loc);
        std::string FileName = SM->getPresumedLoc(loc).getFilename();
	//llvm::errs() << "CU2CL DEBUG: " << FileName;

            llvm::StringRef fileExt = extension(FileName);
            if (fileExt.equals(".cu") || fileExt.equals(".cuh")) return false;
	//TODO check if the file was included by any file matching the below criteria	
	if (filename(FileName).equals("cuda.h") || filename(FileName).equals("cuda_runtime.h") || filename(FileName).equals("cuda_runtime_api.h") || filename(FileName).equals("cuda_gl_interop.h") || filename(FileName).equals("cutil.h") || filename(FileName).equals("cutil_inline.h") || filename(FileName).equals("cutil_gl_inline.h") || filename(FileName).equals("vector_types.h") || SM->isInSystemHeader(loc) || SM->isInExternCSystemHeader(loc) || SM->isInSystemMacro(loc) || SM->isInSystemHeader(sloc) || SM->isInExternCSystemHeader(sloc) || SM->isInSystemMacro(sloc)) {
		//it's a forbidden file, just skip the file
		return true;
	}
	SourceLocation parentLoc = SM->getIncludeLoc(SM->getFileID(loc));
	//If the parent of the regular location isn't valid, try the spelling location
	if (!parentLoc.isValid() && loc.isMacroID()) parentLoc = SM->getIncludeLoc(SM->getFileID(sloc));
	if (!parentLoc.isValid()) {
		if (!SM->isInMainFile(loc)) llvm::errs() << "CU2CL DEBUG: " << loc.printToString(*SM) << "\nInvalid parent IncludeLoc\n";
		return false;
	}
	//If the include location is
	//llvm::errs() << "CU2CL DEBUG: Checking parent include from [" << parentLoc.printToString(*SM) << "]\n";
	Token fileTok;
	Lexer::getRawToken(parentLoc, fileTok, *SM, *LO);
//	SourceLocation angleLoc = Lexer::findLocationAfterToken(parentLoc, tok::angle_string_literal, *SM, *LO, true);
//	if (!angleLoc.isValid()) {
	if (!fileTok.is(tok::angle_string_literal) && !fileTok.is(tok::less)) {
		//llvm::errs() << fileTok.getName() << " :Parent is a quote #include!\n";
	
		//As a fallback, try banning based on the parent
		return isInBannedInclude(parentLoc, SM, LO);
	} else {
		//llvm::errs() << "Parent is an angle #include!\n";
		return true;
	}
	
}

//Simple timer calls that get injected if enabled
#ifdef CU2CL_ENABLE_TIMING
	uint64_t TransTime;    
struct timeval startTime, endTime;

void init_time() {
    gettimeofday(&startTime, NULL);
}

uint64_t get_time() {
    gettimeofday(&endTime, NULL);
    return (uint64_t) (endTime.tv_sec - startTime.tv_sec)*1000000 +
        (endTime.tv_usec - startTime.tv_usec);
}
#endif

//Check which of two DeclGroups come first in the source
struct cmpDG {
    bool operator()(DeclGroupRef a, DeclGroupRef b) {
        SourceLocation aLoc = (a.isSingleDecl() ? a.getSingleDecl() : a.getDeclGroup()[0])->getLocStart();
        SourceLocation bLoc = (b.isSingleDecl() ? b.getSingleDecl() : b.getDeclGroup()[0])->getLocStart();
        return aLoc.getRawEncoding() < bLoc.getRawEncoding();
    }   
};

//FIXME: Borrowed verbatim from Clang's Refactoring.cpp
// Just call theirs once we can (for now it's not recognized as a member of the clang::tooling namespace, though it should be
static int getRangeSize(SourceManager &Sources, const CharSourceRange &Range) {
  SourceLocation SpellingBegin = Sources.getSpellingLoc(Range.getBegin());
  SourceLocation SpellingEnd = Sources.getSpellingLoc(Range.getEnd());
  std::pair<FileID, unsigned> Start = Sources.getDecomposedLoc(SpellingBegin);
  std::pair<FileID, unsigned> End = Sources.getDecomposedLoc(SpellingEnd);
  if (Start.first != End.first) return -1;
  if (Range.isTokenRange())
    End.second += Lexer::MeasureTokenLength(SpellingEnd, Sources, LangOptions());
  return End.second - Start.second;
}

//This method is designed to walk a vector of Replacements that has already
// been deduplicated, and fuse Replacments that are enqueued on the same
// start SourceLocation
//\pre replace is sorted in order of increasing SourceLocation
//\pre replace has no duplicate Replacements
//\post replace has no more than one Replacement per SourceLocation
void coalesceReplacements(std::vector<Replacement> &replace) {
	//Must assemble a new vector in-place
	//Swap the input vector with the work vector so we can add replacements directly back as output
	std::vector<Replacement> work;
	work.swap(replace);

	//track the maximum range for a set of Replacements to be fused
	int max;
	//track the concatenated text for a set of Replacements to be fused
	std::stringstream text;
	std::vector<Replacement>::const_iterator J;
	
	//Iterate over every Replacement in the input vector
	for (std::vector<Replacement>::const_iterator I = work.begin(), E = work.end(); I != E; I++) {
	    //reset the max range size and string to match I
	    max = I->getLength();
	    text.str("");
	    text << I->getReplacementText().str();
	    //Look forward at all Replacements at the same location as I
	    for (J = I+1; J !=E && J->getFilePath() == I->getFilePath() && J->getOffset() == I->getOffset(); J++) {
	    	//Check if they cover a longer range, and concatenate changes
		max = (max > J->getLength() ? max : J->getLength());
		text << J->getReplacementText().str();
		//llvm::errs() << "Merging text: " << text.str();
	    }
	    //Add the coalesced Replacement back to the input vector
	    replace.push_back(Replacement(I->getFilePath(), I->getOffset(), max, text.str()));
	    //And finally move the I iterator forward to the last-fused Replacement
	    I = J-1;
	}
}
    void debugPrintReplacements(std::vector<Replacement> replace) {
	for (std::vector<Replacement>::const_iterator I = replace.begin(), E = replace.end(); I != E; I++) {
	    llvm::errs() << I->toString() << "\n";
	}

    }
    //Comments to be injected into source code are buffered until after translation
    // this struct implements a simple list for storing them, but is not meamnt for
    // use outside the bufferComment and writeComments functions
    // l is the SourceLoc pointer
    // s is the string itself
    // w declares whether it's a host (true) or device (false) comment
    //WARNING: Not threadsafe at all!
    struct commentBufferNode;
    struct commentBufferNode {
	void * l;
	char * s;
	std::vector<Replacement> * r;
	struct commentBufferNode * n;
	};
    struct commentBufferNode * tail, * head;

    //Buffer a new comment destined to be added to output OpenCL source files
    //WARNING: Not threadsafe at all!
    void bufferComment(SourceLocation loc, std::string str, std::vector<Replacement> *replacements) {
	struct commentBufferNode * n = (struct commentBufferNode *)malloc(sizeof(commentBufferNode));
	n->s = (char *)malloc(sizeof(char)*(str.length()+1));
	str.copy(n->s, str.length());
	n->s[str.length()] = '\0';
	n->l = loc.getPtrEncoding(); n->r = replacements; n->n = NULL;

	tail->n = n;
	tail = n;
    }


    
    // Workhorse for CU2CL diagnostics, provides independent specification of multiple err_notes
    //  and inline_notes which should be dumped to stderr and translated output, respectively
    // TODO: Eventually this could stand to be implemented using the real Basic/Diagnostic subsystem
    //  but at the moment, the set of errors isn't mature enough to make it worth it.
    // It's just cheaper to directly throw it more readily-adjustable strings until we set the 
    //  error messages in stone.
    void emitCU2CLDiagnostic(SourceManager * SM, SourceLocation loc, std::string severity_str, std::string err_note, std::string inline_note, std::vector<Replacement> * replacements) {
        //Sanitize all incoming locations to make sure they're not MacroIDs
        SourceLocation expLoc = SM->getExpansionLoc(loc);
        SourceLocation writeLoc;

        //assemble both the stderr and inlined source output strings
        std::stringstream inlineStr;
        std::stringstream errStr;
	inlineStr << "/*";
        if (expLoc.isValid()){
	    //Tack the source line information onto the diagnostic
            //inlineStr << SM->getBufferName(expLoc) << ":" << SM->getExpansionLineNumber(expLoc) << ":" << SM->getExpansionColumnNumber(expLoc) << ": ";
            errStr << SM->getBufferName(expLoc) << ":" << SM->getExpansionLineNumber(expLoc) << ":" << SM->getExpansionColumnNumber(expLoc) << ": ";
            //grab the start of column write location
            writeLoc = SM->translateLineCol(SM->getFileID(expLoc), SM->getExpansionLineNumber(expLoc), 1);
        }
	//Inject the severity string to both outputs
        if (!severity_str.empty()) {
            errStr << severity_str << ": ";
            inlineStr << severity_str << " -- ";
        }
        inlineStr << inline_note << "*/\n";
        errStr << err_note << "\n";

        if (expLoc.isValid()){
            //print the inline string(s) to the output file
            bool isValid;
			//Buffer the comment for outputing after translation is finished.
			//Disable this section to turn off error emission, by default if an
			// inline error string is empty, it will turn off comment insertion for that error
			if (!inline_note.empty() && AddInlineComments) {
				bufferComment(writeLoc, inlineStr.str(), replacements);
			}
        }
        //Send the stderr string to stderr
        llvm::errs() << errStr.str();
    }
    
    // Convenience method for dumping the same CU2CL error to both stderr and inlined comments
    //  using the mechanism above
    // Assumes the err_note is replicated as the inline comment to add to source.
    void emitCU2CLDiagnostic(SourceManager * SM, SourceLocation loc, std::string severity_str, std::string err_note, std::vector<Replacement> * replacements) {
        emitCU2CLDiagnostic(SM, loc, severity_str, err_note, err_note, replacements);
    }
    //Convenience method for getting a string of raw text between two SourceLocations
    std::string getStmtText(LangOptions * LO, SourceManager * SM, Stmt *s) {
        SourceLocation a(SM->getExpansionLoc(s->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
        return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
    }
	//Perform any last-minute checks on the Replacement and add it to the provided list of Replacements
    bool generateReplacement(std::vector<Replacement> &replacements, SourceManager * SM, SourceLocation sloc, int len, StringRef replace) {
	//Insert any protection logic here to make sure only legal replacements get added
	//TODO: Once Macro handling is improved, removing the SourceLoc check
	//if (!SM->isInSameSLocAddrSpace(sloc, sloc.getLocWithOffset(len), NULL)) {
	if (len < 0) { //If for some reason the length is negative (invalid) refuse to perform the replacement
	emitCU2CLDiagnostic(SM, sloc, "CU2CL Unhandled", "Replacement Range out of bounds", replace, &replacements);
	return false;
	}
        else replacements.push_back(Replacement(*SM, sloc, (unsigned)len, replace));

	return true;
    }
    //Method to output comments destined for addition to output OpenCL source
    // which have been buffered to avoid sideeffects with other rewrites
    //WARNING: Not threadsafe at all!
    void writeComments(SourceManager * SM) {
	struct commentBufferNode * curr = head->n;
	while (curr != NULL) { // as long as we have more comment nodes..
	    // inject the comment to the host output stream if true
		generateReplacement(*(curr->r), SM, SourceLocation::getFromPtrEncoding(curr->l), 0, llvm::StringRef(curr->s));
	    //move ahead, then destroy the current node
	    curr = curr->n;
	    free(head->n->s);
	    free(head->n);
	    head->n = curr;
	}

	tail = head;
    }

class RewriteCUDA;

//The class prototype necessary to trigger rewriting #included files
class RewriteIncludesCallback : public PPCallbacks {
private:
    RewriteCUDA *RCUDA;

public:
    RewriteIncludesCallback(RewriteCUDA *);

    virtual void InclusionDirective(SourceLocation, const Token &,
                                    llvm::StringRef, bool,
				    CharSourceRange, const FileEntry *,
                                    StringRef, StringRef,
				    const Module *);

};


/**
 * An AST consumer made to rewrite CUDA to OpenCL.
 * The entire translation process is essentially modeled as an ASTConsumer
 *  so that we can fully rely on Clang to construct the AST, then simply
 *  perform a full walk of the tree to identify the CUDA bits to translate.
 **/
class RewriteCUDA : public ASTConsumer {
protected:

private:
    typedef std::map<llvm::StringRef, std::list<llvm::StringRef> > StringRefListMap;

    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;
    SourceTuple *ST;

    Rewriter HostRewrite;
    Rewriter KernelRewrite;

    //TODO: Once Clang updates to use vectors rather than sets for Replacements
    // change this to reflect that
    std::vector<Replacement> HostReplace;
    std::vector<Replacement> KernReplace;

    //Rewritten files
    FileID MainFileID;
    std::string mainFilename;
    OutputFile *MainOutFile;
    OutputFile *MainKernelOutFile;
    //TODO lump IDs and both outfiles together

    StringRefListMap Kernels;

    std::set<DeclGroupRef, cmpDG> GlobalVarDeclGroups;
    std::set<DeclGroupRef, cmpDG> CurVarDeclGroups;
    std::set<DeclGroupRef, cmpDG> DeviceMemDGs;
    std::set<DeclaratorDecl *> DeviceMemVars;
    std::set<DeclaratorDecl *> HostMemVars;
    std::set<VarDecl *> ConstMemVars;
    std::set<VarDecl *> SharedMemVars;
    std::set<ParmVarDecl *> CurRefParmVars;

    std::map<SourceLocation, Replacement> HostVecVars;

    TypeLoc LastLoc;

    std::string MainFuncName;
    FunctionDecl *MainDecl;

    //Preamble string to insert at top of main host file
    std::string HostPreamble;
    std::string HostIncludes;
    std::string HostDecls;
    std::string HostGlobalVars;
    std::string HostKernels;
    std::string HostFunctions;

    bool IncludingStringH;

    //Preamble string to insert at top of main kernel file
    std::string DevPreamble;
    std::string DevFunctions;

    //Pre- and Postamble strings that bundle OpenCL boilerplate for a translation unit
    //Global boilerplate is generated in CU2CLInit and CU2CLClean
    std::string CLInit;
    std::string CLClean;




void TraverseStmt(Stmt *e, unsigned int indent) {
        for (unsigned int i = 0; i < indent; i++)
            llvm::errs() << "  ";
        llvm::errs() << e->getStmtClassName() << "\n";
        indent++;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI)
            if (*CI)
                TraverseStmt(*CI, indent);
    }

    template <class T>
    T *FindStmt(Stmt *e) {
        if (T *t = dyn_cast<T>(e))
            return t;
        T *ret = NULL;
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI) {
            ret = FindStmt<T>(*CI);
            if (ret)
                return ret;
        }
        return NULL;
    }


    std::string getTextFromLocs(SourceLocation a, SourceLocation b) {
	return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
    }


    //Simple function to strip attributes from host functions that may be declared as 
    // both __host__ and __device__, then passes off to the host-side statement rewriter
    void RewriteHostFunction(FunctionDecl *hostFunc) {
	//Register it on the RedeclMap
	//TODO: We may want to check if this FunctionDecl (by text location) has already been added by another AST
	//TODO:  but for now we are assuming we will generate the same replacements that just get deduped
	    AllFuncsByCanon[hostFunc->getFirstDecl()->getLocStart().printToString(*SM)].push_back(std::pair<FunctionDecl *, SourceTuple *>(hostFunc, ST));
	

        //Remove any CUDA function attributes
        if (CUDAHostAttr *attr = hostFunc->getAttr<CUDAHostAttr>()) {
            RewriteAttr(attr, "", HostReplace);
        }
        if (CUDADeviceAttr *attr = hostFunc->getAttr<CUDADeviceAttr>()) {
            RewriteAttr(attr, "", HostReplace);
        }

        //Rewrite the body
        if (Stmt *body = hostFunc->getBody()) {
            RewriteHostStmt(body);
        }
        CurVarDeclGroups.clear();
    }

    //Forks host-side statement processing between expressions, declarations, and other statements
    void RewriteHostStmt(Stmt *s) {
        //Visit this node
        if (Expr *e = dyn_cast<Expr>(s)) {
            std::string str;
            if (RewriteHostExpr(e, str)) {
                ReplaceStmtWithText(e, str, HostReplace);
            }
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
            DeclGroupRef DG = ds->getDeclGroup();
            Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            //Store VarDecl DeclGroupRefs
            if (firstDecl->getKind() == Decl::Var) {
                CurVarDeclGroups.insert(DG);
            }
            for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    RewriteHostVarDecl(vd);
                }
                //TODO other non-top level declarations??
            }
        }
        //TODO rewrite any other Stmts?

        else {
            //Traverse children and recurse
            for (Stmt::child_iterator CI = s->child_begin(), CE = s->child_end();
                 CI != CE; ++CI) {
                if (*CI)
                    RewriteHostStmt(*CI);
            }
        }
    }

    //Expressions, along with declarations, are the main meat of what needs to be rewritten
    //Host-side we primarily need to deal with CUDA C kernel launches and API call expressions
    bool RewriteHostExpr(Expr *e, std::string &newExpr) {
        //Return value specifies whether or not a rewrite occurred
        if (e->getSourceRange().isInvalid())
            return false;

        //Rewriter used for rewriting subexpressions
        Rewriter exprRewriter(*SM, *LO);
        //Instantiation locations are used to capture macros
        SourceRange realRange(SM->getExpansionLoc(e->getLocStart()),
                              SM->getExpansionLoc(e->getLocEnd()));

	//If DRE, register for potential late translation
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(e)) {

	    AllDeclRefsByDecl[dre->getDecl()->getLocStart().printToString(*SM)].push_back(dre);
	}

	//Detect CUDA C style kernel launches ie. fooKern<<<Grid, Block, shared, stream>>>(args..);
	// the Runtime and Driver API's launch mechanisms would be handled with the rest of the API calls
        if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(e)) {
            //Short-circuit templated kernel launches
            if (kce->isTypeDependent()) {
                emitCU2CLDiagnostic(SM, kce->getLocStart(), "CU2CL Untranslated", "Template-dependent kernel call", &HostReplace);
                return false;
            }
	    //Short-circuit launching a function pointer until we can handle it
	    else if (kce->getDirectCallee() == 0 && dyn_cast<ImplicitCastExpr>(kce->getCallee())) {
                emitCU2CLDiagnostic(SM, kce->getLocStart(), "CU2CL Unhandled", "Function pointer as kernel call", &HostReplace);
                return false;
            }
	    //If it's not a templated or pointer launch, proceed with translation
            newExpr = RewriteCUDAKernelCall(kce);
            return true;
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(e)) {
            if (ce->isTypeDependent()) {
                emitCU2CLDiagnostic(SM, ce->getLocStart(), "CU2CL Untranslated", "Template-dependent host call", &HostReplace);
                return false;
            }
            //This catches some errors encountered with heavily-nested, PP-assembled function-like macros
	    // mostly observed within the OpenGL and GLUT headers
            if (ce->getDirectCallee() == 0) {
                emitCU2CLDiagnostic(SM, SM->getExpansionLoc(ce->getLocStart()), "CU2CL Unhandled", "Could not identify direct callee in expression", &HostReplace);
            }
	    //This catches all Runtime API calls, since they are all prefixed by "cuda"
	    // and all Driver API calls that are prefixed with just "cu"
	    //Also catches cutil, cuFFT, cuBLAS, and other library calls incidentally, which may or may not be wanted
	    //TODO: Perhaps a second tier of filtering is needed
	    else if (ce->getDirectCallee()->getNameAsString().find("cu") == 0)
                return RewriteCUDACall(ce, newExpr);
        }
	//Catches expressions which refer to the member of a struct or class
	// in the CUDA case these are primarily just dim3s and cudaDeviceProp
        else if (MemberExpr *me = dyn_cast<MemberExpr>(e)) {
            //Check base Expr, if DeclRefExpr and a dim3, then rewrite
            if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(me->getBase())) {
                std::string type = dre->getDecl()->getType().getAsString();
                if (type == "dim3") {
                    std::string name = me->getMemberDecl()->getNameAsString();
                    if (name == "x") {
                        name = "[0]";
                    }
                    else if (name == "y") {
                        name = "[1]";
                    }
                    else if (name == "z") {
                        name = "[2]";
                    }
                    newExpr = getStmtText(LO, SM, dre) + name; //PrintStmtToString(dre) + name;
                    return true;
                }
                else if (type == "cudaDeviceProp") {
                    //TODO check what the reference is
                    //TODO if unsupported, print a warning

                    return false;
                }
            }
        }

	//Rewrite explicit casts of CUDA data types
        else if (ExplicitCastExpr *ece = dyn_cast<ExplicitCastExpr>(e)) {
            bool ret = true;

            TypeLoc origTL = ece->getTypeInfoAsWritten()->getTypeLoc();
            TypeLoc tl = origTL;
            while (!tl.getNextTypeLoc().isNull()) {
                tl = tl.getNextTypeLoc();
            }
            QualType qt = tl.getType();
            std::string type = qt.getAsString();

            if (type == "dim3") {
                if (origTL.getTypePtr()->isPointerType())
                    RewriteType(tl, "size_t *", exprRewriter);
                else
                    RewriteType(tl, "size_t[3]", exprRewriter);
            }
            else if (type == "struct cudaDeviceProp") {
                RewriteType(tl, "struct __cu2cl_DeviceProp", exprRewriter);
            }
            else if (type == "cudaStream_t") {
                RewriteType(tl, "cl_command_queue", exprRewriter);
            }
            else if (type == "cudaEvent_t") {
                RewriteType(tl, "cl_event", exprRewriter);
            }
            else {
                ret = false;
            }

            //Rewrite subexpression
            std::string s;
            if (RewriteHostExpr(ece->getSubExpr(), s)) {
                ReplaceStmtWithText(ece->getSubExpr(), s, exprRewriter);
                ret = true;
            }
            newExpr = exprRewriter.getRewrittenText(realRange);
            return ret;
        }
	//Rewrite unary expressions or type trait expressions (things like sizeof)
        else if (UnaryExprOrTypeTraitExpr *soe = dyn_cast<UnaryExprOrTypeTraitExpr>(e)) {
            if (soe->isArgumentType()) {
                bool ret = true;
                TypeLoc tl = soe->getArgumentTypeInfo()->getTypeLoc();
                while (!tl.getNextTypeLoc().isNull()) {
                    tl = tl.getNextTypeLoc();
                }
                QualType qt = tl.getType();
                std::string type = qt.getAsString();

                if (type == "dim3") {
                    RewriteType(tl, "size_t[3]", exprRewriter);
                }
                else if (type == "struct cudaDeviceProp") {
                    RewriteType(tl, "struct __cu2cl_DeviceProp", exprRewriter);
                }
                else if (type == "cudaStream_t") {
                    RewriteType(tl, "cl_command_queue", exprRewriter);
                }
                else if (type == "cudaEvent_t") {
                    RewriteType(tl, "cl_event", exprRewriter);
                }
                else {
                    ret = false;
                }
                    SourceRange newrealRange(SM->getExpansionLoc(e->getLocStart()),
                              SM->getExpansionLoc(e->getLocEnd()));
                newExpr = exprRewriter.getRewrittenText(newrealRange);
                return ret;
            }
        }
	//Catches dim3 declarations of the form: some_var=dim3(x,p,z);
	// the RHS is considered a temporary object
        else if (CXXTemporaryObjectExpr *cte = dyn_cast<CXXTemporaryObjectExpr>(e)) {
            //TODO need to know if in constructor or not... if not in
            //constructor, then need to assign each separately
            CXXConstructorDecl *ccd = cte->getConstructor();
            CXXRecordDecl *crd = ccd->getParent();
            const Type *t = crd->getTypeForDecl();
            QualType qt = t->getCanonicalTypeInternal();
            std::string type = qt.getAsString();

            if (type == "struct dim3") {
                std::string args = "{";
                for (CXXConstructExpr::arg_iterator i = cte->arg_begin(),
                     e = cte->arg_end(); i != e; ++i) {
                    Expr *arg = *i;
                    std::string s;
                    if (CXXDefaultArgExpr *defArg = dyn_cast<CXXDefaultArgExpr>(arg)) {
                        RewriteHostExpr(defArg->getExpr(), s);
                    }
                    else {
                        RewriteHostExpr(arg, s);
                    }
                    args += s;
                    if (i + 1 != e)
                        args += ", ";
                }
                args += "}";
                newExpr = args;
                return true;
            }
        }
	//Catches dim3 declarations of the form: dim3 some_var(x,y,z);
        else if (CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(e)) {
            CXXConstructorDecl *ccd = cce->getConstructor();
            CXXRecordDecl *crd = ccd->getParent();
            const Type *t = crd->getTypeForDecl();
            QualType qt = t->getCanonicalTypeInternal();
            std::string type = qt.getAsString();

            if (type == "struct dim3") {
                if (cce->getNumArgs() == 1) {
                    //Rewrite subexpression
                    bool ret = false;
                    std::string s;
                    if (RewriteHostExpr(cce->getArg(0), s)) {
                        ReplaceStmtWithText(cce->getArg(0), s, exprRewriter);
                        ret = true;
                    }
                    SourceRange newrealRange(SM->getExpansionLoc(e->getLocStart()),
                              SM->getExpansionLoc(e->getLocEnd()));
                    newExpr = exprRewriter.getRewrittenText(newrealRange);
                    return ret;
                }
                else {
                    std::string args = " = {";
                    for (CXXConstructExpr::arg_iterator i = cce->arg_begin(),
                         e = cce->arg_end(); i != e; ++i) {
                        Expr *arg = *i;
                        std::string s;
                        if (CXXDefaultArgExpr *defArg = dyn_cast<CXXDefaultArgExpr>(arg)) {
                            RewriteHostExpr(defArg->getExpr(), s);
                        }
                        else {
                            RewriteHostExpr(arg, s);
                        }
                        args += s;
                        if (i + 1 != e)
                            args += ", ";
                    }
                    args += "}";
                    newExpr = args;
                }
                return true;
            }
        }

        bool ret = false;
        //Do a DFS, recursing into children, then rewriting this expression
        //if rewrite happened, replace text at old sourcerange
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI) {
            std::string s;
            Expr *child = (Expr *) *CI;
            if (child && RewriteHostExpr(child, s)) {
                //Perform "rewrite", which is just a simple replace
                ReplaceStmtWithText(child, s, exprRewriter);
                ret = true;
            }
        }


        SourceRange newrealRange(SM->getExpansionLoc(e->getLocStart()),
                              SM->getExpansionLoc(e->getLocEnd()));
        newExpr = exprRewriter.getRewrittenText(realRange);
        return ret;
    }

    //Rewriter for host-side Runtime API calls, prefixed with "cuda"
    //
    //The major if-else just compares on the name of the function, and when
    // it finds a match, performs the necessary rewrite.
    //In the majority of cases, this requires calling RewriteHostExpr on one
    // or more of the function's arguments
    //In a few cases, we catch something we can't translate yet, and there
    // is a final catch-all for anything that's not caught by the if-else tree
    bool RewriteCUDACall(CallExpr *cudaCall, std::string &newExpr) {
        //TODO all CUDA calls return a cudaError_t, so those semantics need to be preserved where possible
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();

        //Thread Management
        if (funcName == "cudaThreadExit") {
            //Replace with clReleaseContext
            newExpr = "clReleaseContext(__cu2cl_Context)";
        }
        else if (funcName == "cudaThreadSynchronize") {
            //Replace with clFinish
            newExpr = "clFinish(__cu2cl_CommandQueue)";
        }

        //Device Management
        else if (funcName == "cudaGetDevice") {
            //Replace by assigning current value of clDevice to arg
	    //TODO Alternatively, this could be queried from the queue with clGetCommandQueueInfo
            Expr *device = cudaCall->getArg(0);
            std::string newDevice;
            RewriteHostExpr(device, newDevice);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(device);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());

            //Rewrite var type to cl_device_id
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            RewriteType(tl, "cl_device_id", HostReplace);
            newExpr = "*" + newDevice + " = __cu2cl_Device";
        }
        else if (funcName == "cudaGetDeviceCount") {
            //Replace with clGetDeviceIDs
	    //TODO: Update to use the device array from __cu2cl_ScanDevices
            Expr *count = cudaCall->getArg(0);
            std::string newCount;
            RewriteHostExpr(count, newCount);
            newExpr = "clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 0, NULL, (cl_uint *) " + newCount + ")";
        }
        else if (funcName == "cudaSetDevice") {
            if (!UsesCUDASetDevice) {
                UsesCUDASetDevice = true;
		GlobalCDecls["cu2cl_util.c"].push_back("cl_device_id * __cu2cl_AllDevices;\n");
		GlobalCDecls["cu2cl_util.c"].push_back("cl_uint __cu2cl_AllDevices_curr_idx;\n");
		GlobalCDecls["cu2cl_util.c"].push_back("cl_uint __cu2cl_AllDevices_size;\n");
		GlobalCFuncs.push_back(CU2CL_SCAN_DEVICES);
		GlobalHDecls.push_back(CU2CL_SCAN_DEVICES_H);
		GlobalCFuncs.push_back(CU2CL_SET_DEVICE);
		GlobalHDecls.push_back(CU2CL_SET_DEVICE_H);
            }
            Expr *device = cudaCall->getArg(0);
            //Device will only be an integer ID, so don't look for a reference
            //DeclRefExpr *dre = FindStmt<DeclRefExpr>(device);
            //if (dre != NULL) {
            std::string newDevice;
            RewriteHostExpr(device, newDevice);
            //TODO also rewrite type as in cudaGetDevice
            //VarDecl *var = dyn_cast<VarDecl>(dre->getDecl());
            newExpr = "__cu2cl_SetDevice(" + newDevice + ")";
            emitCU2CLDiagnostic(SM, cudaCall->getLocStart(), "CU2CL Warning", "CU2CL Identified cudaSetDevice usage", &HostReplace);
            //}
        }
        else if (funcName == "cudaSetDeviceFlags") {
            //Remove for now, as OpenCL has no device flags to set
	    //TODO: emit a note with the device flags
            newExpr = "";
        }
        else if (funcName == "cudaGetDeviceProperties") {
            //Replace with __cu2cl_GetDeviceProperties
            Expr *prop = cudaCall->getArg(0);
            Expr *device = cudaCall->getArg(1);
            std::string newProp, newDevice;
            RewriteHostExpr(prop, newProp);
            RewriteHostExpr(device, newDevice);
            newExpr = "__cu2cl_GetDeviceProperties(" + newProp + ", " + newDevice + ")";
        }

        //Stream Management
        else if (funcName == "cudaStreamCreate") {
            //Replace with clCreateCommandQueue
            Expr *pStream = cudaCall->getArg(0);
            std::string newPStream;
            RewriteHostExpr(pStream, newPStream);

            newExpr = "*" + newPStream + " = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL)";
        }
        else if (funcName == "cudaStreamDestroy") {
            //Replace with clReleaseCommandQueue
            Expr *stream = cudaCall->getArg(0);
            std::string newStream;
            RewriteHostExpr(stream, newStream);
            newExpr = "clReleaseCommandQueue(" + newStream + ")";
        }
        else if (funcName == "cudaStreamQuery") {
            //Replace with __cu2cl_CommandQueueQuery
            if (!UsesCUDAStreamQuery) {
		GlobalCFuncs.push_back(CL_COMMAND_QUEUE_QUERY);
		GlobalHDecls.push_back(CL_COMMAND_QUEUE_QUERY_H);
                UsesCUDAStreamQuery = true;
            }

            Expr *stream = cudaCall->getArg(0);
            std::string newStream;
            RewriteHostExpr(stream, newStream);
            newExpr = "__cu2cl_CommandQueueQuery(" + newStream + ")";
        }
        else if (funcName == "cudaStreamSynchronize") {
            //Replace with clFinish
            Expr *stream = cudaCall->getArg(0);
            std::string newStream;
            RewriteHostExpr(stream, newStream);
            newExpr = "clFinish(" + newStream + ")";
        }
        else if (funcName == "cudaStreamWaitEvent") {
            //Replace with clEnqueueWaitForEvents
            Expr *stream = cudaCall->getArg(0);
            Expr *event = cudaCall->getArg(1);
            std::string newStream, newEvent;
            RewriteHostExpr(stream, newStream);
            RewriteHostExpr(event, newEvent);
            newExpr = "clEnqueueWaitForEvents(" + newStream + ", 1, &" + newEvent + ")";
        }

        //Event Management
        //else if (funcName == "cudaEventCreate") {
	//TODO: Replace with clCreateUserEvent
            //Remove the call
        //    newExpr = "";
        //}
        //else if (funcName == "cudaEventCreateWithFlags") {
	//TODO: Replace with clSetUserEventStatus
            //Remove the call
        //    newExpr = "";
        //}
        else if (funcName == "cudaEventDestroy") {
            //Replace with clReleaseEvent
            Expr *event = cudaCall->getArg(0);
            std::string newEvent;
            RewriteHostExpr(event, newEvent);
            newExpr = "clReleaseEvent(" + newEvent + ")";
        }
        else if (funcName == "cudaEventElapsedTime") {
            //Replace with __cu2cl_EventElapsedTime
            if (!UsesCUDAEventElapsedTime) {
		GlobalCFuncs.push_back(CL_EVENT_ELAPSED_TIME);
		GlobalHDecls.push_back(CL_EVENT_ELAPSED_TIME_H);
                UsesCUDAEventElapsedTime = true;
            }

            Expr *ms = cudaCall->getArg(0);
            Expr *start = cudaCall->getArg(1);
            Expr *end = cudaCall->getArg(2);
            std::string newMS, newStart, newEnd;
            RewriteHostExpr(ms, newMS);
            RewriteHostExpr(start, newStart);
            RewriteHostExpr(end, newEnd);
            newExpr = "__cu2cl_EventElapsedTime(" + newMS + ", " + newStart + ", " + newEnd + ")";
        }
        else if (funcName == "cudaEventQuery") {
            //Replace with __cu2cl_EventQuery
            if (!UsesCUDAEventQuery) {
		GlobalCFuncs.push_back(CL_EVENT_QUERY);
		GlobalHDecls.push_back(CL_EVENT_QUERY_H);
                UsesCUDAEventQuery = true;
            }

            Expr *event = cudaCall->getArg(0);
            std::string newEvent;
            RewriteHostExpr(event, newEvent);
            newExpr = "__cu2cl_EventQuery(" + newEvent + ")";
        }
        else if (funcName == "cudaEventRecord") {
            //Replace with clEnqueueMarker
            Expr *event = cudaCall->getArg(0);
            Expr *stream = cudaCall->getArg(1);
            std::string newStream, newEvent;
            RewriteHostExpr(stream, newStream);
            RewriteHostExpr(event, newEvent);

            //If stream == 0, then cl_command_queue == __cu2cl_CommandQueue
            if (newStream == "0")
                newStream = "__cu2cl_CommandQueue";
            newExpr = "clEnqueueMarker(" + newStream + ", &" + newEvent + ")";
        }
        else if (funcName == "cudaEventSynchronize") {
            //Replace with clWaitForEvents
            Expr *event = cudaCall->getArg(0);
            std::string newEvent;
            RewriteHostExpr(event, newEvent);
            newExpr = "clWaitForEvents(1, &" + newEvent + ")";
        }

        //Memory Management
        else if (funcName == "cudaHostAlloc") {
            //Replace with __cu2cl_MallocHost
            if (!UsesCUDAMallocHost) {
		GlobalCFuncs.push_back(CL_MALLOC_HOST);
		GlobalHDecls.push_back(CL_MALLOC_HOST_H);
                UsesCUDAMallocHost = true;
            }

            Expr *ptr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            std::string newPtr, newSize;
            RewriteHostExpr(ptr, newPtr);
            RewriteHostExpr(size, newSize);

            DeclRefExpr *dr = FindStmt<DeclRefExpr>(ptr);
	    MemberExpr *mr = FindStmt<MemberExpr>(ptr);
DeclaratorDecl *var = NULL;
	    //If the device pointer is a struct or class member, it shows up as a MemberExpr rather than a DeclRefExpr
	    if (mr != NULL) {
		emitCU2CLDiagnostic(SM, cudaCall->getLocStart(), "CU2CL Note", "Identified member expression in cudaHostAlloc device pointer", &HostReplace);
		var = dyn_cast<DeclaratorDecl>(mr->getMemberDecl());
	    }
	    //If it's just a global or locally-scoped singleton, then it shows up as a DeclRefExpr
	    else {
		var = dyn_cast<VarDecl>(dr->getDecl());
	    }
            llvm::StringRef varName = var->getName();

            newExpr = "__cu2cl_MallocHost(" + newPtr + ", " + newSize + ", &__cu2cl_Mem_" + varName.str() + ")";

            if (HostMemVars.find(var) == HostMemVars.end()) {
                //Create new cl_mem for ptr
                HostGlobalVars += "cl_mem __cu2cl_Mem_" + varName.str() + ";\n";
                //Add var to HostMemVars
                HostMemVars.insert(var);
            }
        }
        else if (funcName == "cudaFree") {
            Expr *devPtr = cudaCall->getArg(0);
            std::string newDevPtr;
            RewriteHostExpr(devPtr, newDevPtr);

            //Replace with clReleaseMemObject
            newExpr = "clReleaseMemObject(" + newDevPtr + ")";
        }
        else if (funcName == "cudaFreeHost") {
            //Replace with __cu2cl_FreeHost
            if (!UsesCUDAFreeHost) {
		GlobalCFuncs.push_back(CL_FREE_HOST);
		GlobalHDecls.push_back(CL_FREE_HOST_H);
                UsesCUDAFreeHost = true;
            }

            Expr *ptr = cudaCall->getArg(0);
            std::string newPtr;
            RewriteHostExpr(ptr, newPtr);

            DeclRefExpr *dr = FindStmt<DeclRefExpr>(ptr);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            llvm::StringRef varName = var->getName();

            newExpr = "__cu2cl_FreeHost(" + newPtr + ", __cu2cl_Mem_" + varName.str() + ")";
        }
        else if (funcName == "cudaMalloc") {
            Expr *devPtr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            std::string newDevPtr, newSize;
            RewriteHostExpr(size, newSize);
            RewriteHostExpr(devPtr, newDevPtr);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(devPtr);
	    MemberExpr *mr = FindStmt<MemberExpr>(devPtr);
DeclaratorDecl *var;
	    //If the device pointer is a struct or class member, it shows up as a MemberExpr rather than a DeclRefExpr
	    if (mr != NULL) {
		emitCU2CLDiagnostic(SM, cudaCall->getLocStart(), "CU2CL Note", "Identified member expression in cudaMalloc device pointer", &HostReplace);
		var = dyn_cast<DeclaratorDecl>(mr->getMemberDecl());
	    }
	    //If it's just a global or locally-scoped singleton, then it shows up as a DeclRefExpr
	    else {
		var = dyn_cast<VarDecl>(dr->getDecl());
	    }

            //Replace with clCreateBuffer
            newExpr = "*" + newDevPtr + " = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, " + newSize + ", NULL, NULL)";

            DeclGroupRef varDG(var);
            if (CurVarDeclGroups.find(varDG) != CurVarDeclGroups.end()) {
                DeviceMemDGs.insert(*CurVarDeclGroups.find(varDG));
            }
            else if (GlobalVarDeclGroups.find(varDG) != GlobalVarDeclGroups.end()) {
                DeviceMemDGs.insert(*GlobalVarDeclGroups.find(varDG));
            }
            else {
emitCU2CLDiagnostic(SM, cudaCall->getLocStart(), "CU2CL Note", "Rewriting single decl", &HostReplace);
                //Change variable's type to cl_mem
                TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
		DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(var)), ST));
            }

            //Add var to DeviceMemVars
            DeviceMemVars.insert(var);
        }
        else if (funcName == "cudaMallocHost") {
            //Replace with __cu2cl_MallocHost
            if (!UsesCUDAMallocHost) {
		GlobalCFuncs.push_back(CL_MALLOC_HOST);
		GlobalHDecls.push_back(CL_MALLOC_HOST_H);
                UsesCUDAMallocHost = true;
            }

            Expr *ptr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            std::string newPtr, newSize;
            RewriteHostExpr(ptr, newPtr);
            RewriteHostExpr(size, newSize);

            DeclRefExpr *dr = FindStmt<DeclRefExpr>(ptr);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            llvm::StringRef varName = var->getName();

            newExpr = "__cu2cl_MallocHost(" + newPtr + ", " + newSize + ", &__cu2cl_Mem_" + varName.str() + ")";

            if (HostMemVars.find(var) == HostMemVars.end()) {
                //Create new cl_mem for ptr
                HostGlobalVars += "cl_mem __cu2cl_Mem_" + varName.str() + ";\n";
                //Add var to HostMemVars
                HostMemVars.insert(var);
            }
        }
        //TODO: support cudaMemcpyDefault
        //TODO support offsets (will need to grab pointer out of cudaMemcpy
        // call, then separate off the rest of the math as the offset)
        else if (funcName == "cudaMemcpy") {
            //Inspect kind of memcpy and rewrite accordingly
            Expr *dst = cudaCall->getArg(0);
            Expr *src = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            Expr *kind = cudaCall->getArg(3);
            std::string newDst, newSrc, newCount;
            RewriteHostExpr(dst, newDst);
            RewriteHostExpr(src, newSrc);
            RewriteHostExpr(count, newCount);

            DeclRefExpr *dr = FindStmt<DeclRefExpr>(kind);
            EnumConstantDecl *enumConst = dyn_cast<EnumConstantDecl>(dr->getDecl());
            std::string enumString = enumConst->getNameAsString();

            if (enumString == "cudaMemcpyHostToHost") {
                //standard memcpy
                //Make sure to include <string.h>
                if (!IncludingStringH) {
                    HostIncludes += "#include <string.h>\n";
                    IncludingStringH = true;
                }

                newExpr = "memcpy(" + newDst + ", " + newSrc + ", " + newCount + ")";
            }
            else if (enumString == "cudaMemcpyHostToDevice") {
                //clEnqueueWriteBuffer
                newExpr = "clEnqueueWriteBuffer(__cu2cl_CommandQueue, " + newDst + ", CL_TRUE, 0, " + newCount + ", " + newSrc + ", 0, NULL, NULL)";
            }
            else if (enumString == "cudaMemcpyDeviceToHost") {
                //clEnqueueReadBuffer
                newExpr = "clEnqueueReadBuffer(__cu2cl_CommandQueue, " + newSrc + ", CL_TRUE, 0, " + newCount + ", " + newDst + ", 0, NULL, NULL)";
            }
            else if (enumString == "cudaMemcpyDeviceToDevice") {
		//clEnqueueCopyBuffer
		newExpr = "clEnqueueCopyBuffer(__cu2cl_CommandQueue, " + newSrc + ", " + newDst + ", 0, 0, " + newCount + ", 0, NULL, NULL)";
            }
            else {
                emitCU2CLDiagnostic(SM, cudaCall->getLocStart(), "CU2CL Unsupported", "Unsupported cudaMemcpyKind: " + enumString, &HostReplace);
            }
        }
        //TODO: support cudaMemcpyDefault
        //TODO support offsets (will need to grab pointer out of cudaMemcpy
        // call, then separate off the rest of the math as the offset)
        else if (funcName == "cudaMemcpyAsync") {
            //Inspect kind of memcpy and rewrite accordingly
            Expr *dst = cudaCall->getArg(0);
            Expr *src = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            Expr *kind = cudaCall->getArg(3);
            Expr *stream = cudaCall->getArg(4);
            std::string newDst, newSrc, newCount, newStream;
            RewriteHostExpr(dst, newDst);
            RewriteHostExpr(src, newSrc);
            RewriteHostExpr(count, newCount);
            RewriteHostExpr(stream, newStream);
            if (newStream == "0")
                newStream = "__cu2cl_CommandQueue";

            DeclRefExpr *dr = FindStmt<DeclRefExpr>(kind);
            EnumConstantDecl *enumConst = dyn_cast<EnumConstantDecl>(dr->getDecl());
            std::string enumString = enumConst->getNameAsString();

            if (enumString == "cudaMemcpyHostToHost") {
                //standard memcpy
                //Make sure to include <string.h>
                if (!IncludingStringH) {
                    HostIncludes += "#include <string.h>\n";
                    IncludingStringH = true;
                }

                //dst and src are HostMemVars, so regular memcpy can be used
                newExpr = "memcpy(" + newDst + ", " + newSrc + ", " + newCount + ")";
            }
            else if (enumString == "cudaMemcpyHostToDevice") {
                //clEnqueueWriteBuffer, src is HostMemVar
                dr = FindStmt<DeclRefExpr>(src);
                VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
                llvm::StringRef varName = var->getName();
                newExpr = "clEnqueueWriteBuffer(" + newStream + ", " + newDst + ", CL_FALSE, 0, " + newCount + ", " + newSrc + ", 0, NULL, NULL)";
            }
            else if (enumString == "cudaMemcpyDeviceToHost") {
                //clEnqueueReadBuffer, dst is HostMemVar
                dr = FindStmt<DeclRefExpr>(dst);
                VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
                llvm::StringRef varName = var->getName();
                newExpr = "clEnqueueReadBuffer(" + newStream + ", " + newSrc + ", CL_FALSE, 0, " + newCount + ", " + newDst + ", 0, NULL, NULL)";
            }
            else if (enumString == "cudaMemcpyDeviceToDevice") {
		//clEnqueueCopyBuffer
		newExpr = "clEnqueueCopyBuffer(__cu2cl_CommandQueue, " + newSrc + ", " + newDst + ", 0, 0, " + newCount + ", 0, NULL, NULL)";
            }
            else {
                emitCU2CLDiagnostic(SM, cudaCall->getLocStart(), "CU2CL Unsupported", "Unsupported cudaMemcpyKind: " + enumString, &HostReplace);
            }
        }
        //else if (funcName == "cudaMemcpyToSymbol") {
            //TODO: implement
        //}
	//FIXME: Generate cu2cl_util.cl and the requisite boilerplate
        else if (funcName == "cudaMemset") {
            if (!UsesCUDAMemset) {
		if(!UsesCU2CLUtilCL) UsesCU2CLUtilCL = true;
		GlobalCFuncs.push_back(CL_MEMSET);
		GlobalHDecls.push_back(CL_MEMSET_H);
		GlobalCLFuncs.push_back(CL_MEMSET_KERNEL);
                UtilKernels.push_back("__cu2cl_Memset");
		GlobalCDecls["cu2cl_util.c"].push_back("cl_kernel __cu2cl_Kernel___cu2cl_Memset;\n");
                UsesCUDAMemset = true;
            }
            //Follow Swan's example of setting via a kernel
            Expr *devPtr = cudaCall->getArg(0);
            Expr *value = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            std::string newDevPtr, newValue, newCount;
            RewriteHostExpr(devPtr, newDevPtr);
            RewriteHostExpr(value, newValue);
            RewriteHostExpr(count, newCount);
            newExpr = "__cu2cl_Memset(" + newDevPtr + ", " + newValue + ", " + newCount + ")";
        }
        else {
            emitCU2CLDiagnostic(SM, SM->getExpansionLoc(cudaCall->getLocStart()), "CU2CL Unsupported", "Unsupported CUDA call: " + funcName, &HostReplace);
            return false;
	    //TODO: Even if the call is unsupported, we should attempt to translate params, need to fire up the standard rewrite machinery for that and return whether or not any children were changed
        }
        return true;
    }

    //The Rewriter for standard CUDA C kernel launches of the form:
    // someKern<<<Grid, Block, shared, stream>>>(args...);
    //TODO: support handling function pointers
    //TODO: support the shared and stream exec-config parameters
    std::string RewriteCUDAKernelCall(CUDAKernelCallExpr *kernelCall) {
        FunctionDecl *callee = kernelCall->getDirectCallee();
        CallExpr *kernelConfig = kernelCall->getConfig();
        
        std::string kernelName = "__cu2cl_Kernel_" + callee->getNameAsString();
        std::ostringstream args;
        unsigned int dims = 1;

        //Set kernel arguments
        for (unsigned i = 0; i < kernelCall->getNumArgs(); i++) {
            Expr *arg = kernelCall->getArg(i);//->IgnoreParenCasts();
            std::string newArg;
            RewriteHostExpr(arg, newArg);
	    //If there's no declaration in the arg, or it isn't a valid L value,
	    // then it must be a "literal argument" (not reducible to an address)
	    if (FindStmt<DeclRefExpr>(arg) == NULL || !arg->IgnoreParenCasts()->isLValue()) {
		//make a temporary variable to hold this value, pass it, and destroy it
		//TODO: Do this in a separate block to guarantee scope
		args << arg->getType().getAsString() << " __cu2cl_Kernel_" << callee->getNameAsString() << "_temp_arg_" << i << " = " << newArg << ";\n";
		args << "clSetKernelArg(" << kernelName << ", " << i << ", sizeof(" << arg->getType().getAsString() <<"), &__cu2cl_Kernel_" << callee->getNameAsString() << "_temp_arg_" << i << ");\n";

		std::stringstream comment;
		comment << "Inserted temporary variable for kernel literal argument " << i << "!";
		emitCU2CLDiagnostic(SM, kernelCall->getLocStart(), "CU2CL Note", comment.str(), &HostReplace);
	    }
	    //If the arg is just a declared variable, simply pass its address
	    else {
		VarDecl *var = dyn_cast<VarDecl>(FindStmt<DeclRefExpr>(arg)->getDecl());

		args << "clSetKernelArg(" << kernelName << ", " << i << ", sizeof(";
		if (DeviceMemVars.find(var) != DeviceMemVars.end()) {
		    //arg var is a cl_mem
		    args << "cl_mem";
		}
		else {
		    args << arg->getType().getAsString();
		}
		args << "), &" << newArg << ");\n";
	    }
        }

        //TODO add additional argument(s) for constant memory that must be explicitly passed

        //Set work sizes
        //Guaranteed to be dim3s, so pull out their x,y,z values
        Expr *grid = kernelConfig->getArg(0);
        Expr *block = kernelConfig->getArg(1);

	//Rewrite the threadblock expression
	CXXConstructExpr *construct = dyn_cast<CXXConstructExpr>(block);
        ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));

	//TODO: Check if all kernel launch parameters now show up as MaterializeTemporaryExpr
	// if so, standardize it as this with the ImplicitCastExpr fallback
	if (cast == NULL) {
	    //try chewing it up as a MaterializeTemporaryExpr
	    MaterializeTemporaryExpr *mat = dyn_cast<MaterializeTemporaryExpr>(construct->getArg(0));
	    if (mat) {
		cast = dyn_cast<ImplicitCastExpr>(mat->GetTemporaryExpr());
	    }
	}

	DeclRefExpr *dre;
	if (cast == NULL) {
	    emitCU2CLDiagnostic(SM, construct->getLocStart(), "CU2CL Note", "Fast-tracked dim3 type without cast", &HostReplace);
	    dre = dyn_cast<DeclRefExpr>(construct->getArg(0));
	} else {
	    dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten());
	}
        if (dre) {
            //Variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                dims = 3;
                for (unsigned int i = 0; i < 3; i++)
                    args << "localWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "];\n";
            }
            else {
                //Some integer type, likely
                args << "localWorkSize[0] = " << getStmtText(LO, SM, dre) << ";\n";
            }
        }
        else {
            //Some other expression passed to block
            Expr *arg = cast->getSubExprAsWritten();
            std::string s;
            RewriteHostExpr(arg, s);
        }

	//Rewrite the grid expression
        construct = dyn_cast<CXXConstructExpr>(grid);
        cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));


	//TODO: Check if all kernel launch parameters now show up as MaterializeTemporaryExpr
	// if so, standardize it as this with the ImplicitCastExpr fallback
	if (cast == NULL) {
	    //try chewing it up as a MaterializeTemporaryExpr
	    MaterializeTemporaryExpr *mat = dyn_cast<MaterializeTemporaryExpr>(construct->getArg(0));
	    if (mat) {
		cast = dyn_cast<ImplicitCastExpr>(mat->GetTemporaryExpr());
	    }
	}

	if (cast == NULL) {
	    emitCU2CLDiagnostic(SM, construct->getLocStart(), "CU2CL Note", "Fast-tracked dim3 type without cast", &HostReplace);
	    dre = dyn_cast<DeclRefExpr>(construct->getArg(0));
	} else {
	    dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten());
	}
        if (dre) {
            //Variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                dims = 3;
                for (unsigned int i = 0; i < 3; i++)
                    args << "globalWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "]*localWorkSize[" << i << "];\n";
            }
            else {
                //Some integer type, likely
                args << "globalWorkSize[0] = (" << getStmtText(LO, SM, dre) << ")*localWorkSize[0];\n";
            }
        }
        else {
            //constant passed to grid
            Expr *arg = cast->getSubExprAsWritten();
            std::string s;
            RewriteHostExpr(arg, s);
            args << "globalWorkSize[0] = (" << s << ")*localWorkSize[0];\n";
        }
        args << "clEnqueueNDRangeKernel(__cu2cl_CommandQueue, " << kernelName << ", " << dims << ", NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)";

        return args.str();
    }

    void RewriteHostVarDecl(VarDecl *var) {
        if (CUDAConstantAttr *constAttr = var->getAttr<CUDAConstantAttr>()) {
            //TODO: 0.9 Do something with __constant__ memory declarations
		//DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(var)), ST));
	//	return;
            RewriteAttr(constAttr, "", HostReplace);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", HostReplace);
            ConstMemVars.insert(var);

            TypeLoc origTL = var->getTypeSourceInfo()->getTypeLoc();
            if (LastLoc.isNull() || origTL.getBeginLoc() != LastLoc.getBeginLoc()) {
                LastLoc = origTL;
		DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(var)), ST));
//                RewriteType(origTL, "cl_mem", HostReplace);
            }
            return;
        }
        else if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
            //Handle __shared__ memory declarations
            RewriteAttr(sharedAttr, "", HostReplace);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", HostReplace);
            //TODO rewrite shared mem
            //If extern, remove extern keyword and make into pointer
            //if (var->isExtern())
            SharedMemVars.insert(var);
        }
        else if (CUDADeviceAttr *attr = var->getAttr<CUDADeviceAttr>()) {
            //Handle __device__ memory declarations
            RewriteAttr(attr, "", HostReplace);
            //TODO add to devmems, rewrite type
        }

        TypeLoc origTL = var->getTypeSourceInfo()->getTypeLoc();

        TypeLoc tl = origTL;
        while (!tl.getNextTypeLoc().isNull()) {
            tl = tl.getNextTypeLoc();
        }
        QualType qt = tl.getType();
        std::string type = qt.getAsString();

        //Rewrite var type
        if (LastLoc.isNull() || origTL.getBeginLoc() != LastLoc.getBeginLoc()) {
            LastLoc = origTL;
            if (type == "dim3") {
                //Rewrite to size_t[3] array
                RewriteType(tl, "size_t", HostReplace);
            }
            else if (type == "struct cudaDeviceProp") {
                if (!UsesCUDADeviceProp) {
		    GlobalHDecls.push_back(CL_DEVICE_PROP);
		    GlobalCFuncs.push_back(CL_GET_DEVICE_PROPS);
		    GlobalHDecls.push_back(CL_GET_DEVICE_PROPS_H);
                    UsesCUDADeviceProp = true;
                }
                RewriteType(tl, "__cu2cl_DeviceProp", HostReplace);
            }
            else if (type == "cudaStream_t") {
                RewriteType(tl, "cl_command_queue", HostReplace);
            }
            else if (type == "cudaEvent_t") {
                RewriteType(tl, "cl_event", HostReplace);
            }
            else {
                std::string newType = RewriteVectorType(type, true);
                if (newType != "") {
		    //Stage the replacement in a map to avoid conflicts with later cl_mem conversions of cudaMalloced host variables
		    Replacement vecType(*SM, tl.getBeginLoc(), getRangeSize(*SM, CharSourceRange::getTokenRange(tl.getLocalSourceRange())), newType);
		    //Try to insert into the map, but just dump a diagnostic warning if we fail
		    // Don't need to try too hard, since if a Replacement is already mapped at this location it must also be another vector rewrite
		    if (!HostVecVars.insert(std::pair<SourceLocation, Replacement>(tl.getBeginLoc(), vecType)).second)
			emitCU2CLDiagnostic(SM, tl.getBeginLoc(), "CU2CL Warning", "Failed to insert host vector type Replacement to cl_mem conflict map!\n" + vecType.toString(), &HostReplace);
		}
            }
            //TODO check other CUDA-only types to rewrite
        }

        //Rewrite initial value
        if (var->hasInit()) {
            Expr *e = var->getInit();
            std::string s;
	    //Given the loss of InsertBefore/After semantics with the switch of
	    // Rewriters to Replacements, the deferred insertion of a default dim3
	    // initialization is now required, the below boolean handles that
	    bool deferInsert = false;
            if (RewriteHostExpr(e, s)) {
                //Special cases for dim3s
                if (type == "dim3") {
                    CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(e);
                    if (cce && cce->getNumArgs() > 1) {
                        SourceRange parenRange = cce->getParenOrBraceRange();
                        if (parenRange.isValid()) {
			    generateReplacement(HostReplace, SM, parenRange.getBegin(), getRangeSize(*SM, CharSourceRange::getTokenRange(parenRange)), s);
                        }
                        else {
			    if (origTL.getTypePtr()->isPointerType())
				HostReplace.push_back(Replacement(*SM, PP->getLocForEndOfToken(var->getLocation()), 0, s));
			    else
				deferInsert = true;
                        }
                    }
                    else {
                        ReplaceStmtWithText(e, s, HostReplace);
                    }

                    //Add [3] to end/start of var identifier
                    if (origTL.getTypePtr()->isPointerType())
			    generateReplacement(HostReplace, SM, var->getLocation(), 0, "*");
                    else {
			if (!deferInsert)
			    generateReplacement(HostReplace, SM, PP->getLocForEndOfToken(var->getLocation()), 0, "[3]");
		    }

		    if (deferInsert) {
			    generateReplacement(HostReplace, SM, PP->getLocForEndOfToken(var->getLocation()), 0, "[3]" + s);
		    }
                }
                else
                    ReplaceStmtWithText(e, s, HostReplace);
            }
        }
    }

    //This is just used to grab the main function as a global variable
    // used by other functions, partiocularly boilerplate insertion
    void RewriteMain(FunctionDecl *mainDecl) {
        MainDecl = mainDecl;
    }

    //Transform kernel functions into their OpenCL form
    //TODO: Support translation-time mangling of template specializations
    // into C-compatible forms.
    void RewriteKernelFunction(FunctionDecl *kernelFunc) {

	//Paul: Adjusted this to *not* register forward declarations of kernels
	// This means that only the CUDA file which includes the *definition* of the kernel will be responsible for clBuildProgram and clCreateKernel
        if (kernelFunc->hasAttr<CUDAGlobalAttr>() && kernelFunc->hasBody()) {
            //If host-callable, get and store kernel filename
            llvm::StringRef r = SM->getFileEntryForID(SM->getFileID(kernelFunc->getLocation()))->getName();
            std::list<llvm::StringRef> &l = Kernels[r];
            l.push_back(kernelFunc->getName());
	    //Do a quick and dirty search that is linear w.r.t. the number of globalized (externed) variables in this source file
	    std::string decl = "cl_kernel __cu2cl_Kernel_" + kernelFunc->getName().str() + ";\n";
	    std::vector<std::string>::iterator j = GlobalCDecls[r].begin(), f = GlobalCDecls[r].end();
	    //Iterate to the end of the vector or til a matching string is found
	    for (; j != f && (*j) != decl; j++);
	
	    if (j == f) { // Not found, add declaration
                GlobalCDecls[r].push_back(decl);
            }
	
        }

        //Rewrite kernel attributes
	//__global__ must be mapped to __kernel
        if (CUDAGlobalAttr *attr = kernelFunc->getAttr<CUDAGlobalAttr>()) {
            RewriteAttr(attr, "__kernel", KernReplace);
        }
	//__device__ functions don't have any attributes in OpenCL
        if (CUDADeviceAttr *attr = kernelFunc->getAttr<CUDADeviceAttr>()) {
            RewriteAttr(attr, "", KernReplace);
        }
	//OpenCL kernel code has no such thing as a __host__ function
	// these are already preserved in the host code elsewhere
        if (CUDAHostAttr *attr = kernelFunc->getAttr<CUDAHostAttr>()) {
            RewriteAttr(attr, "", KernReplace);
        }

        //Rewrite formal parameters
        for (FunctionDecl::param_iterator PI = kernelFunc->param_begin(),
                                          PE = kernelFunc->param_end();
                                          PI != PE; ++PI) {
            RewriteKernelParam(*PI, kernelFunc->hasAttr<CUDAGlobalAttr>());
        }

        //Rewrite the body
        if (kernelFunc->hasBody()) {
            RewriteKernelStmt(kernelFunc->getBody());
        }
        CurRefParmVars.clear();
    }

    //Rewrite individual kernel arguments
    //this is primarily for tagging pointers to device buffers with the 
    // appropriate address space attribute
    void RewriteKernelParam(ParmVarDecl *parmDecl, bool isFuncGlobal) {

        if (parmDecl->getOriginalType()->isTemplateTypeParmType()) emitCU2CLDiagnostic(SM, parmDecl->getLocStart(), "CU2CL Unhandled", "Detected templated parameter", &KernReplace);
        TypeLoc tl = parmDecl->getTypeSourceInfo()->getTypeLoc();

	//A rewrite offset is declared to do bookkeeping for the amount of
	// of characters added. This prevents a bug in which consecutive
	// parameters would be overwritten
	int rewriteOffset = 0;
        if (isFuncGlobal && tl.getTypePtr()->isPointerType()) {
	    generateReplacement(KernReplace, SM, tl.getBeginLoc(), 0, "__global ");
		rewriteOffset -= 9; //ignore the 9 chars of "__global "
		rewriteOffset +=9; //FIXME: Revert this to diagnose range issues
        }
        else if (ReferenceTypeLoc rtl = tl.getAs<ReferenceTypeLoc>()) {
	    generateReplacement(KernReplace, SM, rtl.getSigilLoc(), getRangeSize(*SM, CharSourceRange::getTokenRange(rtl.getLocalSourceRange())), "*");
            CurRefParmVars.insert(parmDecl);
        }

	//scan forward to the last token in the parameter's type declaration
        while (!tl.getNextTypeLoc().isNull()) {
            tl = tl.getNextTypeLoc();
        }
        QualType qt = tl.getType();
        std::string type = qt.getAsString();

	//if it's a vector type, it must be checked for a rewrite
        std::string newType = RewriteVectorType(type, false);
        if (newType != "") {
            RewriteType(tl, newType, KernReplace, rewriteOffset);
	}
    }

    //The basic kernel rewriting driver, just walks the tree passing off
    // the real work of translation to the kernel expression and declaration
    // rewriters
    void RewriteKernelStmt(Stmt *ks) {
        //Visit this node
        if (Expr *e = dyn_cast<Expr>(ks)) {
            std::string str;
            if (RewriteKernelExpr(e, str)) {
                ReplaceStmtWithText(e, str, KernReplace);
            }
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(ks)) {
            DeclGroupRef DG = ds->getDeclGroup();
            for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    RewriteKernelVarDecl(vd);
                }
                //TODO other non-top level declarations??
            }
        }
        //TODO rewrite any other Stmts?

        else {
            //Traverse children and recurse
            for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end();
                 CI != CE; ++CI) {
                if (*CI)
                    RewriteKernelStmt(*CI);
            }
        }
    }

    bool RewriteKernelExpr(Expr *e, std::string &newExpr) {
        //Return value specifies whether or not a rewrite occurred
	//if for some reason the expression is in an invalid source range, abort
        if (e->getSourceRange().isInvalid())
            return false;

        //Rewriter used for rewriting subexpressions
        Rewriter exprRewriter(*SM, *LO);
        SourceRange realRange = SourceRange(SM->getExpansionLoc(e->getLocStart()), SM->getExpansionLoc(e->getLocEnd()));

        if (MemberExpr *me = dyn_cast<MemberExpr>(e)) {
            //Check base expr, if DeclRefExpr and a dim3, then rewrite
            if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(me->getBase())) {
                DeclaratorDecl *dd = dyn_cast<DeclaratorDecl>(dre->getDecl());
                TypeLoc tl = dd->getTypeSourceInfo()->getTypeLoc();
                while (!tl.getNextTypeLoc().isNull()) {
                    tl = tl.getNextTypeLoc();
                }
                QualType qt = tl.getType();
                std::string type = qt.getAsString();

                if (type == "dim3") {
                    std::string name = dre->getDecl()->getNameAsString();
                    if (name == "blockDim")
                        newExpr = "get_local_size";
                    else if (name == "gridDim")
                        newExpr = "get_num_groups";
                    else
                        newExpr = getStmtText(LO, SM, dre);

                    name = me->getMemberDecl()->getNameAsString();
                    if (newExpr != dre->getDecl()->getNameAsString()) {
                        if (name == "x")
                            name = "(0)";
                        else if (name == "y")
                            name = "(1)";
                        else if (name == "z")
                            name = "(2)";
                    }
                    else {
                        if (name == "x")
                            name = "[0]";
                        else if (name == "y")
                            name = "[1]";
                        else if (name == "z")
                            name = "[2]";
                    }
                    newExpr += name;
                    return true;
                }
                if (type == "uint3") {
                    std::string name = dre->getDecl()->getNameAsString();
                    if (name == "threadIdx")
                        newExpr = "get_local_id";
                    else if (name == "blockIdx")
                        newExpr = "get_group_id";
                    else
                        newExpr = getStmtText(LO, SM, dre);

                    name = me->getMemberDecl()->getNameAsString();
                    if (newExpr != dre->getDecl()->getNameAsString()) {
                        if (name == "x")
                            name = "(0)";
                        else if (name == "y")
                            name = "(1)";
                        else if (name == "z")
                            name = "(2)";
                        newExpr += name;
                        return true;
                    }
                    return false;
                }
            }
        }
        else if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(e)) {
	    //Register the DRE for potential late translation
		//PROP: Do kernel Exprs need to be stored?
	    //AllDeclRefsByDecl[dre->getDecl()].push_back(dre);
            //TODO if kernel makes reference to outside var, add arg
            //TODO if references warpSize, print warning
            if (ParmVarDecl *pvd = dyn_cast<ParmVarDecl>(dre->getDecl())) {
                if (CurRefParmVars.find(pvd) != CurRefParmVars.end()) {
                    newExpr = "(*" + exprRewriter.getRewrittenText(realRange) + ")";
                    return true;
                }
            }
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(e)) {
	    //If the expression involves a template, don't bother translating
	    //TODO: Support auto-generation of template specializations
	    if (ce->isTypeDependent()) {
                emitCU2CLDiagnostic(SM, e->getLocStart(), "CU2CL Unhandled", "Template-dependent kernel expression", &KernReplace);
                return false;
            }
	    //This catches potential segfaults related to function pointe usage
            if (ce->getDirectCallee() == 0) {
                emitCU2CLDiagnostic(SM, e->getLocStart(), "CU2CL Warning", "Unable to identify expression direct callee", &KernReplace);
                return false;
            }                

	    //This massive if-else tree catches all kernel API calls
            std::string funcName = ce->getDirectCallee()->getNameAsString();
            if (funcName == "__syncthreads") {
                newExpr = "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)";
            }

	    //begin single precision math API
            else if (funcName == "acosf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "acos(" + newX + ")";
            }
            else if (funcName == "acoshf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "acosh(" + newX + ")";
            }
            else if (funcName == "asinf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "asin(" + newX + ")";
            }
            else if (funcName == "asinhf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "asinh(" + newX + ")";
            }
            else if (funcName == "atan2f") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "atan2(" + newX + ", " + newY + ")";
            }
            else if (funcName == "atanf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "atan(" + newX + ")";
            }
            else if (funcName == "atanhf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "atanh(" + newX + ")";
            }
            else if (funcName == "cbrtf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cbrt(" + newX + ")";
            }
            else if (funcName == "ceilf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "ceil(" + newX + ")";
            }
            else if (funcName == "copysign") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "copysign(" + newX + ", " + newY + ")";
            }
            else if (funcName == "cosf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cos(" + newX + ")";
            }
            else if (funcName == "coshf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cosh(" + newX + ")";
            }
            else if (funcName == "cospif") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cospi(" + newX + ")";
            }
            else if (funcName == "erfcf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "erfc(" + newX + ")";
            }
	    //TODO: support erfcinvf, erfcxf
            else if (funcName == "erff") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "erf(" + newX + ")";
            }
	    //TODO: support erfinvf
            else if (funcName == "exp10f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "exp10(" + newX + ")";
            }
            else if (funcName == "exp2f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "exp2(" + newX + ")";
            }
            else if (funcName == "expf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "exp(" + newX + ")";
            }
            else if (funcName == "expm1f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "expm1(" + newX + ")";
            }
            else if (funcName == "fabsf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "fabs(" + newX + ")";
            }
            else if (funcName == "fdimf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fdim(" + newX + ", " + newY + ")";
            }
            else if (funcName == "fdividef") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "(" + newX + "/" + newY + ")";
            }
            else if (funcName == "floorf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "floor(" + newX + ")";
            }
            else if (funcName == "fmaf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(2);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "fma(" + newX + ", " + newY + ", " + newZ + ")";
            }
            else if (funcName == "fmaxf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fmax(" + newX + ", " + newY + ")";
            }
            else if (funcName == "fminf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fmin(" + newX + ", " + newY + ")";
            }
            else if (funcName == "fmodf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fmod(" + newX + ", " + newY + ")";
            }
            else if (funcName == "frexpf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "frexp(" + newX + ", " + newY + ")";
            }
	        else if (funcName == "hypotf") {
		        Expr *x = ce->getArg(0);
		        Expr *y = ce->getArg(1);
		        std::string newX, newY;
		        RewriteKernelExpr(x, newX);
		        RewriteKernelExpr(y, newY);
		        newExpr = "hypot(" + newX + ", " + newY + ")";
	        }
            else if (funcName == "ilogbf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "ilogb(" + newX + ")";
            }
            else if (funcName == "isfinite") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "isfinite(" + newX + ")";
            }
            else if (funcName == "isinf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "isinf(" + newX + ")";
            }
            else if (funcName == "isnan") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "isnan(" + newX + ")";
            }
	    //TODO: Support j0f, j1f, jnf - Bessel function of first kind order 0, 1, and n
            else if (funcName == "ldexpf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "ldexp(" + newX + ", " + newY + ")";
            }
            else if (funcName == "lgammaf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "lgamma(" + newX + ")";
            }
	    //TODO: suppot llrintf, llroundf - rounding with long long return type
            else if (funcName == "log10f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log10(" + newX + ")";
            }
            else if (funcName == "log1pf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log1p(" + newX + ")";
            }
            else if (funcName == "log2f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log2(" + newX + ")";
            }
            else if (funcName == "logbf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "logb(" + newX + ")";
            }
            else if (funcName == "logf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log(" + newX + ")";
            }
	    //TODO: support lrintf, lroundf - rounding with long return type
            else if (funcName == "modff") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "modf(" + newX + ", " + newY + ")";
            }
            else if (funcName == "nanf") {
                //WARNING: original cuda type of x is const char *, opencl is uintn
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "nan(" + newX + ")";
            }
	    //TODO: Support nearbyintf
            else if (funcName == "nextafterf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "nextafter(" + newX + ", " + newY + ")";
            }
            else if (funcName == "powf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "pow(" + newX + ", " + newY + ")";
            }
            else if (funcName == "rcbrtf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "(1/cbrt(" + newX + "))";
            }
            else if (funcName == "remainderf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "remainder(" + newX + ", " + newY + ")";
            }
            else if (funcName == "remquof") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(1);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "remquo(" + newX + ", " + newY + ", " + newZ + ")";
            }
            else if (funcName == "rintf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "rint(" + newX + ")";
            }
            else if (funcName == "roundf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "round(" + newX + ")";
            }
            else if (funcName == "rsqrtf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "rsqrt(" + newX + ")";
            }
	    //WARNING: Both scalbnf and scalblnf are not guaranteed to use the efficient "native" method of exponent manipulation, but are mathematically correct
            else if (funcName == "scalbnf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "ldexp(" + newX + ", " + newY + ")";
            }
            else if (funcName == "scalblnf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "ldexp(" + newX + ", " + newY + ")";
            }
            else if (funcName == "signbit") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "signbit(" + newX + ")";
            }
            else if (funcName == "sincosf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(2);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "(*" + newY + " = sincos(" + newX + ", " + newZ + "))";
            }
            else if (funcName == "sinf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sin(" + newX + ")";
            }
            else if (funcName == "sinhf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sinh(" + newX + ")";
            }
            else if (funcName == "sinpif") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sinpi(" + newX + ")";
            }
            else if (funcName == "sqrtf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sqrt(" + newX + ")";
            }
            else if (funcName == "tanf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "tan(" + newX + ")";
            }
            else if (funcName == "tanhf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "tanh(" + newX + ")";
            }
            else if (funcName == "tgammaf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "tgamma(" + newX + ")";
            }
            else if (funcName == "truncf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "trunc(" + newX + ")";
            }
	    //TODO: Support y0f, y1f, ynf - Bessel function of first kind order 0, 1, and n

	    //Begin double precision
	    //These are only "translated" to ensure nested expressions get translated
            else if (funcName == "acos") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "acos(" + newX + ")";
            }
            else if (funcName == "acosh") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "acosh(" + newX + ")";
            }
            else if (funcName == "asin") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "asin(" + newX + ")";
            }
            else if (funcName == "asinh") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "asinh(" + newX + ")";
            }
            else if (funcName == "atan") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "atan(" + newX + ")";
            }
            else if (funcName == "atan2") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "atan(" + newX + ", " + newY + ")";
            }
            else if (funcName == "atanh") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "atanh(" + newX + ")";
            }
            else if (funcName == "cbrt") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cbrt(" + newX + ")";
            }
            else if (funcName == "ceil") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "ceil(" + newX + ")";
            }
	    //NOTE: Copysign is already handled in floating point section
            else if (funcName == "cos") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cos(" + newX + ")";
            }
            else if (funcName == "cosh") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cosh(" + newX + ")";
            }
            else if (funcName == "cospi") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "cospi(" + newX + ")";
            }
            else if (funcName == "erf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "erf(" + newX + ")";
            }
            else if (funcName == "erfc") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "erfc(" + newX + ")";
            }
	    //TODO: support erfinv, erfcinv, erfcx
            else if (funcName == "exp") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "exp(" + newX + ")";
            }
            else if (funcName == "exp10") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "exp10(" + newX + ")";
            }
            else if (funcName == "exp2") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "exp2(" + newX + ")";
            }
            else if (funcName == "expm1") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "expm1(" + newX + ")";
            }
            else if (funcName == "fabs") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "fabs(" + newX + ")";
            }
            else if (funcName == "fdim") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fdim(" + newX + ", " + newY + ")";
            }
            else if (funcName == "floor") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "floor(" + newX + ")";
            }
            else if (funcName == "fma") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(2);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "fma(" + newX + ", " + newY + ", " + newZ + ")";
            }
            else if (funcName == "fmax") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fmax(" + newX + ", " + newY + ")";
            }
            else if (funcName == "fmin") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fmin(" + newX + ", " + newY + ")";
            }
            else if (funcName == "fmod") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "fmod(" + newX + ", " + newY + ")";
            }
            else if (funcName == "frexp") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "frexp(" + newX + ", " + newY + ")";
            }
            else if (funcName == "hypot") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "hypot(" + newX + ", " + newY + ")";
            }
            else if (funcName == "ilogb") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "ilogb(" + newX + ")";
            }
	    //NOTE: isfinite, isinf, and isnan are all handled in floating point section
	    //TODO: support j0, j1, jn - Bessel functions of the first kind of order 0, 1, and n
            else if (funcName == "ldexp") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "ldexp(" + newX + ", " + newY + ")";
            }
            else if (funcName == "lgamma") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "lgamma(" + newX + ")";
            }
	    //TODO: support llrint, llround
            else if (funcName == "log") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log(" + newX + ")";
            }
            else if (funcName == "log10") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log10(" + newX + ")";
            }
            else if (funcName == "log1p") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log1p(" + newX + ")";
            }
            else if (funcName == "log2") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "log2(" + newX + ")";
            }
            else if (funcName == "logb") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "logb(" + newX + ")";
            }
	    //TODO: support lrint, lround
            else if (funcName == "modf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "modf(" + newX + ", " + newY + ")";
            }
	    //NOTE: nan is handled in floating point section
	    //TODO: Support nearbyint
            else if (funcName == "nextafter") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "nextafter(" + newX + ", " + newY + ")";
            }
            else if (funcName == "pow") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "pow(" + newX + ", " + newY + ")";
            }
            else if (funcName == "rcbrt") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "(1/cbrt(" + newX + "))";
            }
            else if (funcName == "remainder") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "remainder(" + newX + ", " + newY + ")";
            }
            else if (funcName == "remquo") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(1);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "remquo(" + newX + ", " + newY + ", " + newZ + ")";
            }
            else if (funcName == "rint") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "rint(" + newX + ")";
            }
            else if (funcName == "round") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "round(" + newX + ")";
            }
            else if (funcName == "rsqrt") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sqrt(" + newX + ")";
            }
	    //WARNING: Both scalbnf and scalblnf are not guaranteed to use the efficient "native" method of exponent manipulation, but are mathematically correct
            else if (funcName == "scalbn") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "ldexp(" + newX + ", " + newY + ")";
            }
            else if (funcName == "scalbln") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "ldexp(" + newX + ", " + newY + ")";
            }
	    //NOTE: signbit is already handled in the float section
            else if (funcName == "sin") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sin(" + newX + ")";
            }
            else if (funcName == "sincos") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(2);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "(*" + newY + " = sincos(" + newX + ", " + newZ + "))";
            }
            else if (funcName == "sinh") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sinh(" + newX + ")";
            }
            else if (funcName == "sinpi") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sinpi(" + newX + ")";
            }
            else if (funcName == "sqrt") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sqrt(" + newX + ")";
            }
            else if (funcName == "tan") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "tan(" + newX + ")";
            }
            else if (funcName == "tanh") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "tanh(" + newX + ")";
            }
            else if (funcName == "tgamma") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "tgamma(" + newX + ")";
            }
            else if (funcName == "trunc") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "trunc(" + newX + ")";
            }
	    //TODO: support y0, y1, yn

	    //Begin native floats
            else if (funcName == "__cosf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_cos(" + newX + ")";
            }
            else if (funcName == "__exp10f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_exp10(" + newX + ")";
            }
            else if (funcName == "__expf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_exp(" + newX + ")";
            }
	    //TODO: support fadd and fdiv with rounding modes
	    else if (funcName == "__fdividef") {
		Expr *x = ce->getArg(0);
		Expr *y = ce->getArg(1);
		std::string newX, newY;
		RewriteKernelExpr(x, newX);
		RewriteKernelExpr(y, newY);
		newExpr = "native_divide(" + newX + ", " + newY + ")";
	    }
	    //TODO: support fmaf, fmul, frcp, and fsqrt with rounding modes
            else if (funcName == "__log10f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_log10(" + newX + ")";
            }
            else if (funcName == "__log2f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_log2(" + newX + ")";
            }
            else if (funcName == "__logf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_log(" + newX + ")";
            }
            else if (funcName == "__powf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "native_powr(" + newX + ", " + newY + ")";
            }
	    //NOTE: does not use intrinsics, but returns an equivalent value
            else if (funcName == "__saturatef") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "clamp(" + newX + "0.0f, 1.0f)";
            }
            else if (funcName == "__sinf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_sin(" + newX + ")";
            }
	    //NOTE: does not use intrinsics, but returns an equivalent value
            else if (funcName == "__sincosf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                Expr *z = ce->getArg(2);
                std::string newX, newY, newZ;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                RewriteKernelExpr(z, newZ);
                newExpr = "(*" + newY + " = sincos(" + newX + ", " + newZ + "))";
            }
            else if (funcName == "__tanf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_tan(" + newX + ")";
            }
	    //Begin double intrinsics
	    //TODO: support double intrinsics
	    //Begin integer intrinsics
	    //TODO: support integer intrinsics
	    //Begin type casting intrinsics
            else if (funcName == "__double2float_rd") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_float_rtn(" + newX + ")";
            }
            else if (funcName == "__double2float_rn") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_float_rte(" + newX + ")";
            }
            else if (funcName == "__double2float_ru") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_float_rtp(" + newX + ")";
            }
            else if (funcName == "__double2float_rz") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_float_rtz(" + newX + ")";
            }
	    //TODO: support __double2hiint
            else if (funcName == "__double2int_rd") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_int_rtn(" + newX + ")";
            }
            else if (funcName == "__double2int_rn") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_int_rte(" + newX + ")";
            }
            else if (funcName == "__double2int_ru") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_int_rtp(" + newX + ")";
            }
            else if (funcName == "__double2int_rz") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "convert_int_rtz(" + newX + ")";
            }
            else {
		//TODO: Make sure every possible function call goes through here, or else we may not get rewrites on interior nested calls.
		// any unsupported call should throw an error, but still convert interior nesting.
                return false;
            }
            return true;
        }
        else if (CXXFunctionalCastExpr *cfce = dyn_cast<CXXFunctionalCastExpr>(e)) {
            //TODO rewrite type before wrapping it
            TypeLoc tl = cfce->getTypeInfoAsWritten()->getTypeLoc();
            exprRewriter.ReplaceText(
                    tl.getBeginLoc(),
                    exprRewriter.getRangeSize(tl.getSourceRange()),
                    "(" + tl.getType().getAsString() + ")");

            //Rewrite subexpression
            std::string s;
            if (RewriteHostExpr(cfce->getSubExpr(), s))
                ReplaceStmtWithText(cfce->getSubExpr(), s, exprRewriter);
            newExpr = exprRewriter.getRewrittenText(realRange);
            return true;
        }

        bool ret = false;
        //Do a DFS, recursing into children, then rewriting this expression
        //if rewrite happened, replace text at old sourcerange
        for (Stmt::child_iterator CI = e->child_begin(), CE = e->child_end();
             CI != CE; ++CI) {
            std::string s;
            Expr *child = (Expr *) *CI;
            if (child && RewriteKernelExpr(child, s)) {
                //Perform "rewrite", which is just a simple replace
                ReplaceStmtWithText(child, s, exprRewriter);
                ret = true;
            }
        }
				
        newExpr = exprRewriter.getRewrittenText(realRange);
        return ret;
    }

    void RewriteKernelVarDecl(VarDecl *var) {
        //TODO handle extern __shared__ memory pointers
        if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
            RewriteAttr(sharedAttr, "__local", KernReplace);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", KernReplace);
            //TODO rewrite extern shared mem
            //if (var->isExtern()) {
		//handle the addition of a __local address space kernel param
	    //}
        }

        TypeLoc origTL = var->getTypeSourceInfo()->getTypeLoc();

        TypeLoc tl = origTL;
        while (!tl.getNextTypeLoc().isNull()) {
            tl = tl.getNextTypeLoc();
        }
        QualType qt = tl.getType();
        std::string type = qt.getAsString();

        //Rewrite var type
        if (LastLoc.isNull() || origTL.getBeginLoc() != LastLoc.getBeginLoc()) {
            LastLoc = origTL;
            if (type == "dim3") {
                //Rewrite to size_t[3] array
                RewriteType(tl, "size_t", KernReplace);
            }
            else {
                std::string newType = RewriteVectorType(type, false);
                if (newType != "")
                    RewriteType(tl, newType, KernReplace);
            }
            //TODO check other CUDA-only types to rewrite
        }

        //Rewrite initial value
        if (var->hasInit()) {
            Expr *e = var->getInit();
            std::string s;
            if (RewriteKernelExpr(e, s)) {
                //Special cases for dim3s
                if (type == "dim3") {
                    //TODO fix case of dim3 c = b;
                    CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(e);
                    if (cce && cce->getNumArgs() > 1) {
                        SourceRange parenRange = cce->getParenOrBraceRange();
                        if (parenRange.isValid()) {
	    		    generateReplacement(KernReplace, SM, parenRange.getBegin(), getRangeSize(*SM, CharSourceRange::getTokenRange(parenRange)), s);
                        }
                        else {
	    		    generateReplacement(KernReplace, SM, PP->getLocForEndOfToken(var->getLocation()), 0, s);
                        }
                    }
                    else
                        ReplaceStmtWithText(e, s, KernReplace);

                    //Add [3]/* to end/start of var identifier
                    if (origTL.getTypePtr()->isPointerType())
	    		generateReplacement(KernReplace, SM, var->getLocation(), 0, "*");
                    else
	    		generateReplacement(KernReplace, SM, PP->getLocForEndOfToken(var->getLocation()), 0, "[3]");
                }
                else
                    ReplaceStmtWithText(e, s, KernReplace);
            }
        }
    }

    //TODO: Add an option for OpenCL >= 1.1 to keep 3-member vectors
    std::string RewriteVectorType(std::string type, bool addCL) {
        std::string prepend, append, ret;
        char size = type[type.length() - 1];
        switch (size) {
            case '1':
            case '2':
            case '3':
            case '4':
                break;
            default:
                return "";
        }

        if (addCL)
            prepend = "cl_";
        if (type[0] == 'u')
            prepend += "u";
        if (size == '3') //Only necessary when supporting OpenCL 1.0, otherwise 3 member vectors are supported
            append = '4';
        else if (size != '1')
            append = size;

        llvm::Regex *regex = new llvm::Regex("^u?char[1-4]$");
        if (regex->match(type)) {
            ret = prepend + "char" + append;
        }
        delete regex;
        regex = new llvm::Regex("^u?short[1-4]$");
        if (regex->match(type)) {
            ret = prepend + "short" + append;
        }
        delete regex;
        regex = new llvm::Regex("^u?int[1-4]$");
        if (regex->match(type)) {
            ret = prepend + "int" + append;
        }
        delete regex;
        regex = new llvm::Regex("^u?long[1-4]$");
        if (regex->match(type)) {
            ret = prepend + "long" + append;
        }
        delete regex;
        regex = new llvm::Regex("^u?float[1-4]$");
        if (regex->match(type)) {
            ret = prepend + "float" + append;
        }
        delete regex;	
        return ret;
    }

    //The workhorse that takes the constructed replacement type and inserts it in place of the old one
    //RewriteType requires a rangeOffset parameter to account for a case in which
    // a rewrite to the type has already occured before we get here (i.e. adding "__global " requires an offset of -9)
    void RewriteType(TypeLoc tl, std::string replace, std::vector<Replacement> &replacements, int rangeOffset = 0) {
	SourceRange realRange(tl.getBeginLoc(), PP->getLocForEndOfToken(tl.getBeginLoc()));
	generateReplacement(replacements, SM, tl.getBeginLoc(), getRangeSize(*SM, CharSourceRange::getTokenRange(tl.getLocalSourceRange()))+rangeOffset, replace);
    }

    //Rewrite Type also needs a form that still takes a Rewriter
    // so that local Expr Rewriters can be used
    void RewriteType(TypeLoc tl, std::string replace, Rewriter &rewrite, int rangeOffset = 0) {

	SourceRange realRange(tl.getBeginLoc(), PP->getLocForEndOfToken(tl.getBeginLoc()));

	bool status = rewrite.ReplaceText(tl.getBeginLoc(), rewrite.getRangeSize(tl.getLocalSourceRange()) + rangeOffset, replace);
    }

    //The workhorse that takes the constructed replacement attribute and inserts it in place of the old one
    void RewriteAttr(Attr *attr, std::string replace, std::vector<Replacement> &replacements) {
	SourceLocation instLoc;
	        instLoc = SM->getExpansionLoc(attr->getLocation());

	

        SourceRange realRange(instLoc,
                              PP->getLocForEndOfToken(instLoc));
	generateReplacement(replacements, SM, instLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(realRange)), replace);
	
    }

    //Completely remove a function from one of the output streams
    //Used to split host and kernel code
    void RemoveFunction(FunctionDecl *func, std::vector<Replacement> &replace) {
        SourceLocation startLoc, endLoc, tempLoc;

        FunctionDecl::TemplatedKind tk = func->getTemplatedKind();
        if (tk != FunctionDecl::TK_NonTemplate &&
            tk != FunctionDecl::TK_FunctionTemplate)
            return;

	//If a function has a prototype declaration AND a definition, skip ahead to the definition
	// this prevents a bug where all text between the prototype and the definition would be deleted
        const FunctionDecl * funcDef = func;
        if (func->hasBody()) {
	    func->hasBody(funcDef);
	    func = (FunctionDecl *)funcDef;
	}

	//Start with reasonable defaults
        startLoc = func->getLocStart();
	//endLoc = func->getLocEnd();
		
        //Calculate the SourceLocation of the first token of the function,
	// handling a number of corner cases
        //TODO find first specifier location
        //TODO find storage class specifier
        tempLoc = func->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
	if (SM->isBeforeInTranslationUnit(tempLoc, startLoc)) startLoc = tempLoc;
        if (tk == FunctionDecl::TK_FunctionTemplate) {
            FunctionTemplateDecl *ftd = func->getDescribedFunctionTemplate();
            tempLoc = ftd->getSourceRange().getBegin();
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }
        if (func->hasAttrs()) {
	    //Attributes are stored in reverse order of spelling, get the outermost
            Attr *attr = *(func->attr_end()-1);
	    //Some functions have attributes on both prototype and definition.
	    // This loop ensures we grab the LAST copy of the first attribute
	    int i;
            for (i = 1; i < func->getAttrs().size(); i++) {
                if ((func->getAttrs())[i]->getKind() == attr->getKind()) attr = (func->getAttrs())[i];
            }
            tempLoc = SM->getExpansionLoc(attr->getLocation());
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }
	
	//C++ Constructors and Destructors are not explicitly typed,
	// and may not have attributes. This if block catches them
	if (dyn_cast<CXXConstructorDecl>(func) || dyn_cast<CXXDestructorDecl>(func)) {
            startLoc = func->getQualifierLoc().getBeginLoc();
            if (startLoc.getRawEncoding() != 0) emitCU2CLDiagnostic(SM, startLoc, "CU2CL Note", "Removed constructor/deconstructor", (replace == HostReplace ? &HostReplace : &KernReplace));
        }

	//If there is an extern "C"  qualifier on a function declaration
	// move startLoc back to it (unless it is a block extern)
	if (func->isExternC()) {
	    //Get the function's DeclContext
	    DeclContext * dc = FunctionDecl::castToDeclContext(func);
	    //Find the nearest ancestor that is a terminal extern "C" LinkageSpecDecl
	    //I have not found a case in which this loop iterates more than once
	    // i.e. it always find the extern on the immediate parent
	    // but it remains in-case that is not always the case
	    for (; dc->getParent()->isExternCContext() && !(LinkageSpecDecl::castFromDeclContext(dc)->getExternLoc().isValid()); dc = dc->getParent());
	    //Make a LinkageSpecDecl from the ancestor
	    LinkageSpecDecl * lsd = LinkageSpecDecl::castFromDeclContext(dc);
	    //Exclude block "extern "C" { ... }" variants as these are often quite a distance from the immediate function being deleted.
	    if (!(lsd->hasBraces()) && (tempLoc = lsd->getExternLoc()).isValid()) startLoc = tempLoc;
	}

	//If we still haven't found an appropriate startLoc, something's atypical
	//Grab whatever Clang thinks is the startLoc, and remove from there
        if (startLoc.getRawEncoding() == 0) {
	    startLoc = func->getLocStart();
	    //If even Clang doesn't have any idea where to start, give up
            if (startLoc.getRawEncoding() == 0) {
                emitCU2CLDiagnostic(SM, startLoc, "CU2CL Error", "Unable to determine valid start location for function \"" + func->getNameAsString() + "\"", (replace == HostReplace ? &HostReplace : &KernReplace));
                return;
            }
            emitCU2CLDiagnostic(SM, startLoc, "CU2CL Warning", "Inferred function start location, removal may be incomplete", (replace == HostReplace ? &HostReplace : &KernReplace));
        }

        //Calculate the SourceLocation of the closing brace if it's a definition
        if (func->hasBody()) {
            CompoundStmt *body = (CompoundStmt *) func->getBody();
            endLoc = body->getRBracLoc();
        }
	// or the semicolon if it's just a declaration
        else {
            //Find location of semi-colon
            endLoc = func->getSourceRange().getEnd();
	    if ((tempLoc = Lexer::findLocationAfterToken(endLoc, tok::semi, *SM, *LO, false)).isValid()) {
		//found a semicolon, replace the endLoc with the semicolon's loc
		endLoc = tempLoc;
	    }
        }
	generateReplacement(replace, SM, startLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(startLoc,endLoc))), "");
    }

    //Get rid of a variable declaration
    // useful for pulling global host variables out of kernel code
    void RemoveVar(VarDecl *var, std::vector<Replacement> &replace) {
        SourceLocation startLoc, endLoc, tempLoc;

        //Find startLoc
	//Try just getting the raw startLoc, should grab storage specifiers
	// (i.e. "extern", "const", et.)
	startLoc = var->getLocStart();
        tempLoc = var->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
	if (SM->isBeforeInTranslationUnit(tempLoc, startLoc) || !startLoc.isValid()) {
		startLoc = tempLoc;
	}
        if (var->hasAttrs()) {
	    //Find any __shared__, __constant__, __device__, or other attribs
	    //Attributes are stored in reverse order of spelling, get the outermost
            Attr *attr = *(var->attr_end()-1);
            tempLoc = SM->getExpansionLoc(attr->getLocation());
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }
	
	//default to the perceived end
	endLoc = var->getLocEnd();
        //Check if an initializer is accounted for
        if (var->hasInit()) {
            Expr *init = var->getInit();
            tempLoc = SM->getExpansionLoc(init->getLocEnd());
	    if (SM->isBeforeInTranslationUnit(endLoc, tempLoc) || !endLoc.isValid()) endLoc = tempLoc;
        }
        else {
            //Find the end of the declaration
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            if (ArrayTypeLoc atl = tl.getAs<ArrayTypeLoc>()) {
                tempLoc = SM->getExpansionLoc(atl.getRBracketLoc());
		if (SM->isBeforeInTranslationUnit(endLoc, tempLoc) || !endLoc.isValid()) endLoc = tempLoc;
		//llvm::errs() << "Found ArrayTypeLoc\n";
		
            }
            else {
                tempLoc = SM->getExpansionLoc(var->getSourceRange().getEnd());
		if (SM->isBeforeInTranslationUnit(endLoc, tempLoc) || !endLoc.isValid()) endLoc = tempLoc;
		//llvm::errs() << "Found non-array TypeLoc\n";
	    }
        }
	//then find the semicolon
	if ((tempLoc = Lexer::findLocationAfterToken(endLoc, tok::semi, *SM, *LO, false)).isValid()) {
	    //found a semicolon, replace the endLoc with the semicolon's loc
	    endLoc = tempLoc;
	}
	//TODO Read ahead to the trailing newline if no active code elements are between it and the semicolon
	// (i.e. remove trailing comments identifying the variable and the newline)
	generateReplacement(replace, SM, startLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(startLoc,endLoc))), "");
	//replace.push_back(Replacement(*SM, startLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(startLoc, endLoc))), ""));
    }

    //DEPRECATED: Old method to get the string representation of a Stmt
    std::string PrintStmtToString(Stmt *s) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        s->printPretty(S, 0, PrintingPolicy(*LO));
        return S.str();
    }

    //DEPRECATED: Old method to get the string representation of a Decl
    //TODO: Test replacing the one remaining usage in HandleTranslationUnit with getStmtText 
    std::string PrintDeclToString(Decl *d) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        d->print(S);
        return S.str();
    }

    //Replace a chunk of code represented by a Stmt with a constructed string
    bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr, std::vector<Replacement> &replace) {
        SourceRange origRange = OldStmt->getSourceRange();
        SourceLocation s = SM->getExpansionLoc(origRange.getBegin());
        SourceLocation e = SM->getExpansionLoc(origRange.getEnd());
	//FIXME: Originally, the rewriter method of replacements would return true if for some reason the SourceLocation could not be rewriten, need to make sure switching to Replacements and ASSUMING the location is rewritable is acceptable
	generateReplacement(replace, SM, s, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(s, e))), NewStr);
	return false;
    }

    //ReplaceStmtWithText still needs a Rewriter form so that localized Expr Rewriters can work
    bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr, Rewriter &rewrite) {
        SourceRange origRange = OldStmt->getSourceRange();
        SourceLocation s = SM->getExpansionLoc(origRange.getBegin());
        SourceLocation e = SM->getExpansionLoc(origRange.getEnd());
        return rewrite.ReplaceText(s,
                                   rewrite.getRangeSize(SourceRange(s, e)),
                                   NewStr);
    }

    //Replaces non-alphanumeric characters in a string with underscores
    std::string idCharFilter(llvm::StringRef ref) {
        std::string str = ref.str();
        size_t size = ref.size();
        for (size_t i = 0; i < size; i++)
            if (!isalnum(str[i]) && str[i] != '_')
                str[i] = '_';
        return str;
    }


public:
    RewriteCUDA(CompilerInstance *comp, std::string origFilename, OutputFile * HostOS,
                OutputFile * KernelOS) : mainFilename(origFilename),
        ASTConsumer(), CI(comp),
        MainOutFile(HostOS), MainKernelOutFile(KernelOS) { }

    virtual ~RewriteCUDA() { }

    virtual void Initialize(ASTContext &Context) {
        SM = &Context.getSourceManager();
	SM->Retain(); //Retain the SourceManager so we can use it at the tool layer
	AllSMs.push_back(SM);
	Context.Retain(); //Retain the context so that it remains valid once we return to the tool layer
	AllASTs.push_back(&Context);
        LO = &CI->getLangOpts();
        PP = &CI->getPreprocessor();

	PP->Retain();
	LO->Retain();
	ST = new SourceTuple(SM, PP, LO, &Context);

        PP->addPPCallbacks(new RewriteIncludesCallback(this));

        HostRewrite.setSourceMgr(*SM, *LO);
        KernelRewrite.setSourceMgr(*SM, *LO);
        MainFileID = SM->getMainFileID();
	
        OutFiles[mainFilename] = MainOutFile;
        KernelOutFiles[mainFilename] = MainKernelOutFile;

        if (MainFuncName == "")
            MainFuncName = "main";
	//Ensure that each time a new RewriteCUDA instance is spawned this gets reset
	MainDecl = NULL;

        HostIncludes += "#ifdef __APPLE__\n";
        HostIncludes += "#include <OpenCL/opencl.h>\n";
        HostIncludes += "#else\n";
        HostIncludes += "#include <CL/opencl.h>\n";
        HostIncludes += "#endif\n";
        HostIncludes += "#include <stdlib.h>\n";
        HostIncludes += "#include <stdio.h>\n";
	HostIncludes += "#include \"cu2cl_util.h\"\n";
	//This isn't actually to check if the map has an empty vector for this file
	// rather it is to force a key to be generated for this file, so that it turns up in the output stage
	GlobalCDecls[mainFilename].empty();
	//TODO consider making this default
	// we will almost always need to load a kernel file
	if(!UsesCU2CLLoadSrc) {
	    GlobalCFuncs.push_back(LOAD_PROGRAM_SOURCE);
	    GlobalHDecls.push_back(LOAD_PROGRAM_SOURCE_H);
	    UsesCU2CLLoadSrc = true;
	}

	//Hoisted to Tool level, no longer initialize here
        IncludingStringH = false;

	//Set up the simple linked-list for buffering inserted comments
	head = (struct commentBufferNode *)malloc(sizeof(struct commentBufferNode));
	head->n = NULL;
    	tail = head;
    
	TransTime = 0;
}

    //HandleTopLevelDecl is triggered by Clang's AST walking machinery for each
    // globally-scoped declaration, be it a function, variable, class or whatever
    //It is responsible for identifying the source file each is in, and generating
    // *-cl.h and *-cl.cl host and kernel include files as needed for .cu and .cuh files
    //After identifying what manner of declaration it is, control is passed to
    // the relevant host and kernel rewriters
    virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
        //Check where the declaration(s) comes from (may have been included)
        Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
        SourceLocation loc = firstDecl->getLocation();
	//TODO: early abort if the file is cuda.h, cuda_runtime.h, or #included <> with angle braces
        std::string FileName = SM->getPresumedLoc(loc).getFilename();

	//TODO check if the file was included by any file matching the below criteria	
	if (isInBannedInclude(loc, SM, LO)) {
		//llvm::errs() << " will not be translated!\n";
		//it's a forbidden file, just skip the file
		return true;
	}
	
        if (!SM->isInMainFile(loc)) {
            llvm::StringRef fileExt = extension(SM->getPresumedLoc(loc).getFilename());
                if (OutFiles.find(SM->getPresumedLoc(loc).getFilename()) == OutFiles.end()) {
                    //Create new files
                    FileID fileid = SM->getFileID(loc);
		    std::string origFilename = FileName;
                    size_t dotPos = FileName.rfind('.');
		    FileName = kernelNameFilter(FileName) + "-cl" + FileName.substr(dotPos);
			//PAUL: These calls had to be replaced so the CompilerInstance wouldn't destroy the raw_ostream after translation finished
		    std::string error, HostOutputPathName, HostTempPathName, KernOutputPathName, KernTempPathName;
		    llvm::raw_ostream *hostOS = CI->createOutputFile(StringRef(CI->getFrontendOpts().OutputFile), error, false, true, FileName, "h", true, true, &HostOutputPathName, &HostTempPathName);
		    llvm::raw_ostream *kernelOS = CI->createOutputFile(StringRef(CI->getFrontendOpts().OutputFile), error, false, true, FileName, "cl", true, true, &KernOutputPathName, &KernTempPathName);
			OutputFile *HostOF = new OutputFile(HostOutputPathName, HostTempPathName, hostOS);
			OutputFile *KernOF = new OutputFile(KernOutputPathName, KernTempPathName, kernelOS);
                    if (hostOS && kernelOS) {
                        OutFiles[origFilename] = HostOF;
                        KernelOutFiles[origFilename] = KernOF;
                    }
                    else {
			//We've already registered an output stream for this
			// input file, so proceed
                    }
                }
        }
        //Store VarDecl DeclGroupRefs
        if (firstDecl->getKind() == Decl::Var) {
            GlobalVarDeclGroups.insert(DG);
        }
        //Walk declarations in group and rewrite
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            if (DeclContext *dc = dyn_cast<DeclContext>(*i)) {
                //Basically only handles C++ member functions
                for (DeclContext::decl_iterator di = dc->decls_begin(), de = dc->decls_end();
                     di != de; ++di) {
                    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*di)) {
                        //prevent implicitly defined functions from being rewritten
			// (since there's no source to rewrite..)
                        if (!fd->isImplicit()) {
                            RewriteHostFunction(fd);
                            RemoveFunction(fd, KernReplace);
    
                            if (fd->getNameAsString() == MainFuncName) {
                                RewriteMain(fd);
                            }
                        } else {
                            emitCU2CLDiagnostic(SM, fd->getLocStart(), "CU2CL Note", "Skipped rewrite of implicitly defined function \"" + fd->getNameAsString() + "\"", &HostReplace);
                        }
                    }
                }
            }
	    //Handle both templated and non-templated function declarations
	    FunctionDecl *fd = dyn_cast<FunctionDecl>(*i);
	    if (fd == NULL) {
		FunctionTemplateDecl *ftd = dyn_cast<FunctionTemplateDecl>(*i);
		if (ftd) fd = ftd->getTemplatedDecl();
	    }
            //Handles globally defined C or C++ functions
            if (fd) {
		//Don't translate explicit template specializations
                if(fd->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate || fd->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    //Device function, so rewrite kernel
                        RewriteKernelFunction(fd);
                        if (fd->hasAttr<CUDAHostAttr>())
                            //Also a host function, so rewrite host
                            RewriteHostFunction(fd);
                        else
                            //Simply a device function, so remove from host
                            RemoveFunction(fd, HostReplace);
                    }
                    else {
                        //Simply a host function, so rewrite
                        RewriteHostFunction(fd);
                        //and remove from kernel
                        RemoveFunction(fd, KernReplace);
    
                        if (fd->getNameAsString() == MainFuncName) {
                            RewriteMain(fd);
                        }
                    }
                } else {
                    if (fd->getTemplateSpecializationInfo())
                    emitCU2CLDiagnostic(SM, fd->getTemplateSpecializationInfo()->getTemplate()->getLocStart(), "CU2CL Untranslated", "Unable to translate template function", &HostReplace);
                    else llvm::errs() << "Non-rewriteable function without TemplateSpecializationInfo detected\n";
                }
            }
            else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                RemoveVar(vd, KernReplace);
                RewriteHostVarDecl(vd);
            }
            //Rewrite Structs here
            //Ideally, we should keep the expression inside parentheses ie __align__(<keep this>)
            // and just wrap it with __attribute__((aligned (<kept Expr>)))
	    //TODO: Finish struct attribute translation
            else if (RecordDecl * rd = dyn_cast<RecordDecl>(*i)) {
                if (rd->hasAttrs()) {
                    for (Decl::attr_iterator at = rd->attr_begin(), at_e = rd->attr_end(); at != at_e; ++at) {
                        if (AlignedAttr *align = dyn_cast<AlignedAttr>(*at)) {
                            if (!align->isAlignmentDependent()) {
                                llvm::errs() << "Found an aligned struct of size: " << align->getAlignment(rd->getASTContext()) << " (bits)\n";
                            } else {
                                llvm::errs() << "Found a dependent alignment expresssion\n";
                            }
                        } else {
                            llvm::errs() << "Found other attrib\n";
                        }
                    }
                }
            }
	    else if (TypedefDecl *tdd = dyn_cast<TypedefDecl>(*i)) {
		//Just catch typedefs, do nothing (yet)
		//Eventually, check if the base type is CUDA-specific
	    }
	    else if (EnumDecl *ed = dyn_cast<EnumDecl>(*i)) {
		//Just catch enums, do nothing (yet)
		//Eventually, check if anything inside is CUDA-specific
	    }
	    else if (LinkageSpecDecl *lsd = dyn_cast<LinkageSpecDecl>(*i)) {
		//Just catch externs, do nothing (yet)
		//Eventually, replace (not rewrite) any pieces that are CUDA and trust that the source file that *implemented* the call is translated similarly
	    }
	    else if (EmptyDecl *ed = dyn_cast<EmptyDecl>(*i)) {
		//For some reason, the phrase   extern "C" {
                // is treated as an "Empty Declaration" in the 3.4 AST
		// so once we do something with externs, consider treating them as well
	    }
	    else if (ClassTemplateDecl *ctd = dyn_cast<ClassTemplateDecl>(*i)) {
		//These will likely need to be rewritten, at least internally, eventually
	    }
	    else if (NamespaceDecl *nsd = dyn_cast<NamespaceDecl>(*i)) {
		//Catch them, just so the catchall fires, not likely to do anything with it
	    }
	    else if (UsingDirectiveDecl *udd = dyn_cast<UsingDirectiveDecl>(*i)) {
		//Just catch using directives, don't do anythign with them
		//Eventually, these will need to be pulled from device code, if they're not already
	    }
	    else if (UsingDecl *ud = dyn_cast<UsingDecl>(*i)) {
		//Similar to UsingDirectiveDecl, just pull out of kernel code
	    }
	    else {
		//This catches everything else, including enums
		emitCU2CLDiagnostic(SM, (*i)->getLocStart(), "CU2CL DEBUG", "Decl couldn't be determined", &HostReplace);
	    }
            //TODO rewrite type declarations
        }
return true;
    }

    //Compltely processes each file included on the invokation command line
    virtual void HandleTranslationUnit(ASTContext &) {
	#ifdef CU2CL_ENABLE_TIMING
        	init_time();
	#endif

        //Declare global clPrograms, one for each kernel-bearing source file
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string r = idCharFilter(filename((*i).first));
	    //Insert the cl_program variable for this file IFF it hasn't already been inserted
	    //Paul: This search is linear w.r.t the number of global "extern"-needed variables in the current source file
	    std::string decl = "cl_program __cu2cl_Program_" + r + ";\n";
	    std::vector<std::string>::iterator j = GlobalCDecls[(*i).first].begin(), f = GlobalCDecls[(*i).first].end();
	    //Iterate to the end of the vector or til a matching string is found
	    for (; j != f && (*j) != decl; j++);
	
	    if (j == f) { // Not found, add declaration
                GlobalCDecls[(*i).first].push_back(decl);
            }
        }
        //Insert host preamble at top of main file
        HostPreamble = HostIncludes + "\n" + HostDecls + "\n" + HostGlobalVars + "\n" + HostKernels + "\n" + HostFunctions;
	generateReplacement(HostReplace, SM, SM->getLocForStartOfFile(MainFileID), 0, HostPreamble);
        //Insert device preamble at top of main kernel file
        DevPreamble = DevFunctions;
	generateReplacement(KernReplace, SM, SM->getLocForStartOfFile(MainFileID), 0, DevPreamble);

	//Generate Local init for this TU
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string file = idCharFilter(filename((*i).first));
	    //Paul: This search is linear w.r.t the number of global functions defined in cu2cl_util.h
	    // essentially 2x the number of CUDA files with kernel code + a constant number of utility functions
	    // it's quick and dirty but has the nice property of not reordering the data strucutre
	    std::string decl = "void __cu2cl_Init_" + file + "();\n";
	    std::vector<std::string>::iterator j = GlobalHDecls.begin(), f = GlobalHDecls.end();
	    //Iterate to the end of the vector or til a matching string is found
	    for (; j != f && (*j) != decl; j++);
	
	    if (j == f) { // Not found, add declaration and call
		GlobalHDecls.push_back("void __cu2cl_Init_" + file + "();\n");
		CU2CLInit += "    __cu2cl_Init_" + file + "();\n";
	    }
	    CLInit = "void __cu2cl_Init_" + file + "() {\n";
            std::list<llvm::StringRef> &l = (*i).second;
	//Paul: Addition to generate ALTERA .aocx build from binary with an ifdef
	    CLInit += "    #ifdef WITH_ALTERA\n";
	    CLInit += "    progLen = __cu2cl_LoadProgramSource(\"" + kernelNameFilter(idCharFilter(filename((*i).first))) + "_cl.aocx\", &progSrc);\n";
	    CLInit += "    __cu2cl_Program_" + file + " = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);\n";
	    CLInit += "    #else\n";
            CLInit += "    progLen = __cu2cl_LoadProgramSource(\"" + kernelNameFilter(filename((*i).first).str()) + "-cl.cl\", &progSrc);\n";
            CLInit += "    __cu2cl_Program_" + file + " = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);\n";
	    CLInit += "    #endif\n";
            CLInit += "    free((void *) progSrc);\n";
            CLInit += "    clBuildProgram(__cu2cl_Program_" + file + ", 1, &__cu2cl_Device, \"-I . ";
		CLInit += ExtraBuildArgs;
		CLInit += "\", NULL, NULL);\n";
	    // and initialize all its kernels
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLInit += "    __cu2cl_Kernel_" + kernelName + " = clCreateKernel(__cu2cl_Program_" + file + ", \"" + kernelName + "\", NULL);\n";
            }
	    CLInit += "}\n\n";
	    //Add the initializer to a deferred list of boilerplate
	    // to be inserted after relevant cl_program/cl_kernel declarations
            LocalBoilDefs[(*i).first].push_back(CLInit);
	}
        

        //Insert cleanup code at bottom of main
	//For each loaded cl_program
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            std::string file = idCharFilter(filename((*i).first));
	    //Paul: This search is O(n^2) (where n is 2x the number of CUDA files with kernel code
	    // it's quick and dirty but has the nice property of not reordering the data strucutre
	    std::string decl = "void __cu2cl_Cleanup_" + file + "();\n";
	    std::vector<std::string>::iterator j = GlobalHDecls.begin(), f = GlobalHDecls.end();
	    //Iterate to the end of the vector or til a matching string is found
	    for (; j != f && (*j) != decl; j++);
	
	    if (j == f) { // Not found, add declaration and call
		GlobalHDecls.push_back(decl);
		CU2CLClean = "    __cu2cl_Cleanup_" + file + "();\n" + CU2CLClean;
	    }
	    CLClean = "void __cu2cl_Cleanup_" + file + "() {\n";
	    //Release its kernels
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLClean += "    clReleaseKernel(__cu2cl_Kernel_" + kernelName + ");\n";
            }
	    //Then release the program itself
            CLClean += "    clReleaseProgram(__cu2cl_Program_" + file + ");\n";
	    CLClean += "}\n";
	    //Add the cleanup to a deferred list of boilerplate
	    // to be inserted after relevant cl_program/cl_kernel declarations
            LocalBoilDefs[(*i).first].push_back(CLClean);
        }
	//TODO: Remove this? Unless there's some reason to iterate to the end?
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
        }
        
        //Insert boilerplate at the top of file as a comment, if it doesn't have a main method
	if (MainDecl == NULL || MainDecl->getBody() == NULL) {
	    std::stringstream boilStr;
	    boilStr << "No main() found\nCU2CL Boilerplate inserted here:\nCU2CL Initialization:\n" << "__cu2cl_Init();\n" << "\n\nCU2CL Cleanup:\n" << "__cu2cl_Cleanup();\n"; 
            emitCU2CLDiagnostic(SM, SM->getLocForStartOfFile(MainFileID), "CU2CL Unhandled", "No main() found!\n\tBoilerplate inserted as header comment!\n", boilStr.str(), &HostReplace);
        }
	//Otherwise, insert it the start and end of the main method
	else {
	    CompoundStmt *mainBody = dyn_cast<CompoundStmt>(MainDecl->getBody());
	    generateReplacement(HostReplace, SM, PP->getLocForEndOfToken(mainBody->getLBracLoc(), 0), 0, "\n__cu2cl_Init();\n");
	    generateReplacement(HostReplace, SM, mainBody->getRBracLoc(), 0, "__cu2cl_Cleanup();\n");
        }

        //Rewrite cl_mems in DeclGroups
        for (std::set<DeclGroupRef>::iterator i = DeviceMemDGs.begin(),
             e = DeviceMemDGs.end(); i != e; i++) {
            DeclGroupRef DG = *i;
            SourceLocation start, end;
            std::string replace;
            for (DeclGroupRef::iterator iDG = DG.begin(), eDG = DG.end(); iDG != eDG; ++iDG) {
                VarDecl *vd = (VarDecl *) (*iDG);
                if (iDG == DG.begin()) {
                    start = (*iDG)->getLocStart();
                }
                if (DeviceMemVars.find(vd) != DeviceMemVars.end()) {
		DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(vd)), ST));
                }
                else {
			//If it's not a device variable print it (with type to de-group it)
                    replace += PrintDeclToString(vd);
                }
                if ((iDG + 1) == DG.end()) {
			//We've reached the end of the replace range, record it
                    end = (*iDG)->getLocEnd();
                }
                else {
			//replaceVarDecl handles semicolons and newlines for device variables, so only make changes to host variables
                    if (DeviceMemVars.find(vd) == DeviceMemVars.end()) replace += ";\n";
                }
            }
	    //If the host pointer has been previously Replaced with a vector rewrite, delete it
	    HostVecVars.erase(start);
	    //Before pushing this replacement, check HostReplace for previous vector type rewrites on the variable
	    generateReplacement(HostReplace, SM, start, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(SM->getExpansionLoc(start), SM->getExpansionLoc(end)))), replace);
        }
	//Flush all remaining vector rewrites still in the map to a global map, for pruning after cl_mem propagation
	GlobalHostVecVars.insert(HostVecVars.begin(), HostVecVars.end());
	//Write all buffered comments to output streams
	writeComments(SM);
	//And clean up the list's sentinel
	free(head);
	head = NULL;
	tail = NULL;
	
	//Do final cleanup of the Replacement vectors
	std::vector<Range> conflicts;
	//Get rid of duplicate replacements (e.g. multiple "Cannot translate template" comments
	deduplicate(HostReplace, conflicts);
	//Collapse Replacements on the same SourceLocation (for things like InsertBefore + Replace)
	coalesceReplacements(HostReplace);
	//Share the finished replacements with the global data structure
	GlobalHostReplace.insert(GlobalHostReplace.end(), HostReplace.begin(), HostReplace.end());

	//Do the same steps on kernel code	
	deduplicate(KernReplace, conflicts);
	coalesceReplacements(KernReplace);
	GlobalKernReplace.insert(GlobalKernReplace.end(), KernReplace.begin(), KernReplace.end());

	#ifdef CU2CL_ENABLE_TIMING
	    TransTime += get_time();
	    llvm::errs() << SM->getFileEntryForID(MainFileID)->getName() << " Translation Time: " << TransTime << " microseconds\n";
	#endif
    }

    void RewriteInclude(SourceLocation HashLoc, const Token &IncludeTok,
                        llvm::StringRef FileName, bool IsAngled,
                        const FileEntry *File, SourceLocation EndLoc) {
        llvm::StringRef fileExt = extension(SM->getPresumedLoc(HashLoc).getFilename());
        llvm::StringRef includedFile = filename(FileName);
        llvm::StringRef includedExt = extension(includedFile);
            //If system-wide include style (#include <foo.h>) is used, don't translate
            if (IsAngled) {
	        if (!isInBannedInclude(HashLoc, SM, LO)) generateReplacement(KernReplace, SM, HashLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(HashLoc, EndLoc))), "");
		//Remove reference to the CUDA header
                if (includedFile.equals("cuda.h") || includedFile.equals("cuda_runtime.h"))
	            generateReplacement(HostReplace, SM, HashLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(HashLoc, EndLoc))), "");
            }
	    //Remove quote-included reference to the CUDA header
            else if (includedFile.equals("cuda.h") || includedFile.equals("cuda_runtime.h")) {
	        generateReplacement(HostReplace, SM, HashLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(HashLoc, EndLoc))), "");
	        generateReplacement(KernReplace, SM, HashLoc, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(HashLoc, EndLoc))), "");
            }
	    else if (!isInBannedInclude(HashLoc, SM, LO)) {
	    //If local include style (#include "foo.h") is used, do translate
                FileID fileID = SM->getFileID(HashLoc);
                SourceLocation fileStartLoc = SM->getLocForStartOfFile(fileID);
                llvm::StringRef fileBuf = SM->getBufferData(fileID);
                const char *fileBufStart = fileBuf.begin();
                SourceLocation start = fileStartLoc.getLocWithOffset(includedFile.begin() - fileBufStart);
                SourceLocation end = fileStartLoc.getLocWithOffset((includedExt.end()) - fileBufStart);
		//replace filename and type
		std::string hostname = kernelNameFilter(includedFile.str()) + "-cl.h";		
		std::string kernname = kernelNameFilter(includedFile.str()) + "-cl.cl";
		//FIXME: I am not sure why it's calculating one character larger than it should be, but regression tests indicate the static -1 is working
		// We should figure out the root of the problem to guarantee the fix
	        generateReplacement(HostReplace, SM, start, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(start, end)))-1, hostname);
	        generateReplacement(KernReplace, SM, start, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(start, end)))-1, kernname);
            }
    }

};

class RewriteCUDAAction : public SyntaxOnlyAction {
protected:

    //The factory method needeed to initialize the plugin as an ASTconsumer
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) {
        
        std::string filename = InFile.str();
	std::string origFilename = filename;
        size_t dotPos = filename.rfind('.');
	filename = kernelNameFilter(filename) + "-cl" + filename.substr(dotPos);
		    std::string error, HostOutputPathName, HostTempPathName, KernOutputPathName, KernTempPathName;
		    llvm::raw_ostream *hostOS = CI.createOutputFile(StringRef(CI.getFrontendOpts().OutputFile), error, false, true, filename, "cpp", true, true, &HostOutputPathName, &HostTempPathName);
		    llvm::raw_ostream *kernelOS = CI.createOutputFile(StringRef(CI.getFrontendOpts().OutputFile), error, false, true, filename, "cl", true, true, &KernOutputPathName, &KernTempPathName);
			OutputFile *HostOF = new OutputFile(HostOutputPathName, HostTempPathName, hostOS);
			OutputFile *KernOF = new OutputFile(KernOutputPathName, KernTempPathName, kernelOS);
                    if (hostOS && kernelOS) 
            return new RewriteCUDA(&CI, origFilename, HostOF, KernOF);
        //TODO cleanup files?	
        return NULL;
    }


};

RewriteIncludesCallback::RewriteIncludesCallback(RewriteCUDA *RC) :
    RCUDA(RC) {
}

void RewriteIncludesCallback::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                                                 llvm::StringRef FileName, bool IsAngled,
						 CharSourceRange FileNameRange, const FileEntry *File,
                                                 StringRef SearchPath, StringRef RelativePath,
						 const Module * Imported) {
	if (Imported != NULL) llvm::errs() << "CU2CL DEBUG -- Import directive detected, translation not supported!";
    RCUDA->RewriteInclude(HashLoc, IncludeTok, FileName, IsAngled, File, FileNameRange.getEnd());
}

}


//FIXME: InsertArgumentAdjuster isn't available in Clang 3.4
//This class substitutes the necessary appending of default compilation arguments
// until we can use InsertArgumentAdjuster
//All credit to Clang-check's InsertAdjuster for inspiring this minimal function
class AppendAdjuster: public clang::tooling::ArgumentsAdjuster {
public:

    //For some reason things don't get tokenized by the tool invocation if we don't
    // manually tokenize ourselves. This constructor handles that
    //It ends up forking off the first "-D" as its own token, and leaving the rest as a second
    AppendAdjuster(const char *Add) : AddV() {
	std::stringstream ss(Add);
	std::string item;
	//Tokenize based on a space character delimiter
	while(std::getline(ss, item, ' ')) {
	    AddV.push_back(item);
	}
    }
    virtual CommandLineArguments Adjust(const CommandLineArguments &Args) LLVM_OVERRIDE {
	CommandLineArguments Ret(Args);
	CommandLineArguments::iterator it = Ret.end();

	Ret.insert(it, AddV.begin(), AddV.end());
	return Ret;
    }

private:
    CommandLineArguments AddV;
};


//Add custom cu2cl arguments
llvm::cl::opt<bool, true> Comments("inline-comments", llvm::cl::desc("Add inline descriptive comments to output (boolean, default \"true\")."),  llvm::cl::location(AddInlineComments));
llvm::cl::opt<std::string, true> ExtraArgs("cl-extra-args", llvm::cl::desc("Additional compiler arguments to append to all generated clBuildProgram calls."), llvm::cl::value_desc("<\"args\">"), llvm::cl::location(ExtraBuildArgs), llvm::cl::init(""));
llvm::cl::opt<bool, true> KernelRename("rename-kernel-files", llvm::cl::desc("Replace instances of \"kernel\" in filenames with \"knl\""), llvm::cl::location(FilterKernelName));
llvm::cl::opt<bool, true> ImportGCCPaths("import-gcc-paths", llvm::cl::desc("Use GCC to infer search path(s) for system include directories"), llvm::cl::location(UseGCCPaths));

std::string parseGCCPaths() {
    //create a temporary file
	llvm::SmallVectorImpl<char> * tmpPath = new llvm::SmallVector<char, 128>();
	int tempFD;
	//Even though we don't use the FD, we use this variant to force the file to persist long enough for GCC to use it
	llvm::sys::fs::createTemporaryFile("cu2cl-gcc-dummy", "c", tempFD, *tmpPath);
    //generate a comand
	llvm::SmallString<128> * tmpPathStr = new llvm::SmallString<128>();
	tmpPathStr->assign("gcc -v ");
	tmpPathStr->append(*tmpPath);
	tmpPathStr->append(" 2>&1");
	const char * cmd = tmpPathStr->c_str();
//	llvm::errs() << "Generated GCC Search command: " << cmd << "\n";
    //run GCC and buffer all the data
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL)
            result += buffer;
    }
//	llvm::errs() << "GCC Search Output:\n" << result << "\n";

	//Generate a StringRef for simpler search ops
	StringRef parseStr(result);

	//Find the start demarcator
	size_t start = parseStr.find("#include <...> search starts here:\n");
	//Skip ahead to the first character after the demarcator line
	start = 1 + parseStr.find("\n", start);
	//Find the end demarcator
	size_t end = parseStr.find("End of search list.\n", start);
	//Slice out just the chunk of paths, separated by newlines
	StringRef pathsRef = parseStr.slice(start, end);
	//For each line, add a new -I directive
    std::string directives = " ";
	start = 0, end = pathsRef.find("\n");
	for (; start < pathsRef.size(); start = end +1, end = pathsRef.find("\n", start)) directives += "-I" + pathsRef.slice(start, end).rtrim().str() + " ";
	llvm::errs() << "GCC final directives:" << directives << "\n";
    //use llvm regex to find the important lines
    //and add -I directives

    return directives;
}

void replaceVarDecl(DeclaratorDecl *decl, SourceTuple * ST) {
	//IIRC If the dyn_cast of the VarDecl doesn't work it'll show up as NULL
	//llvm::errs() << "CU2CL DEBUG: Replacing var decl " << (void*)decl << "\n";
	SourceManager * SM = std::get<0>(*ST);
	Preprocessor * PP = std::get<1>(*ST);
	LangOptions * LO = std::get<2>(*ST);
	if (decl == NULL) return;
	std::string replace = "";
        SourceLocation start, end, tempLoc;
		start = decl->getLocStart();
        	if ((decl->getAttr<CUDAConstantAttr>()) || (decl->getAttr<CUDADeviceAttr>() )) {
			start = decl->getTypeSpecStartLoc();
		}
		end = decl->getLocEnd();
			//Make sure we have the correct amount of "pointer to" on the output type
			std::string pointers = " ";
			for (Type * type = (Type *)decl->getType().getTypePtrOrNull(); type != NULL && type->isPointerType(); ) {
				Type * interior =  (Type *) type->getPointeeType().getTypePtrOrNull();
				if (interior->isPointerType()) {
					pointers = pointers + "*";
				}
				type = interior;
			}
			if (pointers == " ") pointers = "";
			
                    replace += "cl_mem" + pointers + " " + decl->getNameAsString();
			if (VarDecl *var = dyn_cast<VarDecl>(decl)) {
		    if (var->getType()->isArrayType()) {
			//make sure to grab the array [...] Expr too
            ArrayTypeLoc arrTL = var->getTypeSourceInfo()->getTypeLoc().getAs<ArrayTypeLoc>();
		while (!arrTL.isNull()) {
			replace += "[";
			if (arrTL.getSizeExpr() != NULL) replace += getStmtText(LO, SM, arrTL.getSizeExpr());
			arrTL = arrTL.getElementLoc().getAs<ArrayTypeLoc>();
			replace += "]";
		}
		    }
			//If it's a parameter, we want to keep the comma, not replace it with a semiclon
			if (dyn_cast<ParmVarDecl>(decl) == NULL) replace += ";";
	    		if ((tempLoc = Lexer::findLocationAfterToken(end, tok::semi, *SM, *LO, false)).isValid()) {
				//found a semicolon, replace the endLoc with the semicolon's loc
			end = tempLoc;
	    		} else {
				//we only want to insert a newline for DeclGroups, which will necessarily not have a semicolon on any but the last member
				if (dyn_cast<ParmVarDecl>(decl) == NULL) replace += "\n";
			}
			
	        	generateReplacement(GlobalHostReplace, SM, start, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(SM->getExpansionLoc(start), SM->getExpansionLoc(end)))), replace);
			} else if (FieldDecl * fdecl = dyn_cast<FieldDecl>(decl)) {
		    if (fdecl->getType()->isArrayType()) {
			//make sure to grab the array [...] Expr too
            ArrayTypeLoc arrTL = fdecl->getTypeSourceInfo()->getTypeLoc().getAs<ArrayTypeLoc>();
		while (!arrTL.isNull()) {
			replace += "[";
			if (arrTL.getSizeExpr() != NULL) replace += getStmtText(LO, SM, arrTL.getSizeExpr());
			arrTL = arrTL.getElementLoc().getAs<ArrayTypeLoc>();
			replace += "]";
		}
		    }
	        		generateReplacement(GlobalHostReplace, SM, start, getRangeSize(*SM, CharSourceRange::getTokenRange(SourceRange(SM->getExpansionLoc(start), SM->getExpansionLoc(end)))), replace);
				

			} else {
				TypeLoc tl = decl->getTypeSourceInfo()->getTypeLoc();
				SourceRange realRange(tl.getBeginLoc(), Lexer::getLocForEndOfToken(tl.getBeginLoc(), 0, *SM, *LO));
	        		generateReplacement(GlobalHostReplace, SM, tl.getBeginLoc(), getRangeSize(*SM, CharSourceRange::getTokenRange(tl.getLocalSourceRange())), "cl_mem");
    			}

}

//return true iff child is a descendant of ancestor (or child == ancestor)
bool isAncestor(Stmt * ancestor, Stmt * child) {
	if (ancestor == child) return true;
	//else if (ancestor->child_begin() != ancestor->child_end()) {
		for (Stmt::child_iterator citr = ancestor->child_begin(); citr != ancestor->child_end(); citr++){
			if (isAncestor(*citr, child)) return true;
		}
		return false;
		
	//}
}

int main(int argc, const char ** argv) {
	
	//Before we do anything, parse off common arguments, a la MPI
	CommonOptionsParser options(argc, argv);

	//create a ClangTool instance
	RefactoringTool cu2cl(options.getCompilations(), options.getSourcePathList());

	//Inject extra default arguments
	//These are needed to override parsing of some CUDA headers Clang doesn't like
	// and putting them here removes the need to put them on every call or in a static compilation database
	//std::string embeddedArgs = "-disable-free -D CUDA_SAFE_CALL(X)=X -D __CUDACC__ -D __SM_32_INTRINSICS_H__ -D __SM_35_INTRINSICS_H__ -D __SURFACE_INDIRECT_FUNCTIONS_H__";
	std::string embeddedArgs = "-disable-free -D CUDA_SAFE_CALL(X)=X -D __CUDACC__ -D __SM_32_INTRINSICS_H__ -D __SM_35_INTRINSICS_H__ -D __SURFACE_INDIRECT_FUNCTIONS_H__ -include cuda_runtime.h";
      
	//TODO: Add a verbose diagnostic function specifying the status of all options 
	if (AddInlineComments) llvm::errs() << "Commenting is enabled\n";
	else llvm::errs() << "Commenting is disabled\n";
	llvm::errs() << "clBuild arguments appended: " << ExtraBuildArgs << "\n";
	if (FilterKernelName) llvm::errs() << "Name filtering is enabled\n";
	else llvm::errs() << "Name filtering is disabled\n";

	if (UseGCCPaths) {
	    llvm::errs() << "GCC include directory import is enabled\n";
	    //logic to spawn a "gcc -v foo.c" proc and parse search path(s)
	    embeddedArgs += parseGCCPaths();
	} else llvm::errs() << "GCC include directory import is disabled\n";
	cu2cl.appendArgumentsAdjuster(new AppendAdjuster(embeddedArgs.c_str()));

	//Boilerplate generation has to start before the tool runs, so the tool
	// instances can contribute their local init calls to it
	//Construct OpenCL initialization boilerplate
        CU2CLInit += "void __cu2cl_Init() {\n";
	GlobalCDecls["cu2cl_util.c"].push_back("const char *progSrc;\n");
	GlobalCDecls["cu2cl_util.c"].push_back("size_t progLen;\n\n");
        //Rather than obviating these lines to support cudaSetDevice, we'll assume these lines
        // are *always* included, and IFF cudaSetDevice is used, include code to instead scan
        // *all* devices, and allow for reinitialization
        CU2CLInit += "    clGetPlatformIDs(1, &__cu2cl_Platform, NULL);\n";
        CU2CLInit += "    clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_ALL, 1, &__cu2cl_Device, NULL);\n";
        CU2CLInit += "    __cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);\n";
        CU2CLInit += "    __cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);\n";

	//Construct OpenCL cleanup boilerplate (bottom first, decl after tool contributes prog/kernl cleanup calls)
	//BOIL: global cleanup
        CU2CLClean += "    clReleaseCommandQueue(__cu2cl_CommandQueue);\n";
        CU2CLClean += "    clReleaseContext(__cu2cl_Context);\n";
	CU2CLClean += "}\n";

	//run the tool (for now, just use the PluginASTAction from original CU2CL
	int result = cu2cl.run(newFrontendActionFactory<RewriteCUDAAction>());

	//After the toos runs, don't forget to re-initialize the comment buffer, in case we need to emit any diagnostics
	head = (struct commentBufferNode *)malloc(sizeof(struct commentBufferNode));
	head->n = NULL;
    	tail = head;
	

	//After the tools run, we can finalize the global boilerplate
	//If __cu2cl_setDevice is used, we need to initialize the scan variables
	if(UsesCUDASetDevice) {    
		CU2CLInit += "    __cu2cl_AllDevices_size = 0;\n";
		CU2CLInit += "    __cu2cl_AllDevices_curr_idx = 0;\n";
		CU2CLInit += "    __cu2cl_AllDevices = NULL;\n";
	}
	//If we need to make use of any custom kernels generated in cu2cl_util.cl
	if (UsesCU2CLUtilCL) {
	    //Declare and build the __cu2cl_Util_Program 
            GlobalCDecls["cu2cl_util.c"].push_back("cl_program __cu2cl_Util_Program;\n");
	    CU2CLInit += "    #ifdef WITH_ALTERA\n";
	    CU2CLInit += "    progLen = __cu2cl_LoadProgramSource(\"cu2cl_util.aocx\", &progSrc);\n";
	    CU2CLInit += "    __cu2cl_Util_Program = clCreateProgramWithBinary(__cu2cl_Context, 1, &__cu2cl_Device, &progLen, (const unsigned char **)&progSrc, NULL, NULL);\n";
	    CU2CLInit += "    #else\n";
            CU2CLInit += "    progLen = __cu2cl_LoadProgramSource(\"cu2cl_util.cl\", &progSrc);\n";
            CU2CLInit += "    __cu2cl_Util_Program = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);\n";
	    CU2CLInit += "    #endif\n";
            CU2CLInit += "    free((void *) progSrc);\n";
            CU2CLInit += "    clBuildProgram(__cu2cl_Util_Program, 1, &__cu2cl_Device, \"-I . ";
		CU2CLInit += ExtraBuildArgs;
		CU2CLInit += "\", NULL, NULL);\n";
	    // and initialize all its kernels
            for (std::vector<std::string>::iterator i = UtilKernels.begin(), e = UtilKernels.end();
                 i != e; i++) {
                CU2CLInit += "    __cu2cl_Kernel_" + (*i) + " = clCreateKernel(__cu2cl_Util_Program, \"" + (*i) + "\", NULL);\n";
            }

	    //Cleanup the kernels and associated program
            CU2CLClean = "    clReleaseProgram(__cu2cl_Util_Program);\n" + CU2CLClean;
            for (std::vector<std::string>::iterator i = UtilKernels.begin(), e = UtilKernels.end();
                 i != e; i++) {
                CU2CLClean = "    clReleaseKernel(__cu2cl_Kernel_" + (*i) + ");\n" + CU2CLClean;
            }
	} 
	CU2CLInit += "}\n";
	CU2CLClean = "void __cu2cl_Cleanup() {\n" + CU2CLClean;

	//Construct a SourceManager for the rewriters the replacements will be applied to
	// We use a stripped-down version of the way clang-apply-replacements sets up their SourceManager
	IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
	DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), DiagOpts.getPtr());
	FileManager Files((FileSystemOptions()));
	SourceManager RewriteSM(Diagnostics, Files);
	//Set up our two rewriters
	LangOptions LOpts = LangOptions();
	Rewriter GlobalHostRewrite(RewriteSM, LOpts);
	Rewriter GlobalKernRewrite(RewriteSM, LOpts);
	//Generate cu2cl_util.c/h
	std::string error;
	raw_ostream * cu2cl_util = new llvm::raw_fd_ostream("cu2cl_util.c", error);
	raw_ostream * cu2cl_header = new llvm::raw_fd_ostream("cu2cl_util.h", error);
	raw_ostream * cu2cl_kernel = new llvm::raw_fd_ostream("cu2cl_util.cl", error);
	
	//Add licensing info to all generated files
	*cu2cl_header << CU2CL_LICENSE;
	*cu2cl_util << CU2CL_LICENSE;
	*cu2cl_kernel << CU2CL_LICENSE;

	//Force cu2cl_util.c to include cu2cl_util.h and it, the other key headers
	*cu2cl_header << "#ifdef __APPLE__\n";
	*cu2cl_header << "#include <OpenCL/opencl.h>\n";
	*cu2cl_header << "#else\n";
	*cu2cl_header << "#include <CL/opencl.h>\n";
	*cu2cl_header << "#endif\n";
	*cu2cl_header << "#include <stdlib.h>\n";
	*cu2cl_header << "#include <stdio.h>\n";
	*cu2cl_header << "\n#ifdef __cplusplus\n";
	*cu2cl_header << "extern \"C\" {\n";
	*cu2cl_header << "#endif\n";
	*cu2cl_header << "void __cu2cl_Init();\n";
	*cu2cl_header << "\nvoid __cu2cl_Cleanup();\n";
	*cu2cl_util << "#include \"cu2cl_util.h\"\n";

	//After all Source files have been processed, they will have generated all global
	// information necessary to finalize declarations
	//Assemble the last remaining declarations
	GlobalCDecls["cu2cl_util.c"].push_back("cl_platform_id __cu2cl_Platform;\n");
        GlobalCDecls["cu2cl_util.c"].push_back("cl_device_id __cu2cl_Device;\n");
        GlobalCDecls["cu2cl_util.c"].push_back("cl_context __cu2cl_Context;\n");
        GlobalCDecls["cu2cl_util.c"].push_back("cl_command_queue __cu2cl_CommandQueue;\n\n");
        GlobalCDecls["cu2cl_util.c"].push_back("size_t globalWorkSize[3];\n");
        GlobalCDecls["cu2cl_util.c"].push_back("size_t localWorkSize[3];\n");
	
	//Then iterate over all the pieces and generate the necessary replacements (necessarily O(m*n^2)
	// where (m is the number of decls in each vector of strings, and n is the number of source files)
	// the i loop makes sure we generate output for each primary file
	//  the j loop makes sure we check decls for every file
	//   the k loop concatenates the vector of strings with the decls in it for each file and applies
	//   "extern " if they are not for the file the i loop is currently generating output for
	for (FileStrCacheMap::iterator i = GlobalCDecls.begin(), e = GlobalCDecls.end(); i != e; i++) {
	    for (FileStrCacheMap:: iterator j = GlobalCDecls.begin(), f = GlobalCDecls.end(); j != f; j++) {
		std::string rep_str = "";
		if (i == j) {
		    //generate non-extern decls
		    for (std::vector<std::string>::iterator k = (*j).second.begin(), g = (*j).second.end(); k != g; k++) {
			//glue decl strings together, assuming they're already \n terminated
			rep_str += (*k);
		    }
		} else {
		    //generate extern decls
		    for (std::vector<std::string>::iterator k = (*j).second.begin(), g = (*j).second.end(); k != g; k++) {
			//glue decl strings together, assuming they're already \n terminated
			rep_str += "extern " + (*k);
		    }
		}
		if ((*i).first == "cu2cl_util.c") {
		    //just add it to the string that will be pushed to the ostream later;
		    *cu2cl_util << rep_str;
		} else {
		    //Get a fid for the file, if it doesn't exist in the SM, force it
		    const FileEntry * FE = Files.getFile((*i).first);
            	    FileID fid = RewriteSM.translateFile(FE);
		    if (fid.isInvalid()) fid = RewriteSM.createFileID(FE, SourceLocation(), SrcMgr::C_User);
		    //Get a SourceLocation for the start of the file
		    SourceLocation Loc = RewriteSM.getLocForStartOfFile(fid);
		    //generate one big Replacement for all the decls
		    //and add it to the GlobalHostReplace
	            generateReplacement(GlobalHostReplace, &RewriteSM, Loc, 0, rep_str);
		}
	    }
	}

	//Process all deferred cl_mem translations iteratively
	//size_t refcount = 0;
	//llvm::errs() << "Indexed " << AllDeclRefsByDecl.size() << " Decls,\n";
	//for (DeclToRefMap::iterator drmt = AllDeclRefsByDecl.begin(); drmt != AllDeclRefsByDecl.end(); drmt++) {
	//	llvm::errs() << "CU2CL DEBUG: " << (*drmt).second.size() << " references to Decl at " << (*drmt).first << "\n";
	//	refcount+=(*drmt).second.size();
	//}
	//llvm::errs() << "\t bearing " << refcount << " total DeclRefExprs\n";

	//for (CanonicalFuncDeclMap::iterator fdit = AllFuncsByCanon.begin(); fdit != AllFuncsByCanon.end(); fdit++) {
	//    llvm::errs() << "CU2CL DEBUG: FuncDecl at : " << (*fdit).first << " has " << (*fdit).second.size() << " redeclarations!\n";
	//    for (std::vector<std::pair<FunctionDecl *, SourceTuple *> >::iterator fitr = (*fdit).second.begin(); fitr != (*fdit).second.end(); fitr++) {
	//	llvm::errs() << "CU2CL DEBUG: At: " << (*fitr).first->getLocStart().printToString(*(std::get<0>(*(*fitr).second))) << "\n";
	//    }
		
	//}
	//for (FlaggedDeclVec::iterator ditr = DeclsToTranslate.begin(); ditr != DeclsToTranslate.end(); ditr++) {
	//Since we are potentially adding things to the back, it is not safe to use an iterator pattern as they are likely invalidated, use count and element access instead
	int ditr = 0, dend = DeclsToTranslate.size();
	for (; ditr != dend; ditr++) {
		//Call the translation function
		NamedDecl *decl = (DeclsToTranslate[ditr]).first;
		SourceTuple * ST = (DeclsToTranslate[ditr]).second;
		replaceVarDecl(dyn_cast<DeclaratorDecl>(decl), ST);
	    	GlobalHostVecVars.erase(decl->getLocStart());

		//DOWNWARD PROPAGATION
		//First, create an iterator for all the DeclRefExprs involving this Decl
		ASTContext * AST = std::get<3>(*ST);
		std::string declLoc = decl->getLocStart().printToString(*(std::get<0>(*ST)));
		std::vector<DeclRefExpr *> refs = AllDeclRefsByDecl[declLoc];
		for (std::vector<DeclRefExpr *>::iterator ref = refs.begin(); ref != refs.end(); ref++) {
			//Add an early abort if the DeclRefExpr doesn't belong to the exact variable requiring translation
			//(this happens in DeclGroups, since AllDeclRefsByDecl contains all references to any member of the group.
			//If multiple members of the group need propagation, they will each get an iteration of this for loop and thus process their respective declrefExprs.)
			if (decl != (*ref)->getFoundDecl() && decl != (*ref)->getDecl()) {
				llvm::errs() << "CU2CL DEBUG: Rejected propagation of DeclRefExpr " << (*ref)->getDecl()->getName() << " not matching Decl " << decl->getName() << "\n";
				continue;

			}

			//For each, iterate upwards to the nearest Stmt ancestor
			//TODO This type changes to DynTypedNodeList in a future version of Clang
			ASTContext::ParentVector parents = AST->getParents(*(dyn_cast<Stmt>(*ref)));
			//The above should give us *all* the ancestors, look backwards until the next parent is a Stmt but not an Expr
			ASTContext::ParentVector::iterator pitr;
			Expr * ancestor;
			for (pitr = parents.begin(); pitr != parents.end(); parents = AST->getParents(*pitr), pitr = parents.begin()) {
				const Stmt * stmt = ((pitr)->get<Stmt>());
				if (stmt != NULL) {
					const Expr * expr = dyn_cast<Expr>(stmt) ;
					if (expr  == NULL) {
						//That means this ancestor is a statement, we should go no further
						break;
					} else if ( dyn_cast<CallExpr>(stmt)) {
						//If it's a CallExpr, we only want the innermost, so update the ancestor and abort
						ancestor = (Expr *) expr;
						break;
					} else {
						//We haven't hit a maximal ancestor, update and keep iterating
						ancestor = (Expr *) expr;
					}
				}	
			}
			
			if (CallExpr * call = dyn_cast<CallExpr>(ancestor)) {
				//If it's a parameter to a CUDAKernelCallExpr, we obviously don't want to translate it, as it'll already be removed from the host
				if (dyn_cast<CUDAKernelCallExpr>(call)) {
					continue;
				}
			//TODO: Abort if for some reason it's a function we shouldn't translate - i.e. if it's a CUDA runtime function, we need to detect which one, and perform the appropriate translation as if it were in the per-AST portion...
				//Ensure the parameter of the function we are translating is added to the list IFF it isn't already in there
				//Figure out which parameter of the function we are supplying the reference to
				//Iterate over all arguments
				unsigned int argNum = 0;
				for (CallExpr::arg_iterator aitr = call->arg_begin(); aitr != call->arg_end(); aitr = aitr+1) {
					//when we find the one that has the DeclRefExpr we're working on, record its position and abort the loop
					if (isAncestor(*aitr, *ref)) break;
					argNum++;
				}
				
				//Check if that parameter is already marked for translation
				//get the FuncDecl (and definition, if it exists)
				FunctionDecl * func = call->getDirectCallee();
				if (func) {
					
					if (func->getNameAsString().find("cu") == 0) {
						llvm::errs() << "CU2CL DEBUG: Skipping propagated CallExpr translation on CUDA function: " << func->getNameAsString() << "!\n";
					} else {
						ParmVarDecl * parm = func->getParamDecl(argNum);
						if (parm) {
							//If it's not already marked for translation
							
							if (!hasFlaggedDecl(&DeclsToTranslate, parm)) {
								//Add it to the list
								DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(parm)), ST));
							} else {
								//TODO Any logic for already-flagged variables?
							}
						}

					}
				//If for some reason we can't get the FuncDecl directly, just dump some debug notes
				} else {
					llvm::errs() << "CU2CL DEBUG: No direct callee to CallExpr marked for propagation: " << getStmtText(std::get<2>(*ST), std::get<0>(*ST), call) << "\n";
//					(call->getCallee())->dump(llvm::errs(), *(std::get<0>(*ST)));
	//				llvm::errs() << "\n";
				}
			} else {
				llvm::errs() << "CU2CL DEBUG: Unpropagated cl_mem AST ancestor for reference: "<< getStmtText(std::get<2>(*ST), std::get<0>(*ST), *ref) << "\n";
	//			ancestor->dump(llvm::errs(), *(std::get<0>(*ST)));
	//			llvm::errs() << "\n";
			}
		}
		

		//HORIZONTAL AND UPWARDS PROPAGATION
		//IFF the Decl is a ParmVarDecl, we need to check all calls of the function it is a parameter to
		if (ParmVarDecl * parm =dyn_cast<ParmVarDecl>(decl)) {
			ASTContext::ParentVector parents = AST->getParents(*(parm));
			//The above should give us *all* the ancestors, look backwards until the next parent is a Stmt but not an Expr
			ASTContext::ParentVector::iterator pitr;
			FunctionDecl * func;
			for (pitr = parents.begin(); pitr != parents.end(); parents = AST->getParents(*pitr), pitr = parents.begin()) {
				const Decl * decl = ((pitr)->get<Decl>());
				if (decl != NULL) {
					const FunctionDecl * ancestor = dyn_cast<FunctionDecl>(decl) ;
					if (func  == NULL) {
						//That means this ancestor is a statement, we should go no further
						break;
					} else if ( dyn_cast<FunctionDecl>(decl)) {
						//If it's a CallExpr, we only want the innermost, so update the ancestor and abort
						func = (FunctionDecl *) ancestor;
						break;
					} else {
						//We haven't hit a maximal ancestor, update and keep iterating
						func = (FunctionDecl *) ancestor;
					}
				}	
			}
				//Figure out which parameter of the function we are supplying the reference to
				//Iterate over all arguments
				unsigned int argNum = 0;
				for (FunctionDecl::param_iterator paitr = func->param_begin(); paitr != func->param_end(); paitr = paitr+1) {
					//when we find the one that has the ParamVarDecl
					if (parm  == (*paitr)) break;
					argNum++;
				}
			//HORIZONTAL PROPAGATION
			//Make sure functions declared in other ASTs with the same prototype inherit the change	
					    std::vector<std::pair<FunctionDecl *, SourceTuple *> > funcVec = AllFuncsByCanon[func->getFirstDecl()->getLocStart().printToString(*(std::get<0>(*ST)))];
	    				    for (std::vector<std::pair<FunctionDecl *, SourceTuple *> >::iterator fitr = funcVec.begin(); fitr != funcVec.end(); fitr++) {
						func = (*fitr).first;
						// Replace the old parameter from the triggering function with this variant's parameter (when declared in multiple ASTs)
						parm = func->getParamDecl(argNum);
										if (!hasFlaggedDecl(&DeclsToTranslate, parm)) {
											//Add it to the list
											DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(parm)), (*fitr).second));
										} else {
	//TODO Any other logic for already-flagged variables
	}
						//Get the target parameter
						//Upwards propagate it
						std::string funcDeclLoc = func->getLocStart().printToString(*(std::get<0>(*((*fitr).second))));
						std::vector<DeclRefExpr *> funcRefs = AllDeclRefsByDecl[funcDeclLoc];
						for (std::vector<DeclRefExpr *>::iterator funcRef = funcRefs.begin(); funcRef != funcRefs.end(); funcRef++) {
						 //Loop over all references to that function declaration
							ASTContext::ParentVector fParents = AST->getParents(*(dyn_cast<Stmt>(*funcRef)));
							//The above should give us *all* the ancestors, look backwards until the next parent is a Stmt but not an Expr
							ASTContext::ParentVector::iterator fPitr;
							Stmt * fStmt;
							Expr *  ancestor;
							for (fPitr = fParents.begin(); fPitr != fParents.end(); fParents = AST->getParents(*fPitr), fPitr = fParents.begin()) {
								fStmt = (Stmt*) ((fPitr)->get<Stmt>());
								if (fStmt != NULL) {
									const Expr * fExpr = dyn_cast<Expr>(fStmt) ;
									if (fExpr  == NULL) {
									//That means this ancestor is a statement, we should go no further
										break;
									} else if ( dyn_cast<CallExpr>(fStmt)) {
									//If it's a CallExpr, we only want the innermost, so update the ancestor and abort
										ancestor = (Expr *) fExpr;
										break;
									} else {
									//We haven't hit a maximal ancestor, update and keep iterating
										ancestor = (Expr *) fExpr;
									}
								}
							}
							CallExpr * funcCall = dyn_cast<CallExpr>(fStmt);
							if (funcCall) {
								Expr* argExpr = funcCall->getArg(argNum)->IgnoreImplicit();
								if (argExpr) {
									DeclRefExpr* argRef = dyn_cast<DeclRefExpr>(argExpr);
									if (argRef) { // It's a single variable reference as the argument
										//Simply grab the Decl referred to and submit it
										NamedDecl * argDecl = argRef->getDecl();
										if (!hasFlaggedDecl(&DeclsToTranslate, argDecl)) {
											//Add it to the list
											DeclsToTranslate.push_back(std::pair<NamedDecl*, SourceTuple*>((dyn_cast<NamedDecl>(argDecl)), (*fitr).second));
										} else {
	//TODO Any other logic for already-flagged variables	
}
									} else {
									llvm::errs() << "Tried to upwards propagate a non-simple argument!\n";
									}
								} else {
									llvm::errs() << "CU2CL Debug: Invalid argExpr in upwards propagation!\n" ;
								}
							} else {
								llvm::errs() << "Error: Couldn't get CallExpr for upwards propagation of called DeclRefExpr!\n";
							}
						    }
						}

		}
		dend = DeclsToTranslate.size();

	}
	//After propagating all cl_mems, clear off any vector rewrites that overlap with them
	for (std::map<SourceLocation, Replacement>::const_iterator I = GlobalHostVecVars.begin(), E = GlobalHostVecVars.end(); I != E; I++) {
		GlobalHostReplace.push_back(I->second);
	}

	//Inject local boilerplate functions that have been staged from each TU
	// (this is done here rather than where they are generated to ensure the program/kernel variables are declared before the functions, but after the include statements)
	//Since we don't need to rewrite utility files, there is no reason to assemble a rep_str
	// instead we create Replacements directly (which has the benefit of deduplication)
	for (FileStrCacheMap::iterator i = LocalBoilDefs.begin(), e = LocalBoilDefs.end(); i != e; i++) {
	    //Get a fid for the file, if it doesn't exist in the SM, force it
	    const FileEntry * FE = Files.getFile((*i).first);
	    FileID fid = RewriteSM.translateFile(FE);
	    if (fid.isInvalid()) fid = RewriteSM.createFileID(FE, SourceLocation(), SrcMgr::C_User);
	    //Get a SourceLocation for the start of the file
	    SourceLocation Loc = RewriteSM.getLocForStartOfFile(fid);
	    for (std::vector<std::string>::iterator j = (*i).second.begin(), f = (*i).second.end(); j != f; j++) {
	        generateReplacement(GlobalHostReplace, &RewriteSM, Loc, 0, (*j));
	    }
	}
	
	//After all Decls are appropriately generated, add the utility functions
	//cu2cl_util.h
	for (std::vector<std::string>::iterator i = GlobalHDecls.begin(), e = GlobalHDecls.end(); i != e; i++) {
	    *cu2cl_header << (*i) + "\n";
	}
	//cu2cl_util.c
	for (std::vector<std::string>::iterator i = GlobalCFuncs.begin(), e = GlobalCFuncs.end(); i != e; i++) {
	    *cu2cl_util << (*i) + "\n";
	}
	//cu2cl_util.cl
	for (std::vector<std::string>::iterator i = GlobalCLFuncs.begin(), e = GlobalCLFuncs.end(); i != e; i++) {
	    *cu2cl_kernel << (*i) + "\n";
	}


	//After pushing all the utility functions out, add the global init/cleanup calls to cu2cl_util.c
	*cu2cl_util << CU2CLInit + "\n";
	*cu2cl_util << CU2CLClean;	

	//After all Source files have been processed, they will have accumulated their Replacments
	// into the global data structures, now deduplicate and fuse across them

	//Before dumping replacements, don't forget to flush the comment buffer
	writeComments(&RewriteSM);
	free(head);
	head = NULL;
	tail = NULL;
	std::vector<Range> conflicts;
	std::vector<Replacement> GlobalHostConflicts, GlobalKernConflicts;
	deduplicate(GlobalHostReplace, conflicts);
	coalesceReplacements(GlobalHostReplace);
	deduplicate(GlobalKernReplace, conflicts);
	coalesceReplacements(GlobalKernReplace);
	

	//Apply the global set of replacements to each of them
	//debugPrintReplacements(GlobalHostReplace);
	applyAllReplacements(GlobalHostReplace, GlobalHostRewrite);
	//debugPrintReplacements(GlobalKernReplace);
	applyAllReplacements(GlobalKernReplace, GlobalKernRewrite);


	//Flush all rewritten #included host files
        for (IDOutFileMap::iterator i = OutFiles.begin(), e = OutFiles.end();
             i != e; i++) {
		
		const FileEntry * FE = Files.getFile((*i).first);
            	FileID fid = RewriteSM.translateFile(FE);
            	OutputFile * outFile = (*i).second;
		if (fid.isInvalid()) {
		    llvm::errs() << "File [" << (*i).first << "] has invalid (zero) FID, attempting forced creation!\n\t(Likely cause is lack of rewrites in both host and kernel outputs.)\n";
		    fid = RewriteSM.createFileID(FE, SourceLocation(), SrcMgr::C_User);
		    if (fid.isInvalid()) {
			llvm::errs() << "\tError file [" << (*i).first << "] still has invalid (zero) FID, dropping output! (Temp files may persist.)\n";
			continue;
		    } else {
			llvm::errs() << "\tForced FID creation for file [" << (*i).first << "] succeeded, proceeding with output.\n";
		    }	
		}
	    //If changes were made, bring them in from the rewriter
            if (const RewriteBuffer *RewriteBuff =
                GlobalHostRewrite.getRewriteBufferFor(fid)) {
                *(outFile->OS) << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
	    //Otherwise just dump the file directly
            else {
                llvm::StringRef fileBuf = RewriteSM.getBufferData(fid);
                *(outFile->OS) << std::string(fileBuf.begin(), fileBuf.end());
		llvm::errs() << "No changes made to " << RewriteSM.getFileEntryForID(fid)->getName() << "\n";
            }
            outFile->OS->flush();
	    clearOutputFile(outFile, &Files);
        }

	//Flush rewritten #included kernel files
        for (IDOutFileMap::iterator i = KernelOutFiles.begin(), e = KernelOutFiles.end();
             i != e; i++) {
            FileID fid = RewriteSM.translateFile(Files.getFile((*i).first));
            OutputFile * outFile = (*i).second;
		if (fid.isInvalid()) {
		    llvm::errs() << "Error file [" << (*i).first << "] has invalid (zero) fid!\n";
		    //Push the file to the redo list, it might show up in the SM once it's relevant main file is processed
		    continue;
		}
            if (const RewriteBuffer *RewriteBuff =
                GlobalKernRewrite.getRewriteBufferFor(fid)) {
                *(outFile->OS) << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                llvm::errs() << "No (kernel) changes made to " << RewriteSM.getFileEntryForID(fid)->getName() << "\n";
            }
            outFile->OS->flush();
	    clearOutputFile(outFile, &Files);
        }


	llvm::errs() << "Retained " << AllASTs.size() << " ASTContexts!\n";
	//Release the retained ASTContexts now that we're done with their contents
	for (ASTContVec::iterator ast = AllASTs.begin(); ast != AllASTs.end(); ast++) (*ast)->Release();


	//Before flushing the header, we must wrap the function definitions with the closing #ifdef __cplusplus brace
	*cu2cl_header << "\n#ifdef __cplusplus\n";
	*cu2cl_header << "}\n";
	*cu2cl_header << "#endif\n";

	//Add standard boilerplate to the header
	cu2cl_util->flush();
	cu2cl_header->flush();
	cu2cl_kernel->flush();

	
}

