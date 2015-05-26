/*
* CU2CL - A prototype CUDA-to-OpenCL translator built on the Clang compiler infrastructure
* Version 0.6.2b (beta)
*
* (c) 2010-2014 Virginia Tech
* This version of CU2CL is licensed for non-commercial use only,
*  as specified in LICENSE. For all other use contact vtiplicensing@vtip.org
* 
* Authors: Gabriel Martinez, Paul Sathre
*
*/

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"

#include "clang/Rewrite/Rewriter.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Regex.h"

#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>

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
#define CL_COMMAND_QUEUE_QUERY \
    "cl_int __cu2cl_CommandQueueQuery(cl_command_queue commands) {\n" \
    "   cl_int ret;\n" \
    "   cl_event event;\n" \
    "   clEnqueueMarker(commands, &event);\n" \
    "   clGetEventInfo(commands, &event);\n" \
    "}\n\n"

//A function to take the time between two events, emulating cudaEventElapsedTime
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
#define CL_EVENT_QUERY \
    "cl_int __cu2cl_EventQuery(cl_event event) {\n" \
    "    cl_int ret;\n" \
    "    clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL);\n" \
    "    return ret;\n" \
    "}\n\n"

//A function to emulate the behavior (not necessarily semantics) of cudaMallocHost
// allocates a device buffer, then maps it into the host address space, and returns a pointer to it
#define CL_MALLOC_HOST \
    "cl_int __cu2cl_MallocHost(void **ptr, size_t size, cl_mem *clMem) {\n" \
    "    cl_int ret;\n" \
    "    *clMem = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);\n" \
    "    *ptr = clEnqueueMapBuffer(__cu2cl_CommandQueue, *clMem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);\n" \
    "    return ret;\n" \
    "}\n\n"

//A function to emulate the behavior (not necessarily semantics) of cudaFreeHost
// unmaps a buffer allocated with __cu2cl_MallocHost, then releases the associated device buffer
#define CL_FREE_HOST \
    "cl_int __cu2cl_FreeHost(void *ptr, cl_mem clMem) {\n" \
    "    cl_int ret;\n" \
    "    ret = clEnqueueUnmapMemObject(__cu2cl_CommandQueue, clMem, ptr, 0, NULL, NULL);\n" \
    "    ret |= clReleaseMemObject(clMem);\n" \
    "    return ret;\n" \
    "}\n\n"

//A helper function to scan all platforms for all devices and accumulate them into a single array
// can be used independently of __cu2cl_setDevice, but not intended
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
#define CU2CL_SET_DEVICE \
    "void  __cu2cl_SetDevice(cl_uint devID) {\n" \
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
using namespace llvm::sys::path;

namespace {


    //internal flags for command-line toggles
    bool AddInlineComments = true; //defaults to ON, turn off with '-plugin-arg-rewrite-cuda no-comments' at the command line

//Simple timer calls that get injected if enabled
#ifdef CU2CL_ENABLE_TIMING
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

class RewriteCUDA;

//The class prototype necessary to trigger rewriting #included files
class RewriteIncludesCallback : public PPCallbacks {
private:
    RewriteCUDA *RCUDA;

public:
    RewriteIncludesCallback(RewriteCUDA *);

    virtual void InclusionDirective(SourceLocation, const Token &,
                                    llvm::StringRef, bool,
                                    const FileEntry *, SourceLocation,
                                    StringRef, StringRef);

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
    typedef std::map<FileID, llvm::raw_ostream *> IDOutFileMap;
    typedef std::map<llvm::StringRef, std::list<llvm::StringRef> > StringRefListMap;

    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;

    Rewriter HostRewrite;
    Rewriter KernelRewrite;

    //Rewritten files
    FileID MainFileID;
    llvm::raw_ostream *MainOutFile;
    llvm::raw_ostream *MainKernelOutFile;
    IDOutFileMap OutFiles;
    IDOutFileMap KernelOutFiles;
    //TODO lump IDs and both outfiles together

    StringRefListMap Kernels;

    std::set<DeclGroupRef, cmpDG> GlobalVarDeclGroups;
    std::set<DeclGroupRef, cmpDG> CurVarDeclGroups;
    std::set<DeclGroupRef, cmpDG> DeviceMemDGs;
    std::set<DeclaratorDecl *> DeviceMemVars;
    std::set<VarDecl *> HostMemVars;
    std::set<VarDecl *> ConstMemVars;
    std::set<VarDecl *> SharedMemVars;
    std::set<ParmVarDecl *> CurRefParmVars;

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

    //Preamble string to insert at top of main kernel file
    std::string DevPreamble;
    std::string DevFunctions;

    //Pre- and Postamble strings that bundle OpenCL boilerplate for a translation unit
    std::string CLInit;
    std::string CLClean;

    //Flags used by the rewriter
    bool IncludingStringH;
    bool UsesCUDADeviceProp;
    bool UsesCUDAMemset;
    bool UsesCUDAStreamQuery;
    bool UsesCUDAEventElapsedTime;
    bool UsesCUDAEventQuery;
    bool UsesCUDAMallocHost;
    bool UsesCUDAFreeHost;
    bool UsesCUDASetDevice;


	uint64_t TransTime;    

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
	bool w;
	struct commentBufferNode * n;
	};
    struct commentBufferNode * tail, * head;

    //Buffer a new comment destined to be added to output OpenCL source files
    //WARNING: Not threadsafe at all!
    void bufferComment(SourceLocation loc, std::string str, bool writer) {
	struct commentBufferNode * n = (struct commentBufferNode *)malloc(sizeof(commentBufferNode));
	n->s = (char *)malloc(sizeof(char)*(str.length()+1));
	str.copy(n->s, str.length());
	n->s[str.length()] = '\0';
	n->l = loc.getPtrEncoding(); n->w = writer; n->n = NULL;

	tail->n = n;
	tail = n;
    }

    //Method to output comments destined for addition to output OpenCL source
    // which have been buffered to avoid sideeffects with other rewrites
    //WARNING: Not threadsafe at all!
    void writeComments() {
	struct commentBufferNode * curr = head->n;
	while (curr != NULL) { // as long as we have more comment nodes..
	    // inject the comment to the host output stream if true
	    if (curr->w) {
		HostRewrite.InsertTextBefore(SourceLocation::getFromPtrEncoding(curr->l), llvm::StringRef(curr->s));
	    } else { // or the kernel output stream if false
		KernelRewrite.InsertTextBefore(SourceLocation::getFromPtrEncoding(curr->l), llvm::StringRef(curr->s));
	    }
	    //move ahead, then destroy the current node
	    curr = curr->n;
	    free(head->n->s);
	    free(head->n);
	    head->n = curr;
	}
	tail = head;
    }
    
    // Workhorse for CU2CL diagnostics, provides independent specification of multiple err_notes
    //  and inline_notes which should be dumped to stderr and translated output, respectively
    // TODO: Eventually this could stand to be implemented using the real Basic/Diagnostic subsystem
    //  but at the moment, the set of errors isn't mature enough to make it worth it.
    // It's just cheaper to directly throw it more readily-adjustable strings until we set the 
    //  error messages in stone.
    void emitCU2CLDiagnostic(SourceLocation loc, std::string severity_str, std::string err_note, std::string inline_note, Rewriter &writer) {
        //Sanitize all incoming locations to make sure they're not MacroIDs
        SourceLocation expLoc = SM->getExpansionLoc(loc);
        SourceLocation writeLoc;

        //assemble both the stderr and inlined source output strings
        std::stringstream inlineStr;
        std::stringstream errStr;
        if (expLoc.isValid()){
	    //Tack the source line information onto the diagnostic
            errStr << SM->getBufferName(expLoc) << ":" << SM->getExpansionLineNumber(expLoc) << ":" << SM->getExpansionColumnNumber(expLoc) << ": ";
            //grab the start of column write location
            writeLoc = SM->translateLineCol(SM->getFileID(expLoc), SM->getExpansionLineNumber(expLoc), 1);
        }
	inlineStr << "/* ";
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
				if (&writer == &HostRewrite) {
				bufferComment(writeLoc, inlineStr.str(), true);
				} else {
				bufferComment(writeLoc, inlineStr.str(), false);

				}
			}
        }
        //Send the stderr string to stderr
        llvm::errs() << errStr.str();
    }
    
    // Convenience method for dumping the same CU2CL error to both stderr and inlined comments
    //  using the mechanism above
    // Assumes the err_note is replicated as the inline comment to add to source.
    void emitCU2CLDiagnostic(SourceLocation loc, std::string severity_str, std::string err_note, Rewriter &writer) {
        emitCU2CLDiagnostic(loc, severity_str, err_note, err_note, writer);
    }

    //Convenience method for getting a string of raw text between two SourceLocations
    std::string getStmtText(Stmt *s) {
        SourceLocation a(SM->getExpansionLoc(s->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
        return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
    }

    //Simple function to strip attributes from host functions that may be declared as 
    // both __host__ and __device__, then passes off to the host-side statement rewriter
    void RewriteHostFunction(FunctionDecl *hostFunc) {
        //Remove any CUDA function attributes
        if (CUDAHostAttr *attr = hostFunc->getAttr<CUDAHostAttr>()) {
            RewriteAttr(attr, "", HostRewrite);
        }
        if (CUDADeviceAttr *attr = hostFunc->getAttr<CUDADeviceAttr>()) {
            RewriteAttr(attr, "", HostRewrite);
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
                ReplaceStmtWithText(e, str, HostRewrite);
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

	//Detect CUDA C style kernel launches ie. fooKern<<<Grid, Block, shared, stream>>>(args..);
	// the Runtime and Driver API's launch mechanisms would be handled with the rest of the API calls
        if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(e)) {
            //Short-circuit templated kernel launches
            if (kce->isTypeDependent()) {
                emitCU2CLDiagnostic(kce->getLocStart(), "CU2CL Untranslated", "Template-dependent kernel call", HostRewrite);
                return false;
            }
	    //Short-circuit launching a function pointer until we can handle it
	    else if (kce->getDirectCallee() == 0 && dyn_cast<ImplicitCastExpr>(kce->getCallee())) {
                emitCU2CLDiagnostic(kce->getLocStart(), "CU2CL Unhandled", "Function pointer as kernel call", HostRewrite);
                return false;
            }
	    //If it's not a templated or pointer launch, proceed with translation
            newExpr = RewriteCUDAKernelCall(kce);
            return true;
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(e)) {
            if (ce->isTypeDependent()) {
                emitCU2CLDiagnostic(ce->getLocStart(), "CU2CL Untranslated", "Template-dependent host call", HostRewrite);
                return false;
            }
            //This catches some errors encountered with heavily-nested, PP-assembled function-like macros
	    // mostly observed within the OpenGL and GLUT headers
            if (ce->getDirectCallee() == 0) {
                emitCU2CLDiagnostic(SM->getExpansionLoc(ce->getLocStart()), "CU2CL Unhandled", "Could not identify direct callee in expression", HostRewrite);
            }
	    //This catches all Runtime API calls, since they are all prefixed by "cuda"
	    //TODO: Recognize Driver API calls that are prefixed with just "cu"
	    else if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0)
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
                    newExpr = getStmtText(dre) + name; //PrintStmtToString(dre) + name;
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
	//Catches dim3 declarations of the form: some_var=dim3(x,y,z);
	// the RHS is considered a temporary object
        else if (CXXTemporaryObjectExpr *cte = dyn_cast<CXXTemporaryObjectExpr>(e)) {
           // emitCU2CLDiagnostic(cte->getLocStart(), "CU2CL Note", "Identified as CXXTemporaryObjectExpr", HostRewrite);
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
    //TODO: add Driver API calls, prefixed with "cu"
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
            RewriteType(tl, "cl_device_id", HostRewrite);
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
                HostGlobalVars += "cl_device_id * __cu2cl_AllDevices = NULL;\n";
                HostGlobalVars += "cl_uint __cu2cl_AllDevices_curr_idx = 0;\n";
                HostGlobalVars += "cl_uint __cu2cl_AllDevices_size = 0;\n";
                HostFunctions += CU2CL_SCAN_DEVICES;
                HostFunctions += CU2CL_SET_DEVICE;
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
            emitCU2CLDiagnostic(cudaCall->getLocStart(), "CU2CL Warning", "CU2CL Identified cudaSetDevice usage", HostRewrite);
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
                HostFunctions += CL_COMMAND_QUEUE_QUERY;
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
        else if (funcName == "cudaEventCreate") {
	//TODO: Replace with clCreateUserEvent
            //Remove the call
            newExpr = "";
        }
        else if (funcName == "cudaEventCreateWithFlags") {
	//TODO: Replace with clSetUserEventStatus
            //Remove the call
            newExpr = "";
        }
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
                HostFunctions += CL_EVENT_ELAPSED_TIME;
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
                HostFunctions += CL_EVENT_QUERY;
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
                HostFunctions += CL_MALLOC_HOST;
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
                HostFunctions += CL_FREE_HOST;
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
		emitCU2CLDiagnostic(cudaCall->getLocStart(), "CU2CL Note", "Identified member expression in cudaMalloc device pointer", HostRewrite);
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
emitCU2CLDiagnostic(cudaCall->getLocStart(), "CU2CL Note", "Rewriting single decl", HostRewrite);
                //Change variable's type to cl_mem
                TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
                RewriteType(tl, "cl_mem ", HostRewrite);
            }

            //Add var to DeviceMemVars
            DeviceMemVars.insert(var);
        }
        else if (funcName == "cudaMallocHost") {
            //Replace with __cu2cl_MallocHost
            if (!UsesCUDAMallocHost) {
                HostFunctions += CL_MALLOC_HOST;
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
                emitCU2CLDiagnostic(cudaCall->getLocStart(), "CU2CL Unsupported", "Unsupported cudaMemcpyKind: " + enumString, HostRewrite);
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
                emitCU2CLDiagnostic(cudaCall->getLocStart(), "CU2CL Unsupported", "Unsupported cudaMemcpyKind: " + enumString, HostRewrite);
            }
        }
        //else if (funcName == "cudaMemcpyToSymbol") {
            //TODO: implement
        //}
        else if (funcName == "cudaMemset") {
            if (!UsesCUDAMemset) {
                HostFunctions += CL_MEMSET;
                DevFunctions += CL_MEMSET_KERNEL;
                llvm::StringRef r = filename(SM->getFileEntryForID(MainFileID)->getName());
                std::list<llvm::StringRef> &l = Kernels[r];
                l.push_back("__cu2cl_Memset");
                HostKernels += "cl_kernel __cu2cl_Kernel___cu2cl_Memset;\n";
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
            emitCU2CLDiagnostic(SM->getExpansionLoc(cudaCall->getLocStart()), "CU2CL Unsupported", "Unsupported CUDA call: " + funcName, HostRewrite);
            return false;
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
		emitCU2CLDiagnostic(kernelCall->getLocStart(), "CU2CL Note", comment.str(), HostRewrite);
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
	    emitCU2CLDiagnostic(construct->getLocStart(), "CU2CL Note", "Fast-tracked dim3 type without cast", HostRewrite);
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
                args << "localWorkSize[0] = " << getStmtText(dre) << ";\n";
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
	    emitCU2CLDiagnostic(construct->getLocStart(), "CU2CL Note", "Fast-tracked dim3 type without cast", HostRewrite);
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
                args << "globalWorkSize[0] = (" << getStmtText(dre) << ")*localWorkSize[0];\n";
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
            //TODO: Do something with __constant__ memory declarations
            RewriteAttr(constAttr, "", HostRewrite);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", HostRewrite);
            //DeviceMemVars.insert(var);
            ConstMemVars.insert(var);

            TypeLoc origTL = var->getTypeSourceInfo()->getTypeLoc();
            if (LastLoc.isNull() || origTL.getBeginLoc() != LastLoc.getBeginLoc()) {
                LastLoc = origTL;
                RewriteType(origTL, "cl_mem", HostRewrite);
            }
            return;
        }
        else if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
            //Handle __shared__ memory declarations
            RewriteAttr(sharedAttr, "", HostRewrite);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", HostRewrite);
            //TODO rewrite shared mem
            //If extern, remove extern keyword and make into pointer
            //if (var->isExtern())
            SharedMemVars.insert(var);
        }
        else if (CUDADeviceAttr *attr = var->getAttr<CUDADeviceAttr>()) {
            //Handle __device__ memory declarations
            RewriteAttr(attr, "", HostRewrite);
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
                RewriteType(tl, "size_t", HostRewrite);
            }
            else if (type == "struct cudaDeviceProp") {
                if (!UsesCUDADeviceProp) {
                    HostDecls += CL_DEVICE_PROP;
                    HostFunctions += CL_GET_DEVICE_PROPS;
                    UsesCUDADeviceProp = true;
                }
                RewriteType(tl, "__cu2cl_DeviceProp", HostRewrite);
            }
            else if (type == "cudaStream_t") {
                RewriteType(tl, "cl_command_queue", HostRewrite);
            }
            else if (type == "cudaEvent_t") {
                RewriteType(tl, "cl_event", HostRewrite);
            }
            else {
                std::string newType = RewriteVectorType(type, true);
                if (newType != "")
                    RewriteType(tl, newType, HostRewrite);
            }
            //TODO check other CUDA-only types to rewrite
        }

        //Rewrite initial value
        if (var->hasInit()) {
            Expr *e = var->getInit();
            std::string s;
            if (RewriteHostExpr(e, s)) {
                //Special cases for dim3s
                if (type == "dim3") {
                    CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(e);
                    if (cce && cce->getNumArgs() > 1) {
                        SourceRange parenRange = cce->getParenRange();
                        if (parenRange.isValid()) {
                            HostRewrite.ReplaceText(
                                    parenRange.getBegin(),
                                    HostRewrite.getRangeSize(parenRange),
                                    s);
                        }
                        else {
                            HostRewrite.InsertTextAfter(
                                    PP->getLocForEndOfToken(var->getLocation()),
                                    s);
                        }
                    }
                    else {
                        ReplaceStmtWithText(e, s, HostRewrite);
                    }

                    //Add [3] to end/start of var identifier
                    if (origTL.getTypePtr()->isPointerType())
                        HostRewrite.InsertTextBefore(
                                var->getLocation(),
                                "*");
                    else
                        HostRewrite.InsertTextBefore(
                                PP->getLocForEndOfToken(var->getLocation()),
                                "[3]");
                }
                else
                    ReplaceStmtWithText(e, s, HostRewrite);
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

        if (kernelFunc->hasAttr<CUDAGlobalAttr>()) {
            //If host-callable, get and store kernel filename
            llvm::StringRef r = filename(SM->getFileEntryForID(SM->getFileID(kernelFunc->getLocation()))->getName());
            std::list<llvm::StringRef> &l = Kernels[r];
            l.push_back(kernelFunc->getName());
            HostKernels += "cl_kernel __cu2cl_Kernel_" + kernelFunc->getName().str() + ";\n";
        }

        //Rewrite kernel attributes
	//__global__ must be mapped to __kernel
        if (CUDAGlobalAttr *attr = kernelFunc->getAttr<CUDAGlobalAttr>()) {
            RewriteAttr(attr, "__kernel", KernelRewrite);
        }
	//__device__ functions don't have any attributes in OpenCL
        if (CUDADeviceAttr *attr = kernelFunc->getAttr<CUDADeviceAttr>()) {
            RewriteAttr(attr, "", KernelRewrite);
        }
	//OpenCL kernel code has no such thing as a __host__ function
	// these are already preserved in the host code elsewhere
        if (CUDAHostAttr *attr = kernelFunc->getAttr<CUDAHostAttr>()) {
            RewriteAttr(attr, "", KernelRewrite);
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

        if (parmDecl->getOriginalType()->isTemplateTypeParmType()) emitCU2CLDiagnostic(parmDecl->getLocStart(), "CU2CL Unhandled", "Detected templated parameter", KernelRewrite);
        TypeLoc tl = parmDecl->getTypeSourceInfo()->getTypeLoc();

	//A rewrite offset is declared to do bookkeeping for the amount of
	// of characters added. This prevents a bug in which consecutive
	// parameters would be overwritten
	int rewriteOffset = 0;
        if (isFuncGlobal && tl.getTypePtr()->isPointerType()) {
            KernelRewrite.InsertTextBefore(
                    tl.getBeginLoc(),
                    "__global ");
		rewriteOffset -= 9; //ignore the 9 chars of "__global "
        }
        else if (ReferenceTypeLoc *rtl = dyn_cast<ReferenceTypeLoc>(&tl)) {
            KernelRewrite.ReplaceText(
                    rtl->getSigilLoc(),
                    KernelRewrite.getRangeSize(rtl->getLocalSourceRange()),
                    "*");
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
            RewriteType(tl, newType, KernelRewrite, rewriteOffset);
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
                ReplaceStmtWithText(e, str, KernelRewrite);
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
                        newExpr = getStmtText(dre);

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
                        newExpr = getStmtText(dre);

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
                emitCU2CLDiagnostic(e->getLocStart(), "CU2CL Unhandled", "Template-dependent kernel expression", KernelRewrite);
                return false;
            }
	    //This catches potential segfaults related to function pointe usage
            if (ce->getDirectCallee() == 0) {
                emitCU2CLDiagnostic(e->getLocStart(), "CU2CL Warning", "Unable to identify expression direct callee", KernelRewrite);
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
            RewriteAttr(sharedAttr, "__local", KernelRewrite);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", KernelRewrite);
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
                RewriteType(tl, "size_t", KernelRewrite);
            }
            else {
                std::string newType = RewriteVectorType(type, false);
                if (newType != "")
                    RewriteType(tl, newType, KernelRewrite);
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
                        SourceRange parenRange = cce->getParenRange();
                        if (parenRange.isValid()) {
                            KernelRewrite.ReplaceText(
                                    parenRange.getBegin(),
                                    KernelRewrite.getRangeSize(parenRange),
                                    s);
                        }
                        else {
                            KernelRewrite.InsertTextAfter(
                                    PP->getLocForEndOfToken(var->getLocation()),
                                    s);
                        }
                    }
                    else
                        ReplaceStmtWithText(e, s, KernelRewrite);

                    //Add [3]/* to end/start of var identifier
                    if (origTL.getTypePtr()->isPointerType())
                        KernelRewrite.InsertTextBefore(
                                var->getLocation(),
                                "*");
                    else
                        KernelRewrite.InsertTextBefore(
                                PP->getLocForEndOfToken(var->getLocation()),
                                "[3]");
                }
                else
                    ReplaceStmtWithText(e, s, KernelRewrite);
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
    void RewriteType(TypeLoc tl, std::string replace, Rewriter &rewrite, int rangeOffset = 0) {

	SourceRange realRange(tl.getBeginLoc(), PP->getLocForEndOfToken(tl.getBeginLoc()));

        rewrite.ReplaceText(tl.getBeginLoc(), rewrite.getRangeSize(tl.getLocalSourceRange()) + rangeOffset, replace);
    }

    //The workhorse that takes the constructed replacement attribute and inserts it in place of the old one
    void RewriteAttr(Attr *attr, std::string replace, Rewriter &rewrite) {
        SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
        SourceRange realRange(instLoc,
                              PP->getLocForEndOfToken(instLoc));
        rewrite.ReplaceText(instLoc, rewrite.getRangeSize(realRange), replace);
	
    }

    //Completely remove a function from one of the output streams
    //Used to split host and kernel code
    void RemoveFunction(FunctionDecl *func, Rewriter &rewrite) {
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

        //Calculate the SourceLocation of the first token of the function,
	// handling a number of corner cases
        //TODO find first specifier location
        //TODO find storage class specifier
        startLoc = func->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        if (tk == FunctionDecl::TK_FunctionTemplate) {
            FunctionTemplateDecl *ftd = func->getDescribedFunctionTemplate();
            tempLoc = ftd->getSourceRange().getBegin();
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }
        if (func->hasAttrs()) {
            Attr *attr = (func->getAttrs())[0];

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
        if (startLoc.getRawEncoding() == NULL) {
            startLoc = func->getQualifierLoc().getBeginLoc();
            if (startLoc.getRawEncoding() != NULL) emitCU2CLDiagnostic(startLoc, "CU2CL Note", "Removed constructor/deconstructor", rewrite);
        }

	//If we still haven't found an appropriate startLoc, something's atypical
	//Grab whatever Clang thinks is the startLoc, and remove from there
        if (startLoc.getRawEncoding() == NULL) {
            startLoc = func->getLocStart();
	    //If even Clang doesn't have any idea where to start, give up
            if (startLoc.getRawEncoding() == NULL) {
                emitCU2CLDiagnostic(startLoc, "CU2CL Error", "Unable to determine valid start location for function \"" + func->getNameAsString() + "\"", rewrite);
                return;
            }
            emitCU2CLDiagnostic(startLoc, "CU2CL Warning", "Inferred function start location, removal may be incomplete", rewrite);
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
        }
        rewrite.RemoveText(startLoc,
                           rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    //Get rid of a variable declaration
    // useful for pulling global host variables out of kernel code
    void RemoveVar(VarDecl *var, Rewriter &rewrite) {
        SourceLocation startLoc, endLoc, tempLoc;

        //Find startLoc
        //TODO find first specifier location
        //TODO find storage class specifier
        startLoc = var->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        if (var->hasAttrs()) {
            Attr *attr = (var->getAttrs())[0];
            tempLoc = SM->getExpansionLoc(attr->getLocation());
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }

        //Find endLoc
        if (var->hasInit()) {
            Expr *init = var->getInit();
            endLoc = SM->getExpansionLoc(init->getLocEnd());
        }
        else {
            //Find location of semi-colon
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            if (ArrayTypeLoc *atl = dyn_cast<ArrayTypeLoc>(&tl)) {
                endLoc = SM->getExpansionLoc(atl->getRBracketLoc());
            }
            else
                endLoc = SM->getExpansionLoc(var->getSourceRange().getEnd());
        }
        rewrite.RemoveText(startLoc,
                           rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
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
    bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr, Rewriter &Rewrite) {
        SourceRange origRange = OldStmt->getSourceRange();
        SourceLocation s = SM->getExpansionLoc(origRange.getBegin());
        SourceLocation e = SM->getExpansionLoc(origRange.getEnd());
        return Rewrite.ReplaceText(s,
                                   Rewrite.getRangeSize(SourceRange(s, e)),
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
    RewriteCUDA(CompilerInstance *comp, llvm::raw_ostream *HostOS,
                llvm::raw_ostream *KernelOS) :
        ASTConsumer(), CI(comp),
        MainOutFile(HostOS), MainKernelOutFile(KernelOS) { }

    virtual ~RewriteCUDA() { }

    virtual void Initialize(ASTContext &Context) {
        SM = &Context.getSourceManager();
        LO = &CI->getLangOpts();
        PP = &CI->getPreprocessor();

        PP->addPPCallbacks(new RewriteIncludesCallback(this));

        HostRewrite.setSourceMgr(*SM, *LO);
        KernelRewrite.setSourceMgr(*SM, *LO);
        MainFileID = SM->getMainFileID();

        if (MainFuncName == "")
            MainFuncName = "main";

        HostIncludes += "#ifdef __APPLE__\n";
        HostIncludes += "#include <OpenCL/opencl.h>\n";
        HostIncludes += "#else\n";
        HostIncludes += "#include <CL/opencl.h>\n";
        HostIncludes += "#endif\n";
        HostIncludes += "#include <stdlib.h>\n";
        HostIncludes += "#include <stdio.h>\n";
        HostGlobalVars += "cl_platform_id __cu2cl_Platform;\n";
        HostGlobalVars += "cl_device_id __cu2cl_Device;\n";
        HostGlobalVars += "cl_context __cu2cl_Context;\n";
        HostGlobalVars += "cl_command_queue __cu2cl_CommandQueue;\n\n";
        HostGlobalVars += "size_t globalWorkSize[3];\n";
        HostGlobalVars += "size_t localWorkSize[3];\n";
        HostFunctions += LOAD_PROGRAM_SOURCE;

        IncludingStringH = false;
        UsesCUDADeviceProp = false;
        UsesCUDAMemset = false;
        UsesCUDAStreamQuery = false;
        UsesCUDAEventElapsedTime = false;
        UsesCUDAEventQuery = false;
        UsesCUDASetDevice = false;

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
        if (!SM->isFromMainFile(loc)) {
            llvm::StringRef fileExt = extension(SM->getPresumedLoc(loc).getFilename());
            if (fileExt.equals(".cu") || fileExt.equals(".cuh")) {
                //If #included and a .cu or .cuh file, rewrite
                //TODO: accept .c/.h/.cpp/.hpp files but only if they contain CUDA Runtime calls
                if (OutFiles.find(SM->getFileID(loc)) == OutFiles.end()) {
                    //Create new files
                    FileID fileid = SM->getFileID(loc);
                    std::string filename = SM->getPresumedLoc(loc).getFilename();
                    size_t dotPos = filename.rfind('.');
		    filename = filename + "-cl" + filename.substr(dotPos);
                    llvm::raw_ostream *hostOS = CI->createDefaultOutputFile(false, filename, "h");
                    llvm::raw_ostream *kernelOS = CI->createDefaultOutputFile(false, filename, "cl");
                    if (hostOS && kernelOS) {
                        OutFiles[fileid] = hostOS;
                        KernelOutFiles[fileid] = kernelOS;
                    }
                    else {
			//We've already registered an output stream for this
			// input file, so proceed
                    }
                }
            }
            else {
                //Don't stop parsing, just skip the file
                return true;
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
                            RemoveFunction(fd, KernelRewrite);
    
                            if (fd->getNameAsString() == MainFuncName) {
                                RewriteMain(fd);
                            }
                        } else {
                            emitCU2CLDiagnostic(fd->getLocStart(), "CU2CL Note", "Skipped rewrite of implicitly defined function \"" + fd->getNameAsString() + "\"", HostRewrite);
                        }
                    }
                }
            }
            //Handles globally defined C or C++ functions
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
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
                            RemoveFunction(fd, HostRewrite);
                    }
                    else {
                        //Simply a host function, so rewrite
                        RewriteHostFunction(fd);
                        //and remove from kernel
                        RemoveFunction(fd, KernelRewrite);
    
                        if (fd->getNameAsString() == MainFuncName) {
                            RewriteMain(fd);
                        }
                    }
                } else {
                    if (fd->getTemplateSpecializationInfo())
                    emitCU2CLDiagnostic(fd->getTemplateSpecializationInfo()->getTemplate()->getLocStart(), "CU2CL Untranslated", "Unable to translate template function", HostRewrite);
                    else llvm::errs() << "Non-rewriteable function without TemplateSpecializationInfo detected\n";
                }
            }
            else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                RemoveVar(vd, KernelRewrite);
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
            //TODO rewrite type declarations
        }
return true;
    }

    //Compltely processes each file included on the invokation command line
    //Traditionally CU2CL is run on one standalone source file (with #includes)
    // at a time 
    virtual void HandleTranslationUnit(ASTContext &) {
#ifdef CU2CL_ENABLE_TIMING
        init_time();
#endif

        //Declare global clPrograms, one for each kernel-bearing source file
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string r = idCharFilter((*i).first);
            HostGlobalVars += "cl_program __cu2cl_Program_" + r + ";\n";
        }
        //TODO: differentiate main preamble from helpers as part of multiple compliation
        //Insert host preamble at top of main file
        HostPreamble = HostIncludes + "\n" + HostDecls + "\n" + HostGlobalVars + "\n" + HostKernels + "\n" + HostFunctions;
        HostRewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), HostPreamble);
        //Insert device preamble at top of main kernel file
        DevPreamble = DevFunctions;
        KernelRewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), DevPreamble);

        //Construct OpenCL initialization boilerplate
        CLInit += "\n";
        CLInit += "const char *progSrc;\n";
        CLInit += "size_t progLen;\n\n";
        //Rather than obviating these lines to support cudaSetDevice, we'll assume these lines
        // are *always* included, and IFF cudaSetDevice is used, include code to instead scan
        // *all* devices, and allow for reinitialization
        CLInit += "clGetPlatformIDs(1, &__cu2cl_Platform, NULL);\n";
        CLInit += "clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 1, &__cu2cl_Device, NULL);\n";
        CLInit += "__cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);\n";
        CLInit += "__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);\n";
	//For each program file, load the translated source
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string file = idCharFilter((*i).first);
            std::list<llvm::StringRef> &l = (*i).second;
            CLInit += "progLen = __cu2cl_LoadProgramSource(\"" + (*i).first.str() + "-cl.cl\", &progSrc);\n";
            CLInit += "__cu2cl_Program_" + file + " = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);\n";
            CLInit += "free((void *) progSrc);\n";
            CLInit += "clBuildProgram(__cu2cl_Program_" + file + ", 1, &__cu2cl_Device, \"-I .\", NULL, NULL);\n";
	    // and initialize all its kernels
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLInit += "__cu2cl_Kernel_" + kernelName + " = clCreateKernel(__cu2cl_Program_" + file + ", \"" + kernelName + "\", NULL);\n";
            }
        }
        

        //TODO: differentiate main cleanup from helpers as part of separate compilation
        //Insert cleanup code at bottom of main
        CLClean += "\n";
	//For each loaded cl_program
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            std::string file = idCharFilter((*i).first);
	    //Release its kernels
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLClean += "clReleaseKernel(__cu2cl_Kernel_" + kernelName + ");\n";
            }
	    //Then release the program itself
            CLClean += "clReleaseProgram(__cu2cl_Program_" + file + ");\n";
        }
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
        }
        CLClean += "clReleaseCommandQueue(__cu2cl_CommandQueue);\n";
        CLClean += "clReleaseContext(__cu2cl_Context);\n";
        
        //Insert boilerplate at the top of file as a comment, if it doesn't have a main method
	if (MainDecl == NULL) {
	    std::stringstream boilStr;
	    boilStr << "No main() found\nCU2CL Boilerplate inserted here:\nCU2CL Initialization:\n" << CLInit << "\n\nCU2CL Cleanup:\n" <<CLClean; 
            emitCU2CLDiagnostic(SM->getLocForStartOfFile(MainFileID), "CU2CL Unhandled", "No main() found!\n\tBoilerplate inserted as header comment!\n", boilStr.str(), HostRewrite);
        }
	//Otherwise, insert it the start and end of the main method
	else {
	    CompoundStmt *mainBody = dyn_cast<CompoundStmt>(MainDecl->getBody());
	    HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(mainBody->getLBracLoc(), 0), CLInit);
	    HostRewrite.InsertTextBefore(mainBody->getRBracLoc(), CLClean);
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
                    //Change variable's type to cl_mem
                    replace += "cl_mem " + vd->getNameAsString();
		    if (vd->getType()->isArrayType()) {
			//make sure to grab the array [...] Expr too
			emitCU2CLDiagnostic(vd->getLocStart(), "CU2CL Note", "Device var \"" + vd->getNameAsString() + "\" has array type!\n", HostRewrite);
			replace += "[";
			if (const DependentSizedArrayType *arr = dyn_cast<DependentSizedArrayType>(vd->getType().getCanonicalType())) {
			    replace += getStmtText(arr->getSizeExpr());
			} else if (const VariableArrayType *arr = dyn_cast<VariableArrayType>(vd->getType().getCanonicalType())) {
			    replace += getStmtText(arr->getSizeExpr());
			} else if (const ConstantArrayType *arr = dyn_cast<ConstantArrayType>(vd->getType().getCanonicalType())) {
			    replace += arr->getSize().toString(10, true);	
			}
			replace += "]";
		    }
                }
                else {
                    replace += PrintDeclToString(vd);
                }
                if ((iDG + 1) == DG.end()) {
                    end = (*iDG)->getLocEnd();
                }
                else {
                    replace += ";\n";
                }
            }
            HostRewrite.ReplaceText(start, HostRewrite.getRangeSize(SourceRange(SM->getExpansionLoc(start), SM->getExpansionLoc(end))), replace);
        }
	//Write all buffered comments to output streams
	writeComments();
	//And clean up the list's sentinel
	free(head);
	head = NULL;
	tail = NULL;

        //Output main file's rewritten buffer
        if (const RewriteBuffer *RewriteBuff =
            HostRewrite.getRewriteBufferFor(MainFileID)) {
            *MainOutFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
        }
        else {
            //TODO use diagnostics for pretty errors
            llvm::errs() << "No changes made to " << SM->getFileEntryForID(MainFileID)->getName() << "\n";
        }
        //Output main kernel file's rewritten buffer
        if (const RewriteBuffer *RewriteBuff =
            KernelRewrite.getRewriteBufferFor(MainFileID)) {
            *MainKernelOutFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
        }
        else {
            //TODO use diagnostics for pretty errors
            llvm::errs() << "No changes made to " << SM->getFileEntryForID(MainFileID)->getName() << " kernel\n";
        }
        //Flush rewritten files
        MainOutFile->flush();
        MainKernelOutFile->flush();

	//Flush rewritten #included host files
        for (IDOutFileMap::iterator i = OutFiles.begin(), e = OutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
	    //If changes were made, bring them in from the rewriter
            if (const RewriteBuffer *RewriteBuff =
                HostRewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
	    //Otherwise just dump the file directly
            else {
                llvm::StringRef fileBuf = SM->getBufferData(fid);
                *outFile << std::string(fileBuf.begin(), fileBuf.end());
            }
            outFile->flush();
        }
	//Flush rewritten #included kernel files
        for (IDOutFileMap::iterator i = KernelOutFiles.begin(), e = KernelOutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
            if (const RewriteBuffer *RewriteBuff =
                KernelRewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                llvm::errs() << "No (kernel) changes made to " << SM->getFileEntryForID(fid)->getName() << " kernel\n";
            }
            outFile->flush();
        }

#ifdef CU2CL_ENABLE_TIMING
        TransTime += get_time();

        llvm::errs() << "Translation Time: " << TransTime << " microseconds\n";
#endif
    }

    void RewriteInclude(SourceLocation HashLoc, const Token &IncludeTok,
                        llvm::StringRef FileName, bool IsAngled,
                        const FileEntry *File, SourceLocation EndLoc) {
        llvm::StringRef fileExt = extension(SM->getPresumedLoc(HashLoc).getFilename());
        llvm::StringRef includedFile = filename(FileName);
        llvm::StringRef includedExt = extension(includedFile);
        if (SM->isFromMainFile(HashLoc) ||
            fileExt.equals(".cu") || fileExt.equals(".cuh")) {
            //If system-wide include style (#include <foo.h>) is used, don't translate
            if (IsAngled) {
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
		//Remove reference to the CUDA header
                if (includedFile.equals("cuda.h"))
                    HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
	    //Remove quote-included reference to the CUDA header
            else if (includedFile.equals("cuda.h")) {
                HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
	    //If local include style (#include "foo.h") is used, do translate
            else if (includedExt.equals(".cu") || includedExt.equals(".cuh")) {
                FileID fileID = SM->getFileID(HashLoc);
                SourceLocation fileStartLoc = SM->getLocForStartOfFile(fileID);
                llvm::StringRef fileBuf = SM->getBufferData(fileID);
                const char *fileBufStart = fileBuf.begin();
                SourceLocation start = fileStartLoc.getLocWithOffset(includedExt.begin() - fileBufStart);
                SourceLocation end = fileStartLoc.getLocWithOffset((includedExt.end()) - fileBufStart);
		//append new file-type to output sourcefile name
                HostRewrite.ReplaceText(end, 0, "-cl.h");
                KernelRewrite.ReplaceText(end, 0, "-cl.cl");
            }
            else {
                //TODO: store include info to rewrite later?
            }
        }
    }

};


class RewriteCUDAAction : public PluginASTAction {
protected:
    //The factory method needeed to initialize the plugin as an ASTconsumer
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) {
        
        std::string filename = InFile.str();
        size_t dotPos = filename.rfind('.');
	filename = filename + "-cl" + filename.substr(dotPos);
        llvm::raw_ostream *hostOS = CI.createDefaultOutputFile(false, filename, "cpp");
        llvm::raw_ostream *kernelOS = CI.createDefaultOutputFile(false, filename, "cl");
        if (hostOS && kernelOS)
            return new RewriteCUDA(&CI, hostOS, kernelOS);
        //TODO cleanup files?
        return NULL;
    }

    //Handle parsing of arguments to the plugin
    bool ParseArgs(const CompilerInstance &CI,
                   const std::vector<std::string> &args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
            llvm::errs() << "RewriteCUDA arg = " << args[i] << "\n";

	    //a toggle for inline comment generation, default is ON
	    if (args[i] == "no-comments") {
		AddInlineComments = false;		
	    }
	    //TODO: Add ocl-version option, default to 1.0, to account for API changes needed in output
        }
        if (args.size() && args[0] == "help")
            PrintHelp(llvm::errs());

        return true;
    }

    //TODO: implement help output, potentially for additional options
    void PrintHelp(llvm::raw_ostream &ros) {
        ros << "Help for RewriteCUDA plugin goes here\n";
    }

};

RewriteIncludesCallback::RewriteIncludesCallback(RewriteCUDA *RC) :
    RCUDA(RC) {
}

void RewriteIncludesCallback::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                                                 llvm::StringRef FileName, bool IsAngled,
                                                 const FileEntry *File, SourceLocation EndLoc,
                                                 StringRef SearchPath, StringRef RelativePath) {
    RCUDA->RewriteInclude(HashLoc, IncludeTok, FileName, IsAngled, File, EndLoc);
}

}

static FrontendPluginRegistry::Add<RewriteCUDAAction>
X("rewrite-cuda", "translate CUDA to OpenCL");
