#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

#include "clang/Lex/PPCallbacks.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "clang/Lex/Preprocessor.h"

#include "clang/Rewrite/Rewriter.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>

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

#define LOAD_PROGRAM_SOURCE \
    "size_t __cu2cl_loadProgramSource(char *filename, const char **progSrc) {\n" \
    "\tFILE *f = fopen(filename, \"r\");\n" \
    "\tfseek(f, 0, SEEK_END);\n" \
    "\tsize_t len = (size_t) ftell(f);\n" \
    "\t*progSrc = (const char *) malloc(sizeof(char)*len);\n" \
    "\trewind(f);\n" \
    "\tfread((void *) *progSrc, len, 1, f);\n" \
    "\tfclose(f);\n" \
    "\treturn len;\n" \
    "}\n\n"

#define CL_MEMSET \
    "cl_int __cu2cl_Memset(cl_mem devPtr, int value, size_t count) {\n" \
    "    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 0, sizeof(cl_mem), &devPtr);\n" \
    "    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 1, sizeof(cl_uchar), &value);\n" \
    "    clSetKernelArg(__cu2cl_Kernel___cu2cl_Memset, 2, sizeof(cl_uint), &count);\n" \
    "    globalWorkSize[0] = count;\n" \
    "    return clEnqueueNDRangeKernel(__cu2cl_CommandQueue, __cu2cl_Kernel___cu2cl_Memset, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);\n" \
    "}\n\n"

#define CL_MEMSET_KERNEL \
    "__kernel void __cu2cl_Memset(__global uchar *ptr, uchar value, uint num) {\n" \
    "    size_t id = get_global_id(0);\n" \
    "    if (get_global_id(0) < num) {\n" \
    "        ptr[id] = value;\n" \
    "    }\n" \
    "}\n\n"

#define CL_GET_DEVICE_INFO(TYPE, NAME) \
    "    ret |= clGetDeviceInfo(device, CL_DEVICE_" #TYPE ", sizeof(prop->" \
    #NAME "), &prop->" #NAME ", NULL);\n"

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

#define CL_COMMAND_QUEUE_QUERY \
    "cl_int __cu2cl_CommandQueueQuery(cl_command_queue commands) {\n" \
    "   cl_int ret;\n" \
    "   cl_event event;\n" \
    "   clEnqueueMarker(commands, &event);\n" \
    "   clGetEventInfo(commands, &event);\n" \
    "}\n\n"

using namespace clang;
using namespace llvm::sys::path;

namespace {

struct cmpDG {
    bool operator()(DeclGroupRef a, DeclGroupRef b) {
        SourceLocation aLoc = (a.isSingleDecl() ? a.getSingleDecl() : a.getDeclGroup()[0])->getLocStart();
        SourceLocation bLoc = (b.isSingleDecl() ? b.getSingleDecl() : b.getDeclGroup()[0])->getLocStart();
        return aLoc.getRawEncoding() < bLoc.getRawEncoding();
    }
};

class RewriteCUDA;

class RewriteIncludesCallback : public PPCallbacks {
private:
    RewriteCUDA *RCUDA;

public:
    RewriteIncludesCallback(RewriteCUDA *);

    virtual void InclusionDirective(SourceLocation, const Token &,
                                    llvm::StringRef, bool,
                                    const FileEntry *, SourceLocation/*,
                                    const llvm::SmallVectorImpl<char> &*/);

};

/**
 * An AST consumer made to rewrite CUDA to OpenCL.
 **/
class RewriteCUDA : public ASTConsumer {
private:
    typedef std::map<FileID, llvm::raw_ostream *> IDOutFileMap;
    typedef std::map<llvm::StringRef, std::list<llvm::StringRef> > StringRefListMap;

    CompilerInstance *CI;
    SourceManager *SM;
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
    std::set<VarDecl *> DeviceMemVars;

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

    std::string CLInit;
    std::string CLClean;

    //Flags used by the rewriter
    bool IncludingStringH;
    bool UsesCUDADeviceProp;
    bool UsesCUDAMemset;
    bool UsesCUDAStreamQuery;

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

    void RewriteHostFunction(FunctionDecl *hostFunc) {
        //llvm::errs() << "Rewriting host function: " << hostFunc->getName() << "\n";
        //Remove from KernelRewrite
        RemoveFunction(hostFunc, KernelRewrite);
        //Rewrite __host__ attribute
        if (hostFunc->hasAttr<CUDAHostAttr>()) {
            SourceLocation loc = FindAttr(hostFunc->getLocStart(), "__host__");
            RewriteAttr("__host__", loc, HostRewrite);
        }
        if (Stmt *body = hostFunc->getBody()) {
            //TraverseStmt(body, 0);
            RewriteHostStmt(body);
        }
        CurVarDeclGroups.clear();
    }

    void RewriteHostStmt(Stmt *s) {
        //Traverse children and recurse
        for (Stmt::child_iterator CI = s->child_begin(), CE = s->child_end();
             CI != CE; ++CI) {
            if (*CI)
                RewriteHostStmt(*CI);
        }
        //Visit this node
        if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(s)) {
            RewriteCUDAKernelCall(kce);
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(s)) {
            if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0)
                RewriteCUDACall(ce);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(s)) {
            DeclGroupRef DG = ds->getDeclGroup();
            Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            //Store VarDecl DeclGroupRefs
            if (firstDecl->getKind() == Decl::Var) {
                CurVarDeclGroups.insert(DG);
            }
            for(DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    RewriteVarDecl(vd);
                }
                //TODO other non-top level declarations??
            }
        }
        else if (MemberExpr *me = dyn_cast<MemberExpr>(s)) {
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
                    //TODO find location of arrow or dot and rewrite
                    SourceRange sr = me->getSourceRange();
                    HostRewrite.ReplaceText(sr.getBegin(), HostRewrite.getRangeSize(sr), PrintStmtToString(dre) + name);
                }
                else if (type == "cudaDeviceProp") {
                    //TODO check what the reference is
                    //TODO if unsupported, print a warning
                }
            }
        }
    }

    std::string RewriteHostExpr(Expr *e) {
    }

    void RewriteKernelFunction(FunctionDecl *kernelFunc) {
        //llvm::errs() << "Rewriting CUDA kernel\n";
        bool hasHost = kernelFunc->hasAttr<CUDAHostAttr>();
        if (!hasHost) {
            RemoveFunction(kernelFunc, HostRewrite);
        }
        llvm::StringRef r = stem(SM->getFileEntryForID(SM->getFileID(kernelFunc->getLocation()))->getName());
        std::list<llvm::StringRef> &l = Kernels[r];
        l.push_back(kernelFunc->getName());
        HostKernels += "cl_kernel __cu2cl_Kernel_" + kernelFunc->getName().str() + ";\n";
        //Rewrite kernel attributes
        if (kernelFunc->hasAttr<CUDAGlobalAttr>()) {
            SourceLocation loc = FindAttr(kernelFunc->getLocStart(), "__global__");
            RewriteAttr("__global__", loc, KernelRewrite);
            if (hasHost)
                RewriteAttr("__global__", loc, HostRewrite);
        }
        if (kernelFunc->hasAttr<CUDADeviceAttr>()) {
            SourceLocation loc = FindAttr(kernelFunc->getLocStart(), "__device__");
            RewriteAttr("__device__", loc, KernelRewrite);
            if (hasHost)
                RewriteAttr("__device__", loc, HostRewrite);
        }
        if (kernelFunc->hasAttr<CUDAHostAttr>()) {
            SourceLocation loc = FindAttr(kernelFunc->getLocStart(), "__host__");
            RewriteAttr("__host__", loc, KernelRewrite);
            if (hasHost)
                RewriteAttr("__host__", loc, HostRewrite);
        }
        //Rewrite arguments
        for (FunctionDecl::param_iterator PI = kernelFunc->param_begin(),
                                          PE = kernelFunc->param_end();
                                          PI != PE; ++PI) {
            RewriteKernelParam(*PI);
        }
        //Rewirte kernel body
        if (kernelFunc->hasBody()) {
            //TraverseStmt(kernelFunc->getBody(), 0);
            RewriteKernelStmt(kernelFunc->getBody());
        }
    }

    void RewriteKernelParam(ParmVarDecl *parmDecl) {
        if (parmDecl->hasAttr<CUDADeviceAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__device__");
            RewriteAttr("__device__", loc, KernelRewrite);
        }
        else if (parmDecl->hasAttr<CUDAConstantAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__constant__");
            RewriteAttr("__constant__", loc, KernelRewrite);
        }
        else if (parmDecl->hasAttr<CUDASharedAttr>()) {
            SourceLocation loc = FindAttr(parmDecl->getLocStart(), "__shared__");
            RewriteAttr("__shared__", loc, KernelRewrite);
        }
        else if (parmDecl->getOriginalType().getTypePtr()->isPointerType()) {
            KernelRewrite.InsertTextBefore(
                    parmDecl->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                    "__global ");
        }
    }

    void RewriteKernelStmt(Stmt *ks) {
        //Traverse children and recurse
        for (Stmt::child_iterator CI = ks->child_begin(), CE = ks->child_end();
             CI != CE; ++CI) {
            if (*CI)
                RewriteKernelStmt(*CI);
        }

        //Visit this node
        std::string str;
        if (MemberExpr *me = dyn_cast<MemberExpr>(ks)) {
            //Check base expr, if DeclRefExpr and a dim3, then rewrite
            if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(me->getBase())) {
                std::string type = dre->getDecl()->getType().getAsString();
                if (type == "dim3" || type == "const uint3") {
                    std::string name = dre->getDecl()->getNameAsString();
                    if (name == "threadIdx")
                        str = "get_local_id";
                    else if (name == "blockIdx")
                        str = "get_group_id";
                    else if (name == "blockDim")
                        str = "get_local_size";
                    else if (name == "gridDim")
                        str = "get_num_groups";

                    name = me->getMemberDecl()->getNameAsString();
                    if (name == "x")
                        str += "(0)";
                    else if (name == "y")
                        str += "(1)";
                    else if (name == "z")
                        str += "(2)";
                    //TODO find location of arrow or dot and rewrite
                    ReplaceStmtWithText(ks, str, KernelRewrite);
                }
            }
        }
        /*else if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(ks)) {
        }*/
        else if (CallExpr *ce = dyn_cast<CallExpr>(ks)) {
            str = PrintStmtToString(ce);
            if (str == "__syncthreads()")
                ReplaceStmtWithText(ks, "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", KernelRewrite);
        }
        else if (DeclStmt *ds = dyn_cast<DeclStmt>(ks)) {
            DeclGroupRef DG = ds->getDeclGroup();
            Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
            //Store VarDecl DeclGroupRefs
            if (firstDecl->getKind() == Decl::Var) {
                CurVarDeclGroups.insert(DG);
            }
            for(DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
                if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                    if (vd->hasAttr<CUDADeviceAttr>()) {
                        SourceLocation loc = FindAttr(vd->getLocStart(), "__device__");
                        RewriteAttr("__device__", loc, KernelRewrite);
                    }
                    else if (vd->hasAttr<CUDAConstantAttr>()) {
                        SourceLocation loc = FindAttr(vd->getLocStart(), "__constant__");
                        RewriteAttr("__constant__", loc, KernelRewrite);
                    }
                    else if (vd->hasAttr<CUDASharedAttr>()) {
                        SourceLocation loc = FindAttr(vd->getLocStart(), "__shared__");
                        RewriteAttr("__shared__", loc, KernelRewrite);
                    }
                }
                //TODO other non-top level declarations??
            }
        }
    }

    std::string RewriteKernelExpr(Expr *e) {
    }

    void RewriteCUDACall(CallExpr *cudaCall) {
        //TODO all CUDA calls return a cudaError_t, so need to find a way to keep that working
        //TODO check if the return value is being used somehow?
        //llvm::errs() << "Rewriting CUDA API call\n";
        std::string funcName = cudaCall->getDirectCallee()->getNameAsString();

        //Thread Management
        if (funcName == "cudaThreadExit") {
            //Replace with clReleaseContext
            ReplaceStmtWithText(cudaCall, "clReleaseContext(__cu2cl_Context)", HostRewrite);
        }
        else if (funcName == "cudaThreadSynchronize") {
            //Replace with clFinish
            ReplaceStmtWithText(cudaCall, "clFinish(__cu2cl_CommandQueue)", HostRewrite);
        }

        //Device Management
        else if (funcName == "cudaGetDevice") {
            //Replace by assigning current value of clDevice to arg
            Expr *device = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(device);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            //Rewrite var type to cl_device_id
            //TODO check if type already rewritten
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            HostRewrite.ReplaceText(tl.getBeginLoc(),
                                    HostRewrite.getRangeSize(tl.getSourceRange()),
                                    "cl_device_id");
            ReplaceStmtWithText(cudaCall, var->getNameAsString() + " = __cu2cl_Device", HostRewrite);
        }
        else if (funcName == "cudaGetDeviceCount") {
            //Replace with clGetDeviceIDs
            Expr *count = cudaCall->getArg(0);
            ReplaceStmtWithText(cudaCall, "clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 0, NULL, " + PrintStmtToString(count) + ")", HostRewrite);
        }
        else if (funcName == "cudaSetDevice") {
            //Replace with clCreateContext and clCreateCommandQueue
            Expr *device = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(device);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            std::string sub = "__cu2cl_Context = clCreateContext(NULL, 1, &" + var->getNameAsString() + ", NULL, NULL, NULL);\n";
            sub += "__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, " + var->getNameAsString() +", 0, NULL);\n";
            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }
        else if (funcName == "cudaGetDeviceProperties") {
            //Replace with __cu2cl_GetDeviceProperties
            HostRewrite.ReplaceText(cudaCall->getLocStart(), funcName.length(), "__cu2cl_GetDeviceProperties");
        }

        //Stream Management
        else if (funcName == "cudaStreamCreate") {
            //Replace with clCreateCommandQueue
            Expr *pStream = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(pStream);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            std::string sub = var->getNameAsString() + " = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, 0, NULL)";

            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }
        else if (funcName == "cudaStreamDestroy") {
            //Replace with clReleaseCommandQueue
            Expr *stream = cudaCall->getArg(0);
            std::string sub = "clReleaseCommandQueue(" + PrintStmtToString(stream) + ")";

            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }
        else if (funcName == "cudaStreamQuery") {
            //Replace with __cu2cl_CommandQueueQuery
            if (!UsesCUDAStreamQuery) {
                HostFunctions += CL_COMMAND_QUEUE_QUERY;
                UsesCUDAStreamQuery = true;
            }

            Expr *stream = cudaCall->getArg(0);
            std::string sub = "__cu2cl_CommandQueueQuery(" + PrintStmtToString(stream) + ")";

            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }
        else if (funcName == "cudaStreamSynchronize") {
            //Replace with clFinish
            Expr *stream = cudaCall->getArg(0);
            std::string sub = "clFinish(" + PrintStmtToString(stream) + ")";

            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }
        else if (funcName == "cudaStreamWaitEvent") {
            //Replace with clEnqueueWaitForEvents
            Expr *stream = cudaCall->getArg(0);
            Expr *event = cudaCall->getArg(1);
            std::string sub = "clEnqueueWaitForEvents(" + PrintStmtToString(stream) + ", 1, &" + PrintStmtToString(event) + ")";

            ReplaceStmtWithText(cudaCall, sub, HostRewrite);
        }

        //Memory Management
        else if (funcName == "cudaMalloc") {
            Expr *varExpr = cudaCall->getArg(0);
            Expr *size = cudaCall->getArg(1);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(varExpr);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
            llvm::StringRef varName = var->getName();
            std::string str;
            str = PrintStmtToString(size);

            //Replace with clCreateBuffer
            std::string sub = varName.str() + " = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, " + str + ", NULL, NULL)";
            ReplaceStmtWithText(cudaCall, sub, HostRewrite);

            DeclGroupRef varDG(var);
            if (CurVarDeclGroups.find(varDG) != CurVarDeclGroups.end()) {
                DeviceMemDGs.insert(*CurVarDeclGroups.find(varDG));
            }
            else if (GlobalVarDeclGroups.find(varDG) != GlobalVarDeclGroups.end()) {
                DeviceMemDGs.insert(*GlobalVarDeclGroups.find(varDG));
            }
            else {
                //TODO single decl, so rewrite now as before
                //Change variable's type to cl_mem
                TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
                HostRewrite.ReplaceText(tl.getBeginLoc(),
                                        HostRewrite.getRangeSize(tl.getSourceRange()),
                                        "cl_mem ");
            }

            //Add var to DeviceMemVars
            DeviceMemVars.insert(var);
        }
        else if (funcName == "cudaFree") {
            Expr *devPtr = cudaCall->getArg(0);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(devPtr);
            llvm::StringRef varName = dr->getDecl()->getName();

            //Replace with clReleaseMemObject
            ReplaceStmtWithText(cudaCall, "clReleaseMemObject(" + varName.str() + ")", HostRewrite);
        }
        else if (funcName == "cudaMallocHost") {
            //TODO implement
        }
        else if (funcName == "cudaMemcpy") {
            std::string replace;

            Expr *dst = cudaCall->getArg(0);
            Expr *src = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            Expr *kind = cudaCall->getArg(3);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(kind);
            EnumConstantDecl *enumConst = dyn_cast<EnumConstantDecl>(dr->getDecl());
            std::string enumString = enumConst->getNameAsString();
            //llvm::errs() << enumString << "\n";
            if (enumString == "cudaMemcpyHostToHost") {
                //standard memcpy
                //make sure to include <string.h>
                if (!IncludingStringH) {
                    HostIncludes += "#include <string.h>\n";
                    IncludingStringH = true;
                }
                replace += "memcpy(";
                replace += PrintStmtToString(dst) + ", ";
                replace += PrintStmtToString(src) + ", ";
                replace += PrintStmtToString(count) + ")";
                ReplaceStmtWithText(cudaCall, replace, HostRewrite);
            }
            else if (enumString == "cudaMemcpyHostToDevice") {
                //clEnqueueWriteBuffer
                replace += "clEnqueueWriteBuffer(__cu2cl_CommandQueue, ";
                replace += PrintStmtToString(dst) + ", CL_TRUE, 0, ";
                replace += PrintStmtToString(count) + ", ";
                replace += PrintStmtToString(src) + ", 0, NULL, NULL)";
                ReplaceStmtWithText(cudaCall, replace, HostRewrite);
            }
            else if (enumString == "cudaMemcpyDeviceToHost") {
                //clEnqueueReadBuffer
                replace += "clEnqueueReadBuffer(__cu2cl_CommandQueue, ";
                replace += PrintStmtToString(src) + ", CL_TRUE, 0, ";
                replace += PrintStmtToString(count) + ", ";
                replace += PrintStmtToString(dst) + ", 0, NULL, NULL)";
                ReplaceStmtWithText(cudaCall, replace, HostRewrite);
            }
            else if (enumString == "cudaMemcpyDeviceToDevice") {
                //TODO clEnqueueReadBuffer; clEnqueueWriteBuffer
                ReplaceStmtWithText(cudaCall, "clEnqueueReadBuffer(__cu2cl_CommandQueue, src, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
                ReplaceStmtWithText(cudaCall, "clEnqueueWriteBuffer(__cu2cl_CommandQueue, dst, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
            }
            else {
                //TODO Use diagnostics to print pretty errors
                llvm::errs() << "Unsupported cudaMemcpy type: " << enumString << "\n";
            }
        }
        else if (funcName == "cudaMemset") {
            if (!UsesCUDAMemset) {
                HostFunctions += CL_MEMSET;
                DevFunctions += CL_MEMSET_KERNEL;
                llvm::StringRef r = stem(SM->getFileEntryForID(MainFileID)->getName());
                std::list<llvm::StringRef> &l = Kernels[r];
                l.push_back("__cu2cl_Memset");
                HostKernels += "cl_kernel __cu2cl_Kernel___cu2cl_Memset;\n";
                UsesCUDAMemset = true;
            }
            //TODO follow Swan's example of setting via a kernel

            HostRewrite.ReplaceText(cudaCall->getLocStart(), funcName.length(), "__cu2cl_Memset");
        }
        else {
            //TODO Use diagnostics to print pretty errors
            llvm::errs() << "Unsupported CUDA call: " << funcName << "\n";
        }
    }

    void RewriteCUDAKernelCall(CUDAKernelCallExpr *kernelCall) {
        //llvm::errs() << "Rewriting CUDA kernel call\n";
        FunctionDecl *callee = kernelCall->getDirectCallee();
        CallExpr *kernelConfig = kernelCall->getConfig();
        std::string kernelName = "__cu2cl_Kernel_" + callee->getNameAsString();
        std::ostringstream args;
        unsigned int dims = 1;
        for (unsigned i = 0; i < kernelCall->getNumArgs(); i++) {
            Expr *arg = kernelCall->getArg(i);
            VarDecl *var = dyn_cast<VarDecl>(FindStmt<DeclRefExpr>(arg)->getDecl());
            args << "clSetKernelArg(" << kernelName << ", " << i << ", sizeof(";
            if (DeviceMemVars.find(var) != DeviceMemVars.end()) {
                //arg var is a cl_mem
                args << "cl_mem";
            }
            else {
                args << var->getType().getAsString();
            }
            args << "), &" << var->getNameAsString() << ");\n";
        }
        //Guaranteed to be dim3s, so pull out their x,y,z values
        Expr *grid = kernelConfig->getArg(0);
        Expr *block = kernelConfig->getArg(1);
        //TraverseStmt(grid, 0);
        //TraverseStmt(block, 0);
        CXXConstructExpr *construct = dyn_cast<CXXConstructExpr>(block);
        ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
            //variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                dims = 3;
                for (unsigned int i = 0; i < 3; i++)
                    args << "localWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "];\n";
            }
            else {
                //TODO ??
                args << "localWorkSize[0] = " << PrintStmtToString(dre) << ";\n";
            }
        }
        else {
            //constant passed to block
            Expr *arg = cast->getSubExprAsWritten();
            args << "localWorkSize[0] = " << PrintStmtToString(arg) << ";\n";
        }
        construct = dyn_cast<CXXConstructExpr>(grid);
        cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
            //variable passed
            ValueDecl *value = dre->getDecl();
            std::string type = value->getType().getAsString();
            if (type == "dim3") {
                dims = 3;
                for (unsigned int i = 0; i < 3; i++)
                    args << "globalWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "]*localWorkSize[" << i << "];\n";
            }
            else {
                //TODO ??
                args << "globalWorkSize[0] = (" << PrintStmtToString(dre) << ")*localWorkSize[0];\n";
            }
        }
        else {
            //constant passed to grid
            Expr *arg = cast->getSubExprAsWritten();
            args << "globalWorkSize[0] = (" << PrintStmtToString(arg) << ")*localWorkSize[0];\n";
        }
        args << "clEnqueueNDRangeKernel(__cu2cl_CommandQueue, " << kernelName << ", " << dims << ", NULL, globalWorkSize, localWorkSize, 0, NULL, NULL)";
        ReplaceStmtWithText(kernelCall, args.str(), HostRewrite);
    }

    void RewriteMain(FunctionDecl *mainDecl) {
        MainDecl = mainDecl;
    }

    void RewriteVarDecl(VarDecl *var) {
        std::string type = var->getType().getAsString();
        if (type == "dim3") {
            //Rewrite to size_t array
            //TODO rewrite statement as a whole with a new string
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            HostRewrite.ReplaceText(tl.getBeginLoc(),
                                HostRewrite.getRangeSize(tl.getSourceRange()),
                                "size_t");
            if (var->hasInit()) {
                //Rewrite initial value
                CXXConstructExpr *ce = (CXXConstructExpr *) var->getInit();
                std::string args = " = {";
                for (CXXConstructExpr::arg_iterator i = ce->arg_begin(), e = ce->arg_end();
                     i != e; ++i) {
                    Expr *arg = *i;
                    //TODO replace these prints with Rewrite{Host,Kernel}Expr
                    if (CXXDefaultArgExpr *defArg = dyn_cast<CXXDefaultArgExpr>(arg)) {
                        args += PrintStmtToString(defArg->getExpr());
                    }
                    else {
                        args += PrintStmtToString(arg);
                    }
                    if (i + 1 != e)
                        args += ", ";
                }
                args += "}";
                SourceRange parenRange = ce->getParenRange();
                if (parenRange.isValid()) {
                    HostRewrite.ReplaceText(parenRange.getBegin(),
                                        HostRewrite.getRangeSize(parenRange),
                                        args);
                    HostRewrite.InsertTextBefore(parenRange.getBegin(), "[3]");
                }
                else {
                    HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(var->getLocEnd()), "[3]");
                    HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(var->getLocEnd()),
                                            args);
                }
            }
        }
        else if (type == "struct cudaDeviceProp") {
            if (!UsesCUDADeviceProp) {
                HostDecls += CL_DEVICE_PROP;
                HostFunctions += CL_GET_DEVICE_PROPS;
                UsesCUDADeviceProp = true;
            }
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            HostRewrite.ReplaceText(tl.getBeginLoc(),
                                    HostRewrite.getRangeSize(tl.getSourceRange()),
                                    "__cu2cl_DeviceProp");
        }
        else if (type == "cudaStream_t") {
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            HostRewrite.ReplaceText(tl.getBeginLoc(),
                                    HostRewrite.getRangeSize(tl.getSourceRange()),
                                    "cl_command_queue");
        }
        //TODO check other CUDA-only types to rewrite
    }

    void RewriteAttr(std::string attr, SourceLocation loc, Rewriter &Rewrite) {
        std::string replace;
        if (attr == "__global__") {
            replace = "__kernel";
        }
        else if (attr == "__device__") {
            replace = "";
        }
        else if (attr == "__host__") {
            replace = "";
        }
        else if (attr == "__constant__") {
            replace = "__constant";
        }
        else if (attr == "__shared__") {
            replace = "__local";
        }
        Rewrite.ReplaceText(loc, attr.length(), replace);
    }

    SourceLocation FindAttr(SourceLocation loc, std::string attr) {
        //TODO inefficient... find optimizations
        const char *s = attr.c_str();
        size_t len = attr.length();
        FileID fileID = SM->getFileID(loc);
        SourceLocation locStart = SM->getLocForStartOfFile(fileID);
        llvm::StringRef fileBuf = SM->getBufferData(fileID);
        const char *fileBufStart = fileBuf.begin();
        for (const char *bufPtr = fileBufStart + SM->getFileOffset(loc);
             bufPtr >= fileBufStart; bufPtr--) {
            if (strncmp(bufPtr, s, len) == 0)
                return locStart.getFileLocWithOffset(bufPtr - fileBufStart);
        }
        return SourceLocation();
    }

    void RemoveFunction(FunctionDecl *func, Rewriter &Rewrite) {
        SourceLocation startLoc, endLoc;
        //Find startLoc
        if (func->hasAttrs()) {
            Attr *attr = (func->getAttrs())[0];
            std::string attrStr;
            switch (attr->getKind()) {
                case attr::CUDAGlobal:
                    attrStr = "__global__";
                    break;
                case attr::CUDADevice:
                    attrStr = "__device__";
                    break;
                case attr::CUDAHost:
                    attrStr = "__host__";
                    break;
                default:
                    break;
            }
            startLoc = FindAttr(func->getLocStart(), attrStr);
        }
        else {
            //TODO find first specifier location
            startLoc = func->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        }
        //Find endLoc
        if (func->hasBody()) {
            CompoundStmt *body = (CompoundStmt *) func->getBody();
            endLoc = body->getRBracLoc();
        }
        else {
            //Find location of semi-colon
            endLoc = func->getSourceRange().getEnd();
        }
        Rewrite.RemoveText(startLoc,
                           Rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    void RemoveVar(VarDecl *var, Rewriter &Rewrite) {
        SourceLocation startLoc, endLoc;
        //Find startLoc
        if (var->hasAttrs()) {
            Attr *attr = (var->getAttrs())[0];
            std::string attrStr;
            switch (attr->getKind()) {
                case attr::CUDAGlobal:
                    attrStr = "__constant__";
                    break;
                case attr::CUDADevice:
                    attrStr = "__device__";
                    break;
                case attr::CUDAHost:
                    attrStr = "__shared__";
                    break;
                default:
                    break;
            }
            startLoc = FindAttr(var->getLocStart(), attrStr);
        }
        else {
            //TODO find first specifier location
            startLoc = var->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        }
        //Find endLoc
        if (var->hasInit()) {
            Expr *init = var->getInit();
            endLoc = init->getLocEnd();;
        }
        else {
            //Find location of semi-colon
            endLoc = var->getSourceRange().getEnd();
        }
        Rewrite.RemoveText(startLoc,
                           Rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    std::string PrintStmtToString(Stmt *s) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        s->printPretty(S, 0, PrintingPolicy(HostRewrite.getLangOpts()));
        return S.str();
    }

    std::string PrintDeclToString(Decl *d) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        d->print(S);
        return S.str();
    }

    bool ReplaceStmtWithText(Stmt *OldStmt, llvm::StringRef NewStr, Rewriter &Rewrite) {
        return Rewrite.ReplaceText(OldStmt->getLocStart(),
                                   Rewrite.getRangeSize(OldStmt->getSourceRange()),
                                   NewStr);
    }

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
        PP = &CI->getPreprocessor();

        PP->addPPCallbacks(new RewriteIncludesCallback(this));

        HostRewrite.setSourceMgr(Context.getSourceManager(), Context.getLangOptions());
        KernelRewrite.setSourceMgr(Context.getSourceManager(), Context.getLangOptions());
        MainFileID = Context.getSourceManager().getMainFileID();

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
    }

    virtual void HandleTopLevelDecl(DeclGroupRef DG) {
        //Check where the declaration(s) comes from (may have been included)
        Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
        SourceLocation loc = firstDecl->getLocation();
        if (!SM->isFromMainFile(loc)) {
            if (strstr(SM->getPresumedLoc(loc).getFilename(), ".cu") != NULL) {
                if (OutFiles.find(SM->getFileID(loc)) == OutFiles.end()) {
                    //Create new files
                    FileID fileid = SM->getFileID(loc);
                    std::string filename = SM->getPresumedLoc(loc).getFilename();
                    size_t dotPos = filename.rfind('.');
                    filename = filename.substr(0, dotPos) + "-cl" + filename.substr(dotPos);
                    llvm::raw_ostream *hostOS = CI->createDefaultOutputFile(false, filename, "h");
                    llvm::raw_ostream *kernelOS = CI->createDefaultOutputFile(false, filename, "cl");
                    if (hostOS && kernelOS) {
                        OutFiles[fileid] = hostOS;
                        KernelOutFiles[fileid] = kernelOS;
                    }
                    else {
                        //TODO report and print error
                    }
                }
            }
            else {
                return;
            }
        }
        //Store VarDecl DeclGroupRefs
        if (firstDecl->getKind() == Decl::Var) {
            GlobalVarDeclGroups.insert(DG);
        }
        //Walk declarations in group and rewrite
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
                if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    RewriteKernelFunction(fd);
                }
                else {
                    RewriteHostFunction(fd);
                }
                if (fd->getNameAsString() == MainFuncName) {
                    RewriteMain(fd);
                }
            }
            else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                RemoveVar(vd, KernelRewrite);
                RewriteVarDecl(vd);
            }
        }
    }

    virtual void HandleTranslationUnit(ASTContext &) {
        //Declare global clPrograms
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string r = idCharFilter((*i).first);
            HostGlobalVars += "cl_program __cu2cl_Program_" + r + ";\n";
        }
        //Insert host preamble at top of main file
        HostPreamble = HostIncludes + "\n" + HostDecls + "\n" + HostGlobalVars + "\n" + HostKernels + "\n" + HostFunctions;
        HostRewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), HostPreamble);
        //Insert device preamble at top of main kernel file
        DevPreamble = DevFunctions;
        KernelRewrite.InsertTextBefore(SM->getLocForStartOfFile(MainFileID), DevPreamble);

        CompoundStmt *mainBody = dyn_cast<CompoundStmt>(MainDecl->getBody());
        //Insert OpenCL initialization stuff at top of main
        CLInit += "\n";
        CLInit += "const char *progSrc;\n";
        CLInit += "size_t progLen;\n\n";
        CLInit += "clGetPlatformIDs(1, &__cu2cl_Platform, NULL);\n";
        CLInit += "clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 1, &__cu2cl_Device, NULL);\n";
        CLInit += "__cu2cl_Context = clCreateContext(NULL, 1, &__cu2cl_Device, NULL, NULL, NULL);\n";
        CLInit += "__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, 0, NULL);\n";
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string r = idCharFilter((*i).first);
            CLInit += "progLen = __cu2cl_loadProgramSource(\"" + (*i).first.str() + "-cl.cl\", &progSrc);\n";
            CLInit += "__cu2cl_Program_" + r + " = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);\n";
            CLInit += "free((void *) progSrc);\n";
            CLInit += "clBuildProgram(__cu2cl_Program_" + r + ", 1, &__cu2cl_Device, \"-I .\", NULL, NULL);\n";
        }
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string r = idCharFilter((*i).first);
                std::string kernelName = (*li).str();
                CLInit += "__cu2cl_Kernel_" + kernelName + " = clCreateKernel(__cu2cl_Program_" + r + ", \"" + kernelName + "\", NULL);\n";
            }
        }
        HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(mainBody->getLBracLoc()), CLInit);

        //Insert cleanup code at bottom of main
        CLClean += "\n";
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLClean += "clReleaseKernel(__cu2cl_Kernel_" + kernelName + ");\n";
            }
        }
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string r = idCharFilter((*i).first);
            CLClean += "clReleaseProgram(__cu2cl_Program_" + r + ");\n";
        }
        CLClean += "clReleaseCommandQueue(__cu2cl_CommandQueue);\n";
        CLClean += "clReleaseContext(__cu2cl_Context);\n";
        HostRewrite.InsertTextBefore(mainBody->getRBracLoc(), CLClean);

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
                if ((iDG + 1) == DG.end()) {
                    end = (*iDG)->getLocEnd();
                }
                if (DeviceMemVars.find(vd) != DeviceMemVars.end()) {
                    //Change variable's type to cl_mem
                    replace += "cl_mem " + vd->getNameAsString() + ";\n";
                }
                else {
                    replace += PrintDeclToString(vd) + ";\n";
                }
            }
            HostRewrite.ReplaceText(start, HostRewrite.getRangeSize(SourceRange(start, end)), replace);
        }

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

        for (IDOutFileMap::iterator i = OutFiles.begin(), e = OutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
            if (const RewriteBuffer *RewriteBuff =
                HostRewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                //TODO use diagnostics for pretty errors
                llvm::errs() << "No changes made to " << SM->getFileEntryForID(fid)->getName() << "\n";
            }
            outFile->flush();
        }
        for (IDOutFileMap::iterator i = KernelOutFiles.begin(), e = KernelOutFiles.end();
             i != e; i++) {
            FileID fid = (*i).first;
            llvm::raw_ostream *outFile = (*i).second;
            if (const RewriteBuffer *RewriteBuff =
                KernelRewrite.getRewriteBufferFor(fid)) {
                *outFile << std::string(RewriteBuff->begin(), RewriteBuff->end());
            }
            else {
                //TODO use diagnostics for pretty errors
                llvm::errs() << "No changes made to " << SM->getFileEntryForID(fid)->getName() << " kernel\n";
            }
            outFile->flush();
        }
    }

    void RewriteInclude(SourceLocation HashLoc, const Token &IncludeTok,
                        llvm::StringRef FileName, bool IsAngled,
                        const FileEntry *File, SourceLocation EndLoc/*,
                        const llvm::SmallVectorImpl<char> &RawPath*/) {
        if (SM->isFromMainFile(HashLoc) ||
            extension(SM->getPresumedLoc(HashLoc).getFilename()).str() == ".cu") {
            if (IsAngled) {
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                if (filename(FileName).str() == "cuda.h")
                    HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
            else if (filename(FileName).str() == "cuda.h") {
                HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
            else if (extension(FileName).str() == ".cu") {
                FileID fileID = SM->getFileID(HashLoc);
                SourceLocation fileStartLoc = SM->getLocForStartOfFile(fileID);
                llvm::StringRef fileBuf = SM->getBufferData(fileID);
                const char *fileBufStart = fileBuf.begin();

                llvm::StringRef ext = extension(FileName);
                SourceLocation start = fileStartLoc.getFileLocWithOffset(ext.begin() - fileBufStart);
                SourceLocation end = fileStartLoc.getFileLocWithOffset((ext.end()-1) - fileBufStart);
                HostRewrite.ReplaceText(start, HostRewrite.getRangeSize(SourceRange(start, end)), "-cl.h");
                KernelRewrite.ReplaceText(start, KernelRewrite.getRangeSize(SourceRange(start, end)), "-cl.cl");
            }
            else {
                //TODO store include info to rewrite later?
            }
        }
    }

};

class RewriteCUDAAction : public PluginASTAction {
protected:
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) {
        std::string filename = InFile.str();
        size_t dotPos = filename.rfind('.');
        filename = filename.substr(0, dotPos) + "-cl" + filename.substr(dotPos);
        //std::string newName = stem(InFile).str() + "-cl" + extension(InFile).str();
        llvm::raw_ostream *hostOS = CI.createDefaultOutputFile(false, filename, "cpp");
        llvm::raw_ostream *kernelOS = CI.createDefaultOutputFile(false, filename, "cl");
        if (hostOS && kernelOS)
            return new RewriteCUDA(&CI, hostOS, kernelOS);
        //TODO cleanup files?
        return NULL;
    }

    bool ParseArgs(const CompilerInstance &CI,
                   const std::vector<std::string> &args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
            llvm::errs() << "RewriteCUDA arg = " << args[i] << "\n";
            //TODO parse arguments
        }
        if (args.size() && args[0] == "help")
            PrintHelp(llvm::errs());

        return true;
    }

    void PrintHelp(llvm::raw_ostream &ros) {
        ros << "Help for RewriteCUDA plugin goes here\n";
    }

};

RewriteIncludesCallback::RewriteIncludesCallback(RewriteCUDA *RC) :
    RCUDA(RC) {
}

void RewriteIncludesCallback::InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                                                 llvm::StringRef FileName, bool IsAngled,
                                                 const FileEntry *File, SourceLocation EndLoc/*,
                                                 const llvm::SmallVectorImpl<char> &RawPath*/) {
    RCUDA->RewriteInclude(HashLoc, IncludeTok, FileName, IsAngled, File, EndLoc);
}

}

static FrontendPluginRegistry::Add<RewriteCUDAAction>
X("rewrite-cuda", "translate CUDA to OpenCL");
