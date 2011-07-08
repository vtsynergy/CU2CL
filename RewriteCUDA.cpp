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
#include "llvm/Support/Regex.h"

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

#define CL_EVENT_QUERY \
    "cl_int __cu2cl_EventQuery(cl_event event) {\n" \
    "    cl_int ret;\n" \
    "    clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &ret, NULL);\n" \
    "    return ret;\n" \
    "}\n\n"

#define CL_MALLOC_HOST \
    "cl_int __cu2cl_MallocHost(void **ptr, size_t size, cl_mem *clMem) {\n" \
    "    cl_int ret;\n" \
    "    *clMem = clCreateBuffer(__cu2cl_Context, CL_MEM_READ_WRITE, size, NULL, NULL);\n" \
    "    *ptr = clEnqueueMapBuffer(__cu2cl_CommandQueue, *clMem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);\n" \
    "    return ret;\n" \
    "}\n\n"

#define CL_FREE_HOST \
    "cl_int __cu2cl_FreeHost(void *ptr, cl_mem clMem) {\n" \
    "    cl_int ret;\n" \
    "    ret = clEnqueueUnmapMemObject(__cu2cl_CommandQueue, clMem, ptr, 0, NULL, NULL);\n" \
    "    ret |= clReleaseMemObject(clMem);\n" \
    "    return ret;\n" \
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
    std::set<VarDecl *> DeviceMemVars;
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

    bool RewriteHostExpr(Expr *e, std::string &newExpr) {
        //Return value specifies whether or not a rewrite occurred
        if (e->getSourceRange().isInvalid())
            return false;

        //Rewriter used for rewriting subexpressions
        Rewriter exprRewriter(*SM, *LO);
        //Instantiation locations are used to capture macros
        SourceRange realRange(SM->getInstantiationLoc(e->getLocStart()),
                              SM->getInstantiationLoc(e->getLocEnd()));

        if (CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(e)) {
            newExpr = RewriteCUDAKernelCall(kce);
            return true;
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(e)) {
            //TODO fix case where non-API call starts with "cuda"
            if (ce->getDirectCallee()->getNameAsString().find("cuda") == 0)
                return RewriteCUDACall(ce, newExpr);
        }
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
                    newExpr = PrintStmtToString(dre) + name;
                    return true;
                }
                else if (type == "cudaDeviceProp") {
                    //TODO check what the reference is
                    //TODO if unsupported, print a warning

                    return false;
                }
            }
        }
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
        else if (SizeOfAlignOfExpr *soe = dyn_cast<SizeOfAlignOfExpr>(e)) {
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
                newExpr = exprRewriter.getRewrittenText(realRange);
                return ret;
            }
        }
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
                    newExpr = exprRewriter.getRewrittenText(realRange);
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
        newExpr = exprRewriter.getRewrittenText(realRange);
        return ret;
    }

    bool RewriteCUDACall(CallExpr *cudaCall, std::string &newExpr) {
        //TODO all CUDA calls return a cudaError_t, so need to find a way to keep that working
        //TODO check if the return value is being used somehow?
        //llvm::errs() << "Rewriting CUDA API call\n";
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
            Expr *device = cudaCall->getArg(0);
            std::string newDevice;
            RewriteHostExpr(device, newDevice);
            DeclRefExpr *dr = FindStmt<DeclRefExpr>(device);
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());

            //Rewrite var type to cl_device_id
            //TODO properly rewrite and check if type already rewritten
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            RewriteType(tl, "cl_device_id", HostRewrite);
            newExpr = "*" + newDevice + " = __cu2cl_Device";
        }
        else if (funcName == "cudaGetDeviceCount") {
            //Replace with clGetDeviceIDs
            Expr *count = cudaCall->getArg(0);
            std::string newCount;
            RewriteHostExpr(count, newCount);
            newExpr = "clGetDeviceIDs(__cu2cl_Platform, CL_DEVICE_TYPE_GPU, 0, NULL, (cl_uint *) " + newCount + ")";
        }
        else if (funcName == "cudaSetDevice") {
            //Replace with clCreateContext and clCreateCommandQueue
            //TODO first, delete old queue and context
            Expr *device = cudaCall->getArg(0);
            DeclRefExpr *dre = FindStmt<DeclRefExpr>(device);
            if (dre != NULL) {
                std::string newDevice;
                RewriteHostExpr(device, newDevice);
                //TODO also rewrite type as in cudaGetDevice
                //VarDecl *var = dyn_cast<VarDecl>(dre->getDecl());
                newExpr = "__cu2cl_Context = clCreateContext(NULL, 1, &" + newDevice + ", NULL, NULL, NULL);\n";
                newExpr += "__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, " + newDevice + ", CL_QUEUE_PROFILING_ENABLE, NULL)\n";
            }
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
            //Remove the call
            newExpr = "";
        }
        else if (funcName == "cudaEventCreateWithFlags") {
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
            VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());

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
                //TODO single decl, so rewrite now as before
                //TODO check the type, if pointertype, rewrite as you have already
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
        else if (funcName == "cudaMemcpy") {
            //TODO support offsets
            //Inspect kind of memcpy and rewrite accordingly
            Expr *dst = cudaCall->getArg(0);
            Expr *src = cudaCall->getArg(1);
            Expr *count = cudaCall->getArg(2);
            Expr *kind = cudaCall->getArg(3);
            std::string newDst, newSrc, newCount;
            RewriteHostExpr(dst, newDst);
            RewriteHostExpr(src, newSrc);
            RewriteHostExpr(count, newCount);

            //TODO simply dyn_cast to the DeclRefExpr here?
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
                //TODO implement __cu2cl_MemcpyDevToDev
                ReplaceStmtWithText(cudaCall, "clEnqueueReadBuffer(__cu2cl_CommandQueue, src, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
                ReplaceStmtWithText(cudaCall, "clEnqueueWriteBuffer(__cu2cl_CommandQueue, dst, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
            }
            else {
                //TODO Use diagnostics to print pretty errors
                llvm::errs() << "Unsupported cudaMemcpyKind: " << enumString << "\n";
            }
        }
        else if (funcName == "cudaMemcpyAsync") {
            //TODO support offsets
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

            //TODO simply dyn_cast to the DeclRefExpr here?
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
                //TODO figure out if you need the cl_mems of HostMemVars
                //clEnqueueWriteBuffer, src is HostMemVar
                dr = FindStmt<DeclRefExpr>(src);
                VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
                llvm::StringRef varName = var->getName();
                newExpr = "clEnqueueWriteBuffer(" + newStream + ", " + newDst + ", CL_FALSE, 0, " + newCount + ", __cu2cl_Mem_" + varName.str() + ", 0, NULL, NULL)";
            }
            else if (enumString == "cudaMemcpyDeviceToHost") {
                //TODO figure out if you need the cl_mems of HostMemVars
                //clEnqueueReadBuffer, dst is HostMemVar
                dr = FindStmt<DeclRefExpr>(dst);
                VarDecl *var = dyn_cast<VarDecl>(dr->getDecl());
                llvm::StringRef varName = var->getName();
                newExpr = "clEnqueueReadBuffer(" + newStream + ", " + newSrc + ", CL_FALSE, 0, " + newCount + ", __cu2cl_Mem_" + varName.str() + ", 0, NULL, NULL)";
            }
            else if (enumString == "cudaMemcpyDeviceToDevice") {
                //TODO implement __cu2cl_MemcpyDevToDev
                ReplaceStmtWithText(cudaCall, "clEnqueueReadBuffer(__cu2cl_CommandQueue, src, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
                ReplaceStmtWithText(cudaCall, "clEnqueueWriteBuffer(__cu2cl_CommandQueue, dst, CL_TRUE, 0, count, temp, 0, NULL, NULL)", HostRewrite);
            }
            else {
                //TODO Use diagnostics to print pretty errors
                llvm::errs() << "Unsupported cudaMemcpyKind: " << enumString << "\n";
            }
        }
        else if (funcName == "cudaMemcpyToSymbol") {
            //TODO implement
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
            //TODO Use diagnostics to print pretty errors
            llvm::errs() << "Unsupported CUDA call: " << funcName << "\n";
            return false;
        }
        return true;
    }

    std::string RewriteCUDAKernelCall(CUDAKernelCallExpr *kernelCall) {
        //llvm::errs() << "Rewriting CUDA kernel call\n";
        FunctionDecl *callee = kernelCall->getDirectCallee();
        CallExpr *kernelConfig = kernelCall->getConfig();
        std::string kernelName = "__cu2cl_Kernel_" + callee->getNameAsString();
        std::ostringstream args;
        unsigned int dims = 1;

        //Set kernel arguments
        for (unsigned i = 0; i < kernelCall->getNumArgs(); i++) {
            Expr *arg = kernelCall->getArg(i);
            std::string newArg;
            RewriteHostExpr(arg, newArg);
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

        //TODO add constant memory arguments

        //TODO handle passing in a new dim3? (i.e. dim3(1,2,3))
        //Set work sizes
        //Guaranteed to be dim3s, so pull out their x,y,z values
        Expr *grid = kernelConfig->getArg(0);
        Expr *block = kernelConfig->getArg(1);
        CXXConstructExpr *construct = dyn_cast<CXXConstructExpr>(block);
        ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
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
                args << "localWorkSize[0] = " << PrintStmtToString(dre) << ";\n";
            }
        }
        else {
            //Some other expression passed to block
            Expr *arg = cast->getSubExprAsWritten();
            std::string s;
            RewriteHostExpr(arg, s);
            args << "localWorkSize[0] = " << s << ";\n";
        }

        construct = dyn_cast<CXXConstructExpr>(grid);
        cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));
        if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten())) {
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
                args << "globalWorkSize[0] = (" << PrintStmtToString(dre) << ")*localWorkSize[0];\n";
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
            //Handle __constant__ memory declarations
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
                    //TODO fix case of dim3 c = b;
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
                    else
                        ReplaceStmtWithText(e, s, HostRewrite);

                    //Add [3]/* to end/start of var identifier
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

    void RewriteMain(FunctionDecl *mainDecl) {
        MainDecl = mainDecl;
    }

    void RewriteKernelFunction(FunctionDecl *kernelFunc) {
        if (kernelFunc->hasAttr<CUDAGlobalAttr>()) {
            //If host-callable, get and store kernel filename
            llvm::StringRef r = stem(SM->getFileEntryForID(SM->getFileID(kernelFunc->getLocation()))->getName());
            std::list<llvm::StringRef> &l = Kernels[r];
            l.push_back(kernelFunc->getName());
            HostKernels += "cl_kernel __cu2cl_Kernel_" + kernelFunc->getName().str() + ";\n";
        }

        //Rewrite kernel attributes
        if (CUDAGlobalAttr *attr = kernelFunc->getAttr<CUDAGlobalAttr>()) {
            RewriteAttr(attr, "__kernel", KernelRewrite);
        }
        if (CUDADeviceAttr *attr = kernelFunc->getAttr<CUDADeviceAttr>()) {
            RewriteAttr(attr, "", KernelRewrite);
        }
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

    void RewriteKernelParam(ParmVarDecl *parmDecl, bool isFuncGlobal) {
        TypeLoc tl = parmDecl->getTypeSourceInfo()->getTypeLoc();
        if (isFuncGlobal && tl.getTypePtr()->isPointerType()) {
            KernelRewrite.InsertTextBefore(
                    tl.getBeginLoc(),
                    "__global ");
        }
        else if (ReferenceTypeLoc *rtl = dyn_cast<ReferenceTypeLoc>(&tl)) {
            KernelRewrite.ReplaceText(
                    rtl->getSigilLoc(),
                    KernelRewrite.getRangeSize(rtl->getLocalSourceRange()),
                    "*");
            CurRefParmVars.insert(parmDecl);
        }

        while (!tl.getNextTypeLoc().isNull()) {
            tl = tl.getNextTypeLoc();
        }
        QualType qt = tl.getType();
        std::string type = qt.getAsString();

        std::string newType = RewriteVectorType(type, false);
        if (newType != "")
            RewriteType(tl, newType, KernelRewrite);
    }

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
        if (e->getSourceRange().isInvalid())
            return false;

        //Rewriter used for rewriting subexpressions
        Rewriter exprRewriter(*SM, *LO);
        //Instantiation locations are used to capture macros
        SourceRange realRange(SM->getInstantiationLoc(e->getLocStart()),
                              SM->getInstantiationLoc(e->getLocEnd()));

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
                        newExpr = PrintStmtToString(dre);

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
                        newExpr = PrintStmtToString(dre);

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
                    //TODO check bug that happens when an inserted character
                    //ends the new range
                    newExpr = "(*" + exprRewriter.getRewrittenText(realRange) + ")";
                    return true;
                }
            }
        }
        else if (CallExpr *ce = dyn_cast<CallExpr>(e)) {
            std::string funcName = ce->getDirectCallee()->getNameAsString();
            if (funcName == "__syncthreads") {
                newExpr = "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)";
            }
            else if (funcName == "sqrtf") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "sqrt(" + newX + ")";
            }
            else if (funcName == "__log2f") {
                Expr *x = ce->getArg(0);
                std::string newX;
                RewriteKernelExpr(x, newX);
                newExpr = "native_log2(" + newX + ")";
            }
            else if (funcName == "__powf") {
                Expr *x = ce->getArg(0);
                Expr *y = ce->getArg(1);
                std::string newX, newY;
                RewriteKernelExpr(x, newX);
                RewriteKernelExpr(y, newY);
                newExpr = "native_powr(" + newX + ", " + newY + ")";
            }
            else {
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
        //TODO handle __shared__ memory
        if (CUDASharedAttr *sharedAttr = var->getAttr<CUDASharedAttr>()) {
            RewriteAttr(sharedAttr, "__local", KernelRewrite);
            if (CUDADeviceAttr *devAttr = var->getAttr<CUDADeviceAttr>())
                RewriteAttr(devAttr, "", KernelRewrite);
            //TODO rewrite shared mem
            //if (var->isExtern())
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
        if (size == '3')
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

    void RewriteType(TypeLoc tl, std::string replace, Rewriter &rewrite) {
        rewrite.ReplaceText(
                tl.getBeginLoc(),
                rewrite.getRangeSize(tl.getLocalSourceRange()),
                replace);
    }

    void RewriteAttr(Attr *attr, std::string replace, Rewriter &rewrite) {
        SourceLocation instLoc = SM->getInstantiationLoc(attr->getLocation());
        SourceRange realRange(instLoc,
                              PP->getLocForEndOfToken(instLoc));
        rewrite.ReplaceText(instLoc, rewrite.getRangeSize(realRange), replace);
    }

    void RemoveFunction(FunctionDecl *func, Rewriter &rewrite) {
        SourceLocation startLoc, endLoc, tempLoc;

        FunctionDecl::TemplatedKind tk = func->getTemplatedKind();
        if (tk != FunctionDecl::TK_NonTemplate &&
            tk != FunctionDecl::TK_FunctionTemplate)
            return;

        //Find startLoc
        //TODO find first specifier location
        startLoc = func->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        if (tk == FunctionDecl::TK_FunctionTemplate) {
            FunctionTemplateDecl *ftd = func->getDescribedFunctionTemplate();
            tempLoc = ftd->getSourceRange().getBegin();
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }
        if (func->hasAttrs()) {
            Attr *attr = (func->getAttrs())[0];
            tempLoc = SM->getInstantiationLoc(attr->getLocation());
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
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
        rewrite.RemoveText(startLoc,
                           rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    void RemoveVar(VarDecl *var, Rewriter &rewrite) {
        SourceLocation startLoc, endLoc, tempLoc;

        //Find startLoc
        //TODO find first specifier location
        startLoc = var->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
        if (var->hasAttrs()) {
            Attr *attr = (var->getAttrs())[0];
            tempLoc = SM->getInstantiationLoc(attr->getLocation());
            if (SM->isBeforeInTranslationUnit(tempLoc, startLoc))
                startLoc = tempLoc;
        }

        //Find endLoc
        if (var->hasInit()) {
            Expr *init = var->getInit();
            endLoc = init->getLocEnd();;
        }
        else {
            //Find location of semi-colon
            TypeLoc tl = var->getTypeSourceInfo()->getTypeLoc();
            if (ArrayTypeLoc *atl = dyn_cast<ArrayTypeLoc>(&tl)) {
                endLoc = atl->getRBracketLoc();
            }
            else
                endLoc = var->getSourceRange().getEnd();
        }
        rewrite.RemoveText(startLoc,
                           rewrite.getRangeSize(SourceRange(startLoc, endLoc)));
    }

    std::string PrintStmtToString(Stmt *s) {
        std::string SStr;
        llvm::raw_string_ostream S(SStr);
        s->printPretty(S, 0, PrintingPolicy(*LO));
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
    }

    virtual void HandleTopLevelDecl(DeclGroupRef DG) {
        //Check where the declaration(s) comes from (may have been included)
        Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
        SourceLocation loc = firstDecl->getLocation();
        if (!SM->isFromMainFile(loc)) {
            llvm::StringRef fileExt = extension(SM->getPresumedLoc(loc).getFilename());
            if (fileExt.equals(".cu") || fileExt.equals(".cuh")) {
                //If #included and a .cu or .cuh file, rewrite
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
            if (DeclContext *dc = dyn_cast<DeclContext>(*i)) {
                for (DeclContext::decl_iterator di = dc->decls_begin(), de = dc->decls_end();
                     di != de; ++di) {
                    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*di)) {
                        RewriteHostFunction(fd);
                        RemoveFunction(fd, KernelRewrite);

                        if (fd->getNameAsString() == MainFuncName) {
                            RewriteMain(fd);
                        }
                    }
                }
            }
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
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
            }
            else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
                RemoveVar(vd, KernelRewrite);
                RewriteHostVarDecl(vd);
            }
            //TODO rewrite type declarations
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
        CLInit += "__cu2cl_CommandQueue = clCreateCommandQueue(__cu2cl_Context, __cu2cl_Device, CL_QUEUE_PROFILING_ENABLE, NULL);\n";
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::string file = idCharFilter((*i).first);
            std::list<llvm::StringRef> &l = (*i).second;
            CLInit += "progLen = __cu2cl_LoadProgramSource(\"" + (*i).first.str() + "-cl.cl\", &progSrc);\n";
            CLInit += "__cu2cl_Program_" + file + " = clCreateProgramWithSource(__cu2cl_Context, 1, &progSrc, &progLen, NULL);\n";
            CLInit += "free((void *) progSrc);\n";
            CLInit += "clBuildProgram(__cu2cl_Program_" + file + ", 1, &__cu2cl_Device, \"-I .\", NULL, NULL);\n";
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLInit += "__cu2cl_Kernel_" + kernelName + " = clCreateKernel(__cu2cl_Program_" + file + ", \"" + kernelName + "\", NULL);\n";
            }
        }
        HostRewrite.InsertTextAfter(PP->getLocForEndOfToken(mainBody->getLBracLoc()), CLInit);

        //Insert cleanup code at bottom of main
        CLClean += "\n";
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
            std::list<llvm::StringRef> &l = (*i).second;
            std::string file = idCharFilter((*i).first);
            for (std::list<llvm::StringRef>::iterator li = l.begin(), le = l.end();
                 li != le; li++) {
                std::string kernelName = (*li).str();
                CLClean += "clReleaseKernel(__cu2cl_Kernel_" + kernelName + ");\n";
            }
            CLClean += "clReleaseProgram(__cu2cl_Program_" + file + ");\n";
        }
        for (StringRefListMap::iterator i = Kernels.begin(),
             e = Kernels.end(); i != e; i++) {
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
                if (DeviceMemVars.find(vd) != DeviceMemVars.end()) {
                    //Change variable's type to cl_mem
                    replace += "cl_mem " + vd->getNameAsString();
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
        llvm::StringRef fileExt = extension(SM->getPresumedLoc(HashLoc).getFilename());
        llvm::StringRef includedFile = filename(FileName);
        llvm::StringRef includedExt = extension(includedFile);
        if (SM->isFromMainFile(HashLoc) ||
            fileExt.equals(".cu") || fileExt.equals(".cuh")) {
            if (IsAngled) {
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                if (includedFile.equals("cuda.h"))
                    HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
            else if (includedFile.equals("cuda.h")) {
                HostRewrite.RemoveText(HashLoc, HostRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
                KernelRewrite.RemoveText(HashLoc, KernelRewrite.getRangeSize(SourceRange(HashLoc, EndLoc)));
            }
            else if (includedExt.equals(".cu") || includedExt.equals(".cuh")) {
                FileID fileID = SM->getFileID(HashLoc);
                SourceLocation fileStartLoc = SM->getLocForStartOfFile(fileID);
                llvm::StringRef fileBuf = SM->getBufferData(fileID);
                const char *fileBufStart = fileBuf.begin();

                SourceLocation start = fileStartLoc.getFileLocWithOffset(includedExt.begin() - fileBufStart);
                SourceLocation end = fileStartLoc.getFileLocWithOffset((includedExt.end()-1) - fileBufStart);
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
